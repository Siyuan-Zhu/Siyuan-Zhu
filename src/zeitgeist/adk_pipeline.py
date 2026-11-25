"""Google ADK (Agent Development Kit) Multi-Agent Pipeline for Daily Sentiment Synthesis.

Implements a 3-Agent Sequential Architecture designed for GCP Cloud Run + GitHub Actions:
  1. HarvesterAgent: Samples public RSS/Atom feeds across the 4 weighted lenses
     (40% Personal Orbit, 25% Global, 20% Markets, 15% AI/Systems) and computes
     the deterministic VAD + sublinear TF-IDF baseline.
  2. SemanticAnalystAgent: Synthesizes the single composite sentiment score,
     One Word of the Day, One Short Explanation Phrase, and Concrete Visual Anchor
     (supports live `google-genai` / `google-adk` structured output with deterministic
     offline fallback so daily execution stays under 0.05 NIU/month).
  3. VisualDirectorAgent: Routes between Curated Photography, Original Isobaric Blueprint SVG,
     and Imagen 3 Fallback, enforcing the Multimodal Safety & Minimalist Fit Gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

from .collectors import (
    IngestionUnavailable,
    public_collection_health,
    compute_url_hash,
    embed_missing_items,
    extract_one_hop_mentions,
    fetch_all_curated_feeds,
    fetch_instagram_batch,
    fetch_macro_scout_signals,
    sanitize_orbit_roster_state,
)
from .understanding import (
    echo_violation,
    link_candidates,
    link_keywords,
    links_enabled,
    public_keywords,
    ranking_stats,
    realm_counts,
    render_evidence_packet,
    repair_enabled,
    shown_groups,
    source_realm_map,
    understanding_v1_enabled,
)
from .methodology import (
    CachedItem,
    FeedDocument,
    ScoredCluster,
    WeightedOrbitAnalysis,
    analyze_seven_bucket_orbit,
    cluster_and_score,
    default_curated_corpus,
    interim_grouping_enabled,
    is_captionless,
    legacy_top_cluster,
    load_asymmetric_window,
    load_orbit_config,
    score_and_cluster_signals,
    validate_owner_interjection,
)
from .visual_router import (
    PublishedPulse,
    VisualDecision,
    normalize_mood_votes,
    route_and_render_visual,
    select_and_render_pulse_assets,
)

from .spend_guard import SpendBlocked, refused_before_dispatch, sampling_controls_supported
from . import embedding_migration
from . import evidence_archive
from .public_schema import drop_false_fields_enabled
from . import judge_calibration
from . import own_group, run_record
from . import reflector
from .embedding_migration import EmbeddingRun
from .model_choice import GEMINI_4, RunModels, activate, model_choice_line, resolve_run_models
from .orbit_roster import advance_roster_after_batch, apply_winning_cluster_bonus, select_instagram_batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ADKRunTelemetry:
    """Cost and execution telemetry proving adherence to the monthly budget ceiling."""

    timestamp_utc: str
    adk_mode: str  # "google-adk-live" | "deterministic-hybrid-adk"
    model_id: str
    estimated_input_tokens: int | None
    estimated_output_tokens: int | None
    estimated_monthly_cost_index: float | None
    monthly_budget_ceiling_index: float
    within_budget: bool
    spending_status: str = "unknown"
    collection_health: dict = field(default_factory=dict)


GROUPING_LINE = "[grouping] captionless_items=%d"


def _row_to_cached_item(row: Any, idx: int, ref_dt_str: str) -> Optional[CachedItem]:
    """Coerce a raw signal dict or CachedItem into a normalized CachedItem."""
    if isinstance(row, CachedItem):
        return row
    if not isinstance(row, dict):
        return None
    emb_b64 = row.get("embedding_f16") or row.get("embedding_f16_b64")
    url_val = str(row.get("url") or "")
    raw_hash = str(row.get("url_hash") or row.get("url_sha256") or "").strip()
    if not raw_hash:
        raw_hash = compute_url_hash(url_val if url_val else f"cache_{idx}")
    weight_val = row.get("orbit_weight", row.get("weight"))
    handles = row.get("mentioned_handles")
    extra = {"mentioned_handles": list(handles)} if isinstance(handles, (list, tuple)) and handles else {}
    if isinstance(row.get("text_missing"), bool):
        extra["text_missing"] = row["text_missing"]
    # Scoring reads these to tell a feed's own date from the collector's fetch-time fallback.
    if isinstance(row.get("date_known"), bool):
        extra["date_known"] = row["date_known"]
    if row.get("fetched_at"):
        extra["fetched_at"] = str(row["fetched_at"])
    return CachedItem(
        url_hash=raw_hash,
        source_id=str(row.get("source_id") or f"source_{idx}"),
        bucket=str(row.get("bucket") or "MACRO"),
        tier=str(row.get("tier") or row.get("source_tier") or "M1"),
        origin=str(row.get("origin") or "curated_M1_M3"),
        title=str(row.get("title") or ""),
        summary=str(row.get("summary") or ""),
        url=url_val,
        published_at=str(row.get("published_at") or ref_dt_str),
        embedding_f16=emb_b64 if isinstance(emb_b64, (str, bytes)) else None,
        weight=float(weight_val) if weight_val is not None else None,
        extra=extra,
    )


def _print_understanding_line(
    scored_clusters: Sequence[Any],
    commission: "DirectorCommission",
    original: Optional["DirectorCommission"] = None,
    extra_group: Any = None,
) -> None:
    """One diagnostic line per run for the understanding-v1 switch; never raises.

    final_echo is measured on commission (the one published), against the five groups and the reflector's
    extra group when one was offered; the attempt counters come from original (the Director's own
    commission, before any photo-recovery revision).
    """
    try:
        counters = original if original is not None else commission
        top_changed, rank = ranking_stats(scored_clusters)
        final_echo = echo_violation(
            [commission.word, commission.gloss, commission.commissioned_thesis, commission.visual_anchor,
             *commission.unsplash_queries, commission.imagen_commission_prompt],
            list(scored_clusters)[:5] + ([extra_group] if extra_group is not None else []),
        )
        print(
            f"[understanding_v1] top_changed={int(top_changed)} new_top_legacy_rank={int(rank)} "
            f"attempt={int(counters.attempts_used)} packet={int(bool(counters.packet_used))} "
            f"echo_rejections={int(counters.echo_rejections)} final_echo={int(final_echo is not None)}"
        )
    except Exception:
        pass


_EMBEDDING_FIELDS = frozenset(
    {"embedding_f32", "embedding_f16", "embedding_f16_b64", "embedding_model", "embedding_source"}
)


def _embed_rows(
    rows: Dict[str, Dict[str, Any]], cache: Dict[str, Dict[str, Any]], client: Any, model: str
) -> Optional[List[Dict[str, Any]]]:
    """Embed copies of the rows live; None on any embedding failure the run absorbs.

    A SpendBlocked raised after a provider call propagates unchanged; every other
    failure, a SpendBlocked refused before any provider call included, gives None.
    """
    try:
        return embed_missing_items(
            [dict(row) for row in rows.values()],
            existing_cache=dict(cache),
            genai_client=client,
            model=model,
            require_live=True,
        )
    except SpendBlocked as exc:
        if refused_before_dispatch(exc):
            return None
        raise
    except Exception:
        return None


def _hold_switch_over_rows(
    run: EmbeddingRun, retained: Dict[str, Dict[str, Any]], client: Any, ref_dt_str: str
) -> None:
    """Switch-over night: embed the retained pool with embedding-2 and hold it unsaved on the run."""
    if len(retained) < embedding_migration.MIN_CALIBRATION_ITEMS:
        run.deferred = True
        return
    stripped = {key: {k: v for k, v in row.items() if k not in _EMBEDDING_FIELDS}
                for key, row in retained.items()}
    embedded = _embed_rows(stripped, {}, client, embedding_migration.EMBED_2)
    try:
        pending = {str(row.get("url_hash") or ""): row for row in embedded or () if isinstance(row, dict)}
        items = [_row_to_cached_item(pending.get(key), idx, ref_dt_str) for idx, key in enumerate(retained)]
    except Exception:
        embedded = None
    if embedded is None:
        # Tonight stays on 001: nothing is held, so no calibration or embedding-2 row is written.
        run.fallback = True
        return
    run.pending_rows = pending
    run.pending_items = [item for item in items if item is not None]


def _live_collection_enabled(value: bool | None) -> bool:
    return value if value is not None else os.environ.get("ZEITGEIST_LIVE_COLLECT", "").strip() == "1"


def _offline_demo_enabled(value: bool | None) -> bool:
    return value if value is not None else os.environ.get("ZEITGEIST_OFFLINE_DEMO", "").strip() == "1"


def _reference_clock(date_iso: str, now_epoch: float | None, offline_demo: bool) -> datetime:
    if now_epoch is not None:
        return datetime.fromtimestamp(float(now_epoch), tz=timezone.utc)
    if offline_demo:
        return datetime.fromisoformat(f"{str(date_iso)[:10]}T12:00:00+00:00")
    return datetime.now(timezone.utc)


def _coverage_requirements(config: dict) -> dict:
    requirements = config.get("coverage_requirements")
    if not isinstance(requirements, dict):
        raise IngestionUnavailable("Reviewed coverage_requirements configuration is required")
    for key in ("min_orbit_sources", "min_macro_sources"):
        value = requirements.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 6:
            raise IngestionUnavailable(f"{key} must be an integer of at least six distinct nonsocial sources")
    for key, expected in (("required_orbit_tiers", {"O1", "O2", "O3"}),
                          ("required_macro_tiers", {"M1", "M2", "M3"})):
        values = requirements.get(key)
        if (not isinstance(values, list) or len(values) != 3
                or not all(isinstance(value, str) for value in values) or set(values) != expected):
            raise IngestionUnavailable(f"{key} must explicitly retain all three reviewed tiers")
    return requirements


def collect_96h_signals(
    repo_root: Optional[Path | Any] = None,
    *,
    date_iso: str = "2026-10-27",
    store: Optional[Any] = None,
    raw_signals: Optional[Sequence[Any]] = None,
    now_epoch: Optional[float] = None,
    live_collect: Optional[bool] = None,
    genai_client: Optional[Any] = None,
    health: Optional[dict] = None,
    offline_demo: bool | None = None,
    embedding_run: Optional[EmbeddingRun] = None,
    loaded_rows_out: Optional[List[Dict[str, Any]]] = None,
    reached_rows_out: Optional[List[Any]] = None,
) -> List[CachedItem]:
    """Load, collect, or synthesize 96h Macro + 336h Orbit items with 768-dim L2-normalized embeddings.

    `loaded_rows_out`, given only by the evidence archive, receives the loaded rows with every field they carry.
    `reached_rows_out`, given only with the failed-night switch on, always holds every row the collector holds so
    far (the cached rows, then the live ones), so a raise inside the collector keeps them.
    """
    from src.state_store import get_state_store

    if (
        repo_root is not None
        and not isinstance(repo_root, (str, Path))
        and (hasattr(repo_root, "load_ingestion_cache") or hasattr(repo_root, "load_pulse_state"))
    ):
        store = repo_root
        repo_root = getattr(store, "repo_root", None)

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    active_store = store if store is not None else get_state_store(repo_root=root)

    should_live_collect = _live_collection_enabled(live_collect)
    demo = _offline_demo_enabled(offline_demo) and not should_live_collect
    now_utc = _reference_clock(date_iso, now_epoch, demo)
    date_str = now_utc.strftime("%Y-%m-%d")
    ref_dt_str = now_utc.isoformat()

    def fresh(row, hours, fetch=False):
        try:
            value = row.get("fetched_at" if fetch else "published_at", "")
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            age = (now_utc - dt).total_seconds() / 3600
            return 0 <= age <= hours
        except (ValueError, TypeError, OverflowError):
            return False

    def eligible_live_row(row):
        if not isinstance(row, dict) or row.get("origin") == "offline_demo":
            return False
        bucket = str(row.get("bucket", "")).upper()
        return fresh(row, 336 if bucket == "ORBIT" else 96) and bucket in {"ORBIT", "MACRO"}

    def hold(rows):
        if reached_rows_out is not None:
            reached_rows_out[:] = [dict(row) if isinstance(row, dict) else row for row in rows]

    existing_cache_map: Dict[str, Dict[str, Any]] = {}
    if hasattr(active_store, "load_ingestion_cache"):
        cached_rows = active_store.load_ingestion_cache(now_utc=now_utc)
        if isinstance(cached_rows, dict):
            for k, v in cached_rows.items():
                if isinstance(v, dict):
                    existing_cache_map[str(k)] = dict(v)
        elif isinstance(cached_rows, (list, tuple)):
            for idx, r in enumerate(cached_rows):
                if isinstance(r, dict):
                    key = str(r.get("url_hash") or r.get("url_sha256") or f"cache_{idx}")
                    existing_cache_map[key] = dict(r)
    hold(existing_cache_map.values())

    if raw_signals is not None:
        if should_live_collect:
            raise IngestionUnavailable("Fixture signals cannot bypass live source coverage checks")
        explicit_items: List[CachedItem] = []
        dicts_to_embed: List[Dict[str, Any]] = []
        for sig in raw_signals:
            if isinstance(sig, CachedItem):
                explicit_items.append(sig)
            elif isinstance(sig, dict):
                dicts_to_embed.append(dict(sig))
        if dicts_to_embed:
            embed_missing_items(
                dicts_to_embed,
                existing_cache=existing_cache_map,
                genai_client=genai_client,
            )
            for idx, d in enumerate(dicts_to_embed):
                coerced = _row_to_cached_item(d, idx, ref_dt_str)
                if coerced is not None:
                    explicit_items.append(coerced)
        return explicit_items


    health = health if health is not None else {}
    health.update(mode="cache", instagram="not_requested", apify_batch_succeeded=False,
                  live_item_count=0, cache_item_count=len(existing_cache_map))
    def require_live_coverage(candidate_rows, *, final=False):
        rows = [r for r in candidate_rows if r.get("origin") != "offline_demo"]
        orbit=[r for r in rows if str(r.get("bucket","")).upper()=="ORBIT" and fresh(r,336)]
        macro=[r for r in rows if str(r.get("bucket","")).upper()=="MACRO" and fresh(r,96)]
        instagram=[r for r in orbit if "_ig_" in str(r.get("source_id","")) and fresh(r,72,fetch=True)]
        tier_counts = {}
        for row in orbit+macro:
            tier=str(row.get("tier") or row.get("source_tier") or "unknown")
            tier_counts[tier]=tier_counts.get(tier,0)+1
        health.update(orbit_item_count=len(orbit), macro_item_count=len(macro), instagram_item_count=len(instagram), seeded=False,
                      fresh_items_by_tier=tier_counts,
                      orbit_source_count=len({str(r.get("source_id")) for r in orbit}),
                      macro_source_count=len({str(r.get("source_id")) for r in macro}),
                      instagram_fetched_successfully=health.get("instagram") in {"live","empty"})
        nonsocial_orbit = [r for r in orbit if "_ig_" not in str(r.get("source_id", ""))
                           and str(r.get("tier") or r.get("source_tier")) in requirements["required_orbit_tiers"]]
        nonsocial_macro = [r for r in macro if str(r.get("tier") or r.get("source_tier")) in requirements["required_macro_tiers"]]
        orbit_sources = {str(r.get("source_id")) for r in nonsocial_orbit}
        macro_sources = {str(r.get("source_id")) for r in nonsocial_macro}
        orbit_tiers = {str(r.get("tier") or r.get("source_tier")) for r in nonsocial_orbit}
        macro_tiers = {str(r.get("tier") or r.get("source_tier")) for r in nonsocial_macro}
        health.update(nonsocial_orbit_source_count=len(orbit_sources), nonsocial_macro_source_count=len(macro_sources),
                      required_coverage=requirements, reference_time=now_utc.isoformat())
        if (not instagram or len(orbit_sources) < requirements["min_orbit_sources"]
                or len(macro_sources) < requirements["min_macro_sources"]
                or not set(requirements["required_orbit_tiers"]).issubset(orbit_tiers)
                or not set(requirements["required_macro_tiers"]).issubset(macro_tiers)):
            health["publishable"]=False
            raise IngestionUnavailable("Required diverse ORBIT, MACRO and Instagram coverage unavailable; retain last publication",
                public_diagnostics={"code": "required-source-coverage-unavailable"}, collection_health=health)
        if final:
            health["publishable"]=True

    if should_live_collect:
        cfg_path = root / "config" / "orbit_sources.json"
        if not cfg_path.exists():
            fallback_cfg = Path(__file__).resolve().parents[2] / "config" / "orbit_sources.json"
            if fallback_cfg.exists():
                cfg_path = fallback_cfg
        sources_config = load_orbit_config(cfg_path) if cfg_path.exists() else {}
        requirements = _coverage_requirements(sources_config)

        orbit_roster: Dict[str, Any] = {}
        if hasattr(active_store, "load_orbit_roster"):
            loaded_roster = active_store.load_orbit_roster()
            if isinstance(loaded_roster, dict):
                orbit_roster = loaded_roster

        # V3 collects every configured non-social source each day. Instagram's
        # separate eight-of-twelve rotation must not filter the O3 RSS tier.
        curated_items = fetch_all_curated_feeds(sources_config)
        health["curated_item_count"] = len(curated_items or [])
        hold(list(existing_cache_map.values()) + list(curated_items or []))

        curated_ig_handles = [
            str(h).lstrip("@").strip().lower()
            for h in (sources_config.get("curated_o3_instagram_handles") or [])
            if str(h).strip()
        ]
        blocklist_set = {str(h).lstrip("@").strip().lower()
                         for h in (sources_config.get("handle_blocklist") or []) if str(h).strip()}
        combined_ig_handles = select_instagram_batch(orbit_roster, curated_ig_handles, blocklist=blocklist_set)
        tier_map: Dict[str, str] = {h: "O3" for h in curated_ig_handles}
        for sh in combined_ig_handles:
            tier_map.setdefault(sh, "O4_SCOUT")

        apify_token = (
            os.environ.get("APIFY_API_TOKEN") or os.environ.get("APIFY_TOKEN") or ""
        ).strip()
        try:
            if not apify_token or not combined_ig_handles:
                raise IngestionUnavailable("Required Instagram credentials or source handles missing")
            ig_items = fetch_instagram_batch(
                combined_ig_handles,
                apify_token=apify_token,
                tier_map=tier_map,
                health=health,
            )
            if not isinstance(ig_items, list):
                raise IngestionUnavailable("Instagram did not return a confirmed batch result")
            health["apify_batch_succeeded"] = True
            health["instagram"] = "live" if ig_items else "empty"
        except IngestionUnavailable as exc:
            health["instagram"] = "unavailable"
            health["instagram_error_type"] = type(exc).__name__
            # Instagram is a required dependency, not an optional fail-open feed.
            raise IngestionUnavailable("Required Instagram collection failed; retain last publication",
                public_diagnostics=exc.public_diagnostics,
                collection_health={**exc.public_collection_health, **health}) from None
        macro_items = fetch_macro_scout_signals(apify_token=apify_token or None)

        # A live collector stamps fetched_at after this function's entry clock.
        # Validate against observed collection completion, keeping explicit test/
        # replay clocks fixed rather than granting an arbitrary future-time skew.
        now_utc = _reference_clock(date_iso, now_epoch, demo)
        date_str = now_utc.strftime("%Y-%m-%d")
        ref_dt_str = now_utc.isoformat()

        live_raw_items = (
            list(curated_items or []) + list(ig_items or []) + list(macro_items or [])
        )

        health["live_item_count"] = len(live_raw_items)
        health["mode"] = "live" if live_raw_items else "cache"

        mentions = extract_one_hop_mentions(
            live_raw_items,
            curated_handles=set(curated_ig_handles),
            blocklist=blocklist_set,
            instagram_only=True,
            require_curated_origin=True,
        )
        updated_roster = advance_roster_after_batch(orbit_roster, mentions, now_utc=now_utc,
            apify_batch_succeeded=health["apify_batch_succeeded"], curated_handles=curated_ig_handles)
        if hasattr(active_store, "save_orbit_roster"):
            active_store.save_orbit_roster(sanitize_orbit_roster_state(updated_roster))

        active_genai_client = genai_client
        if (
            active_genai_client is None
            and os.environ.get("GEMINI_API_KEY", "").strip()
            and os.environ.get("ZEITGEIST_LIVE_LLM") == "1"
        ):
            try:
                from google import genai

                from .spend_guard import guard_genai_client
                active_genai_client = guard_genai_client(genai.Client(
                    api_key=os.environ["GEMINI_API_KEY"].strip()
                ), credential_key=os.environ["GEMINI_API_KEY"].strip())
            except SpendBlocked:
                raise
            except Exception as exc:
                logger.warning("Failed to initialize google.genai Client for embeddings: %s", exc)
                active_genai_client = None

        # Exclude expired, future and malformed dates before buying embeddings.
        # Retained cache vectors with unknown model provenance must be regenerated
        # too; otherwise a new model silently shares a cluster space with old/hash vectors.
        active_rows = {key: dict(row) for key, row in existing_cache_map.items() if eligible_live_row(row)}
        live_hashes = set()
        for row in live_raw_items:
            if not eligible_live_row(row):
                continue
            key = str(row.get("url_hash") or row.get("url_sha256") or compute_url_hash(str(row.get("url") or row.get("title") or "")))
            active_rows[key] = dict(row, url_hash=key)
            live_hashes.add(key)
        health.update(embedding_eligible_item_count=len(active_rows),
            discarded_before_embedding_count=sum(not eligible_live_row(row) for row in live_raw_items))
        hold({**existing_cache_map, **active_rows}.values())
        # Coverage depends on source metadata, not embeddings. Reject an
        # unusable corpus before model calls or cache writes; recheck afterward.
        require_live_coverage(active_rows.values())
        if embedding_run is not None:
            embedding_run.date_str = date_str
        if active_rows:
            from src.state_store import pack_ingestion_cache_pages
            save_day = True
            if embedding_run is not None and embedding_run.mode == "migrated":
                pack_ingestion_cache_pages(date_str, active_rows, reserve_embedding=True,
                    embedding_model=embedding_migration.EMBED_2)
                cache_before_embedding = dict(existing_cache_map)
                embedded_live = _embed_rows(active_rows, cache_before_embedding, active_genai_client,
                    embedding_migration.EMBED_2)
                if embedded_live is not None:
                    embedding_run.used = embedding_migration.EMBED_2
                else:
                    # Refused or unavailable: publish on 001 vectors held in memory only.
                    embedded_live = embed_missing_items(
                        [dict(row) for row in active_rows.values()],
                        existing_cache=dict(cache_before_embedding),
                        genai_client=active_genai_client,
                        model=embedding_migration.EMBED_001,
                        require_live=True,
                    )
                    embedding_run.fallback = True
                    embedding_run.used = embedding_migration.EMBED_001
                    save_day = False
            else:
                pack_ingestion_cache_pages(date_str, active_rows, reserve_embedding=True)
                cache_before_embedding = dict(existing_cache_map)
                embedded_live = embed_missing_items(
                    list(active_rows.values()),
                    existing_cache=existing_cache_map,
                    genai_client=active_genai_client,
                    require_live=True,
                )
                if embedding_run is not None:
                    embedding_run.used = embedding_migration.EMBED_001
            day_shard: Dict[str, Dict[str, Any]] = {}
            retained_cache: Dict[str, Dict[str, Any]] = {}
            for item_dict in embedded_live:
                if not isinstance(item_dict, dict):
                    continue
                u_hash = str(
                    item_dict.get("url_hash") or item_dict.get("url_sha256") or ""
                ).strip()
                if u_hash:
                    old_row = cache_before_embedding.get(u_hash) or {}
                    if (u_hash in live_hashes or old_row.get("embedding_model") != item_dict.get("embedding_model")
                            or old_row.get("embedding_source") != "gemini"
                            or old_row.get("embedding_f16") != item_dict.get("embedding_f16")):
                        day_shard[u_hash] = dict(item_dict)
                    retained_cache[u_hash] = dict(item_dict)
            existing_cache_map = retained_cache
            if save_day and day_shard and hasattr(active_store, "save_ingestion_cache_day"):
                active_store.save_ingestion_cache_day(date_str, day_shard)
            if embedding_run is not None:
                embedding_run.items = len(retained_cache)
                if embedding_run.mode == "switch_over":
                    _hold_switch_over_rows(embedding_run, retained_cache, active_genai_client, ref_dt_str)
        else:
            existing_cache_map = {}

    items: List[CachedItem] = []
    for idx, row in enumerate(existing_cache_map.values()):
        coerced = _row_to_cached_item(row, idx, ref_dt_str)
        if coerced is not None:
            items.append(coerced)
    if loaded_rows_out is not None:
        # In memory only: the rows as loaded, caption, fetch time and embedding model included.
        loaded_rows_out.extend(dict(row) for row in existing_cache_map.values() if isinstance(row, dict))
    hold(existing_cache_map.values())

    if should_live_collect:
        require_live_coverage(existing_cache_map.values(), final=True)
    if items:
        return items

    if not demo:
        raise IngestionUnavailable("No usable cache; seeded data requires explicit offline_demo=True")
    health.update(mode="seeded", seeded=True, publishable=False)
    # Offline demonstration only: live mode is gated above before any seed fallback.
    # Synthesize deterministic 768-dim embedded items from default_curated_corpus when cache is empty
    raw_signal_dicts: List[Dict[str, Any]] = []
    for idx, doc in enumerate(default_curated_corpus()):
        lens_val = str(getattr(doc, "lens_id", getattr(doc, "lens_key", "")) or "")
        is_orbit = lens_val == "personal_orbit"
        tier_str = "O1" if (is_orbit and idx % 2 == 0) else ("O2" if is_orbit else "M1")
        src_name = str(getattr(doc, "source", getattr(doc, "source_name", f"src_{idx}")) or f"src_{idx}")
        summary_txt = str(getattr(doc, "body", getattr(doc, "excerpt", "")) or "")
        url_txt = str(getattr(doc, "url", f"https://github.com/Siyuan-Zhu#{idx}") or "")
        url_hash_val = compute_url_hash(f"{url_txt}?idx={idx}")
        raw_signal_dicts.append(
            {
                "url_hash": url_hash_val,
                "url_sha256": url_hash_val,
                "source_id": f"{'o1' if is_orbit else 'm1'}_{src_name.lower().replace(' ', '_')}",
                "bucket": "ORBIT" if is_orbit else "MACRO",
                "tier": tier_str,
                "origin": "curated_o1_o3" if is_orbit else "curated_M1_M3",
                "title": str(getattr(doc, "title", f"Signal {idx}") or ""),
                "summary": summary_txt,
                "url": url_txt,
                "published_at": ref_dt_str,
            }
        )
    embed_missing_items(raw_signal_dicts, client=None)
    for row in raw_signal_dicts:
        items.append(
            CachedItem(
                url_hash=str(row["url_hash"]),
                source_id=str(row["source_id"]),
                bucket=str(row["bucket"]),
                tier=str(row["tier"]),
                origin="offline_demo",
                title=str(row["title"]),
                summary=str(row["summary"]),
                url=str(row["url"]),
                published_at=str(row["published_at"]),
                embedding_f16=str(row.get("embedding_f16_b64") or ""),
            )
        )
    return items


class ZeitgeistADKOrchestrator:
    """Four-stage Google ADK SequentialAgent orchestrator (v2.0 pipeline)."""

    def __init__(self, repo_root: Path, state_store: Optional[Any] = None) -> None:
        self.repo_root = repo_root
        self._state_store = state_store
        self.config_path = repo_root / "config" / "orbit_sources.json"
        if not self.config_path.exists():
            fallback_cfg = Path(__file__).resolve().parents[2] / "config" / "orbit_sources.json"
            if fallback_cfg.exists():
                self.config_path = fallback_cfg
        self.state_path = repo_root / "data" / "zeitgeist_state.json"
        self.svg_path = repo_root / "assets" / "daily_pulse.svg"
        self.last_commission: Optional[DirectorCommission] = None
        self.last_published_pulse: Optional[PublishedPulse] = None
        # The private run record's inputs; `run()` writes the record only once the edition is published.
        self._run_record_inputs: Optional[Dict[str, Any]] = None
        # The own-group check's inputs, kept only with its switch on; `run()` uses them after the record.
        self._own_group_inputs: Optional[Dict[str, Any]] = None
        # The reflector's inputs, kept only with its switch on; `run()` stores them after the own-group check.
        self._reflector_inputs: Optional[Dict[str, Any]] = None
        # The evidence archive's inputs, kept only with its switch on; `run()` archives them last.
        self._evidence_archive_inputs: Optional[Dict[str, Any]] = None
        # What the night has reached, given by `run()` only with the failed-night switch on.
        self._failed_night: Optional[FailedNightCapture] = None

    def run(
        self,
        date_iso=None,
        prefer_visual_mode=None,
        *,
        live_collect=None,
        now_epoch=None,
        offline_demo=None,
    ):
        """Render and validate in isolation, then conditionally publish one bundle."""
        from src.state_store import get_state_store, PublicationOutcomeUnknown
        from .publication_commit import PublicationWorkspace, publish_bundle
        from .public_schema import validate_editorial_entry

        store = (
            self._state_store
            if self._state_store is not None
            else get_state_store(repo_root=self.repo_root)
        )
        if (self.repo_root / "data/publication_reconcile.json").exists():
            raise RuntimeError(
                "Resolve the pending canonical publication reconciliation before generating another publication"
            )
        from zoneinfo import ZoneInfo
        from src.state_store import FirestoreStateStore
        from .edition_approval import require_edition_approval
        date_iso = str(date_iso or datetime.now(ZoneInfo('America/New_York')).date())[:10]
        if isinstance(store, FirestoreStateStore) and not store.collection_prefix:
            require_edition_approval({'date': date_iso})
        expected = store.load_pulse_state()
        workspace = PublicationWorkspace(self.repo_root)
        committed = False
        installed = False
        journal = self.repo_root / "data/publication_reconcile.json"
        journal_created = False
        # What the night reaches, kept only with the failed-night switch on, for the archive if it fails.
        failed_night = FailedNightCapture() if evidence_archive.failed_night_enabled() else None

        def prepare(state, entry, feed):
            nonlocal journal_created
            workspace.prepare(state, entry, feed)
            recovery_path = workspace.preserve(state)
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text(json.dumps({"publication_id": state["publication_id"],
                "status": "canonical_commit_pending", "recovery_path": recovery_path}) + "\n")
            journal_created = True

        try:
            workspace.stage_canonical_game(store)
            staged = ZeitgeistADKOrchestrator(workspace.path, state_store=store)
            staged._failed_night = failed_night
            analysis, decision, telemetry = staged._run_staged(
                date_iso,
                prefer_visual_mode,
                live_collect=live_collect,
                now_epoch=now_epoch,
                offline_demo=offline_demo,
            )
            payload = staged._prepared_state
            payload["visual_decision"]["local_svg_path"] = "assets/daily_pulse.svg"
            manifest = json.loads(
                (workspace.path / "assets/pulse_manifest.json").read_text()
            )
            payload["publication_id"] = manifest["publication_id"]
            feed_path = workspace.path / "data/pulse_feed.json"
            feed = json.loads(feed_path.read_text())
            generated_entry = staged.last_published_pulse.to_feed_entry()
            _reach(failed_night, "validation", entry=generated_entry)
            validate_editorial_entry(generated_entry)
            _reach(failed_night, "publication")
            state = publish_bundle(
                store,
                payload,
                generated_entry,
                expected=expected,
                local_feed=feed,
                prepare=prepare,
            )
            committed = True
            if isinstance(state, dict) and isinstance(state.get("published_pulse"), dict):
                # The entry as committed, with its publication id.
                _reach(failed_night, entry=state["published_pulse"])
            workspace.install(state)
            installed = True
            from .visual_router import confirm_published_photo_download
            published_photo = staged.last_published_pulse
            if (published_photo.attribution.get("download_location")
                    and os.environ.get("ZEITGEIST_LIVE_UNSPLASH") == "1"):
                marker = json.loads(journal.read_text())
                marker.update(status="canonical_committed_download_tracking_pending",
                    download_location=published_photo.attribution["download_location"],
                    photo_id=published_photo.attribution.get("photo_id"),
                    tracking_delivery="at_least_once_response_may_be_ambiguous")
                journal.write_text(json.dumps(marker) + "\n")
                confirm_published_photo_download(published_photo)
            journal.unlink(missing_ok=True)
            self.last_commission = staged.last_commission
            self.last_published_pulse = replace(
                staged.last_published_pulse, mood_votes=state["visitor_mood_tally"]
            )
            result = (
                analysis,
                replace(decision, local_svg_path=str(self.svg_path)),
                telemetry,
            )
        except BaseException as exc:
            if committed or isinstance(exc, PublicationOutcomeUnknown):
                store.set_needs_git_reconcile(True)
                tracking_marked = False
                try:
                    marker = json.loads(journal.read_text())
                    # The download-tracking step marks the journal with its link before it runs.
                    tracking_marked = "download_location" in marker
                    marker["status"] = ("canonical_committed_download_tracking_pending" if installed else "canonical_committed_local_install_failed" if committed
                                        else "canonical_commit_outcome_unknown")
                    journal.write_text(json.dumps(marker) + "\n")
                except (OSError, ValueError):
                    pass  # The pre-dispatch journal remains the authoritative recovery signal.
                # Only once the recovery marker and the journal are written, labelled by what was observed.
                _archive_failed_night(store, failed_night, exc, date_iso,
                                      ("committed_download_tracking_failed" if tracking_marked
                                       else "committed_installed_followup_failed") if installed
                                      else "committed_install_failed" if committed else "outcome_unknown")
                from .publication_commit import PublicationMirrorError

                raise PublicationMirrorError(
                    "Canonical publication committed or outcome unknown; reconcile retained bundle before retry"
                ) from exc
            if journal_created:
                journal.unlink(missing_ok=True)
            _archive_failed_night(store, failed_night, exc, date_iso, "failed")
            raise
        finally:
            workspace.close()
        # Published, installed and the download confirmed: only now, outside the publication's
        # failure handler, the private run record.
        written = _write_pending_run_record(store, staged)
        # Then the own-group check on that record and then what the reflector needs from the run,
        # guarded as the record write is and held to one wall-clock budget, so a stalled provider
        # cannot cost the job's later steps (the README and the push). With their switches off no
        # inputs were kept and nothing is called.
        try:
            own_group_inputs = staged._own_group_inputs
            # The reflector's inputs go only on a record this run wrote.
            reflector_inputs = staged._reflector_inputs if written else None
            if own_group_inputs is not None or reflector_inputs is not None:
                checked = own_group.ClosableStore(store)

                def after_publication() -> None:
                    if own_group_inputs is not None:
                        try:
                            _daily_own_group_check(checked, **own_group_inputs)
                        except BaseException:
                            pass  # Its own thread: a failed check still leaves the reflector's inputs.
                    if reflector_inputs is not None:
                        _daily_reflector_inputs(checked, **reflector_inputs)

                if not own_group.run_with_budget(after_publication, own_group.BUDGET_SECONDS):
                    # The abandoned steps may still finish; they store nothing.
                    checked.close()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass
        # Last, and only with its switch on: the night's evidence, kept privately and forever, within its
        # own budget. With the switch off no inputs were kept and nothing is called or printed.
        archive_inputs = getattr(staged, "_evidence_archive_inputs", None)
        if archive_inputs is not None:
            # The entry as committed (with its publication id), else the one generated.
            committed = state.get("published_pulse") if isinstance(state, dict) else None
            _archive_evidence(store, archive_inputs, committed if isinstance(committed, dict) else generated_entry)
        return result

    def _run_staged(
        self,
        date_iso: str | None = None,
        prefer_visual_mode: Optional[str] = None,
        *,
        live_collect: Optional[bool] = None,
        now_epoch: float | None = None,
        offline_demo: bool | None = None,
    ) -> Tuple[WeightedOrbitAnalysis, VisualDecision, ADKRunTelemetry]:
        """Execute 96h signal collection -> clustering -> Editorial Director -> Visual Critic & Renderer."""
        config = load_orbit_config(self.config_path)
        run_models = resolve_run_models(config)
        with activate(run_models):
            return self._run_staged_on_models(
                config, run_models, date_iso, prefer_visual_mode,
                live_collect=live_collect, now_epoch=now_epoch, offline_demo=offline_demo,
            )

    def _run_staged_on_models(
        self,
        config: dict,
        run_models: RunModels,
        date_iso: str | None,
        prefer_visual_mode: Optional[str],
        *,
        live_collect: Optional[bool],
        now_epoch: float | None,
        offline_demo: bool | None,
    ) -> Tuple[WeightedOrbitAnalysis, VisualDecision, ADKRunTelemetry]:
        from src.state_store import get_state_store

        date_iso = str(date_iso or datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10]
        owner = dict(config.get("active_interjection") or {})
        if prefer_visual_mode is not None:
            owner.update(enabled=True, override_visual_mode=prefer_visual_mode)
        owner = validate_owner_interjection(owner)
        requested_medium = owner.get("override_visual_mode") if owner.get("enabled") else None
        store = self._state_store if self._state_store is not None else get_state_store(repo_root=self.repo_root)
        existing_state = store.load_pulse_state()
        tally = normalize_mood_votes(
            existing_state.get(
                "visitor_mood_tally",
                {
                    "spot_on": 20,
                    "too_sunny": 3,
                    "too_heavy": 4,
                },
            )
        )

        # Stage 1 & 2: Collect 96h/336h signals, apply stratified windowing, and cluster
        capture = self._failed_night
        reached_rows: Optional[List[Any]] = [] if capture is not None else None
        _reach(capture, "collection", rows=reached_rows)
        collection_health = {}
        embedding_run = (embedding_migration.plan_embedding_run(store, run_models.embedding)
                         if _live_collection_enabled(live_collect) else None)
        embedding_kwargs = {"embedding_run": embedding_run} if embedding_run is not None else {}
        # The evidence archive's capture, each given only with its switch on.
        archive_on = evidence_archive.archive_enabled()
        loaded_rows: Optional[List[Dict[str, Any]]] = [] if archive_on else None
        photo_phases: Optional[List[List[Dict[str, Any]]]] = [] if archive_on else None
        photo_revisions: Optional[List[DirectorCommission]] = [] if archive_on else None
        archive_collect_kwargs = {"loaded_rows_out": loaded_rows} if loaded_rows is not None else {}
        if reached_rows is not None:
            archive_collect_kwargs["reached_rows_out"] = reached_rows
        try:
            raw_signals = collect_96h_signals(
                self.repo_root,
                date_iso=date_iso,
                store=store,
                live_collect=live_collect,
                health=collection_health,
                now_epoch=now_epoch,
                offline_demo=offline_demo,
                **embedding_kwargs,
                **archive_collect_kwargs,
            )
        except IngestionUnavailable as exc:
            exc.public_collection_health = public_collection_health({**exc.public_collection_health, **collection_health})
            raise
        _reach(capture, "grouping", items=raw_signals)
        reference_time = _reference_clock(date_iso, now_epoch,
            _offline_demo_enabled(offline_demo) and not _live_collection_enabled(live_collect))
        understanding_v1 = understanding_v1_enabled()
        # With the interim grouping switch off, both calls are today's.
        interim_grouping = interim_grouping_enabled()
        if embedding_run is None:
            scored_clusters = score_and_cluster_signals(
                raw_signals,
                now_utc=reference_time,
                **({"exclude_captionless": True} if interim_grouping else {}),
            )
        else:
            scored_clusters = embedding_migration.cluster_for_run(
                raw_signals, embedding_run, store=store, now_utc=reference_time,
                **({"interim_grouping": True} if interim_grouping else {}))
            print(embedding_migration.embedding_line(embedding_run))
        if interim_grouping:
            print(GROUPING_LINE % sum(1 for item in raw_signals if is_captionless(item)))
        _reach(capture, "director", clusters=scored_clusters)

        # Extract 30-day word & visual medium history from pulse_feed.json
        feed_entries: List[Dict[str, Any]] = []
        if hasattr(store, "load_pulse_feed_history"):
            loaded_hist = store.load_pulse_feed_history(days=30)
            if isinstance(loaded_hist, list):
                feed_entries = loaded_hist
            elif isinstance(loaded_hist, dict):
                feed_entries = list(loaded_hist.get("entries", []))
        else:
            feed_file = self.repo_root / "data" / "pulse_feed.json"
            if feed_file.exists():
                try:
                    raw_feed = json.loads(feed_file.read_text(encoding="utf-8"))
                    if isinstance(raw_feed, dict):
                        feed_entries = list(raw_feed.get("entries", []))
                except Exception:
                    feed_entries = []

        history_words_30d: List[str] = []
        medium_history_30d: List[str] = []
        archetype_history_30d: List[str] = []
        for entry in feed_entries[:30]:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("date", ""))[:10] == str(date_iso)[:10]:
                continue
            w = entry.get("artwork_title") or entry.get("word")
            if w:
                history_words_30d.append(str(w))
            m = entry.get("visual_medium") or entry.get("visual_tier_used")
            if m:
                medium_history_30d.append(str(m))
            arch = entry.get("curatorial_archetype")
            if arch:
                archetype_history_30d.append(str(arch))
        if not archetype_history_30d:
            prev_pulse = existing_state
            for arch in prev_pulse.get("archetype_history_30d") or ():
                if arch:
                    archetype_history_30d.append(str(arch))

        # Stage 3: Editorial Director Agent (with Latin-script & 30-day deduplication gate)
        genai_client = None
        if (os.environ.get("GEMINI_API_KEY", "").strip() and os.environ.get("ZEITGEIST_LIVE_LLM") == "1"
                and not _offline_demo_enabled(offline_demo)):
            try:
                from google import genai

                from .spend_guard import guard_genai_client
                genai_client = guard_genai_client(genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip()), credential_key=os.environ["GEMINI_API_KEY"].strip())
            except SpendBlocked:
                raise
            except Exception:
                genai_client = None

        commission_kwargs: Dict[str, Any] = (
            {"evidence_packet": True, "now_utc": reference_time} if understanding_v1 else {}
        )
        # Links between groups in different realms, offered on the packet path only. With the switch
        # off nothing is read and the Director call is today's.
        night_links: Optional[List[Any]] = None
        realm_map: Any = None
        if links_enabled():
            night_links, realm_map = _night_links(store, config, scored_clusters)
            commission_kwargs["links"] = night_links
        # The group the reflector pointed to, offered as one more candidate on the packet path. With its switch or
        # the keyword repair off nothing is read and no thread starts. The lookup is held to its own budget, and
        # any failure or timeout offers nothing, as with the switch off.
        reflector_candidate: Optional[Dict[str, Any]] = None
        extra_group = None
        if understanding_v1 and reflector.candidate_enabled():
            reflector_candidate = _reflector_candidate(store, scored_clusters)
            extra_group = reflector_candidate.get("group")
            if extra_group is not None:
                commission_kwargs["extra_group"] = extra_group
        if capture is not None:
            commission_kwargs["attempts_out"] = capture.director_attempts.setdefault("main", [])
        commission = commission_daily_zeitgeist(
            scored_clusters,
            history_words_30d=history_words_30d,
            genai_client=genai_client,
            owner_override=owner,
            model_name=str(config.get("budget_guardrails", {}).get("primary_text_model", "gemini-3.8-flash")),
            run_models=run_models,
            **commission_kwargs,
        )
        _reach(capture, main_commission=commission)
        fallback_shown = run_models.fallback_fired
        if genai_client is not None:
            print(model_choice_line(run_models, commission.slate_count))
        if (os.environ.get("ZEITGEIST_LIVE_LLM") == "1"
                and not _offline_demo_enabled(offline_demo) and commission.fallback_used):
            raise ValueError("Live Director commissioning unavailable; retain last publication")
        if (owner.get("enabled") and not _offline_demo_enabled(offline_demo)
                and any(owner.get(k) for k in ("override_word", "override_phrase", "personal_note"))
                and commission.fallback_used):
            raise ValueError("Owner-constrained live commissioning unavailable; retain last publication")
        self.last_commission = commission
        original_commission = commission

        top_cluster = scored_clusters[0] if scored_clusters else None
        bonus_cluster = top_cluster
        if understanding_v1 and scored_clusters:
            try:
                bonus_cluster = legacy_top_cluster(scored_clusters) or top_cluster
            except Exception:
                bonus_cluster = top_cluster
        if (bonus_cluster is not None and collection_health.get("apify_batch_succeeded") is True
                and hasattr(store, "load_orbit_roster") and hasattr(store, "save_orbit_roster")):
            roster_before_bonus = store.load_orbit_roster()
            roster_with_bonus = apply_winning_cluster_bonus(roster_before_bonus, bonus_cluster.items,
                reference_time, apify_batch_succeeded=True)
            if roster_with_bonus != roster_before_bonus:
                store.save_orbit_roster(sanitize_orbit_roster_state(roster_with_bonus))
        # The published figures and keywords: the top group's, or with the repair switch on the candidate
        # the word is about (for a link, its merged keywords and its highest-ranked member's figures).
        field_cluster, link_words = top_cluster, None
        if repair_enabled() and top_cluster is not None:
            field_cluster, link_words = _published_candidate(
                commission.selected_cluster_id, scored_clusters, night_links, top_cluster, extra_group=extra_group)
        orbit_res = float(field_cluster.orbit_resonance) if field_cluster is not None else 1.50
        vel_score = float(field_cluster.velocity) if field_cluster is not None else 2.50
        is_breakout = bool(getattr(field_cluster, "breakout_surge", False)) if field_cluster is not None else False
        if drop_false_fields_enabled() and field_cluster is not None:
            # The multiplier is 1.0 or 1.25, so the bare truth test is always true; only a real surge counts.
            is_breakout = float(getattr(field_cluster, "breakout_surge", 0.0) or 0.0) > 1.0
        c_keywords = list(field_cluster.cluster_keywords) if field_cluster is not None else [commission.word]
        if understanding_v1 and field_cluster is not None:
            try:
                c_keywords = public_keywords(c_keywords, field_cluster)
            except Exception:
                pass
        if link_words is not None:
            c_keywords = link_words
        c_sil = float(field_cluster.silhouette_score) if field_cluster is not None else 0.75

        # Stage 4: Multimodal Visual Critic Funnel & Dual-Resolution WebP/SVG Renderer
        _reach(capture, "photo", photo_phases=photo_phases, revisions=photo_revisions)
        curated_candidates = None
        live_unsplash_enabled = os.environ.get("ZEITGEIST_LIVE_UNSPLASH", "1").strip() != "0"
        has_live_unsplash_key = (
            bool(os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()) and live_unsplash_enabled
        )
        if requested_medium != "blueprint" and live_unsplash_enabled and not has_live_unsplash_key:
            from .visual_router import fetch_curated_library_candidates

            curated_candidates = fetch_curated_library_candidates(word=commission.word)

        from .visual_router import PhotoSelectionUnavailable
        # The stored calibration is read only when a Gemini 4 model judges a live run. At the gate the marks are
        # those of the judge in force, which produced every score of the stage: after a hand-over to Flash the
        # stage's earlier batches are scored again on Flash, so Flash's marks never meet Gemini 4 scores.
        judge_document = (judge_calibration.load_document(store)
                          if genai_client is not None and run_models.current("judge") in GEMINI_4 else None)
        pass_marks = (judge_calibration.DEFAULT_MARKS if judge_document is None
                      else lambda: judge_calibration.marks_for(run_models.current("judge"), judge_document))
        exception_candidates = []
        # Photos of a batch the judge safety-blocked stay out of every later photo phase of this run.
        safety_blocked_ids = set()
        def render_commission(current, *, photo_only=None):
            nonlocal exception_candidates
            archive_kwargs = {}
            if photo_phases is not None:
                photo_phases.append([])
                archive_kwargs["scored_candidates_out"] = photo_phases[-1]
            try:
                return select_and_render_pulse_assets(
                    commission=current,
                    medium_history_30d=medium_history_30d,
                    archetype_history_30d=archetype_history_30d,
                    breakout_surge=is_breakout,
                    repo_root=self.repo_root,
                    date_str=str(date_iso)[:10],
                    candidates=exception_candidates if photo_only is False else curated_candidates,
                    orbit_resonance_score=orbit_res,
                    velocity_score=vel_score,
                    cluster_keywords=c_keywords,
                    cluster_silhouette=c_sil,
                    mood_votes=tally,
                    requested_medium="photo" if photo_only is True else requested_medium,
                    require_semantic_rescore=bool(owner.get("enabled") and not _offline_demo_enabled(offline_demo)),
                    photo_history_30d=feed_entries,
                    judge_model=(run_models.current("judge") if genai_client is not None else
                                 str(config.get("budget_guardrails", {}).get("multimodal_judge_model", "gemini-3.8-flash"))),
                    strict_photo_errors=genai_client is not None,
                    offline_demo=offline_demo is True,
                    pass_marks=pass_marks,
                    safety_blocked_ids=safety_blocked_ids,
                    **archive_kwargs,
                )
            except PhotoSelectionUnavailable as exc:
                # Only candidates scored against this exact commission survive to
                # the allowed exception phase; never reuse another story's score.
                exception_candidates = exc.exception_candidates
                raise
        try:
            if genai_client is not None and requested_medium == "blueprint":
                from .photo_recovery import ai_exception_allowed
                from .media_provenance import publication_media_history
                modes = publication_media_history(feed_entries, replacing_date=str(date_iso))
                if not ai_exception_allowed(modes):
                    raise ValueError("Non-photo cadence does not permit this requested edition")
            if genai_client is not None and requested_medium != "blueprint":
                from .photo_recovery import recover_visual, revision_owner_constraints
                def revise_for_photo(current, *, change_word):
                    # A chosen link's story is its highest-ranked member; the revision keeps the link's id.
                    chosen_link = _shown_link(current.selected_cluster_id, night_links)
                    if chosen_link is not None:
                        selected = [chosen_link.members[0]]
                    else:
                        selected = [cluster for cluster in scored_clusters if getattr(cluster, "cluster_id", None)==current.selected_cluster_id]
                    if not selected:
                        raise ValueError("Original story unavailable; cannot revise unrelated signals")
                    story_id = getattr(selected[0], "cluster_id", None) if chosen_link is not None else current.selected_cluster_id
                    locks = revision_owner_constraints(owner, current, change_word=change_word)
                    revision_kwargs = {}
                    if capture is not None:
                        revision_kwargs["attempts_out"] = capture.director_attempts.setdefault(
                            f"revision:{len(photo_revisions) + 1}", [])
                    revised = commission_daily_zeitgeist(selected,
                        history_words_30d=history_words_30d+([current.word] if change_word and not locks.get("override_word") else []),
                        genai_client=genai_client, model_name=run_models.current("director"),
                        run_models=run_models, max_retries=0, owner_override=locks, check_shown_ids=False,
                        **revision_kwargs)
                    if photo_revisions is not None:
                        photo_revisions.append(revised)
                    if revised.fallback_used or revised.selected_cluster_id!=story_id:
                        raise ValueError("Same-story photo recovery did not produce a grounded commission")
                    if chosen_link is not None:
                        revised.selected_cluster_id = current.selected_cluster_id
                    return revised
                from .media_provenance import publication_media_history
                cadence_history = publication_media_history(feed_entries, replacing_date=str(date_iso))
                commission, published, recovery_trace = recover_visual(commission,
                    render=render_commission, revise=revise_for_photo, history=cadence_history,
                    photo_required=requested_medium=="photo")
                self.last_commission = commission
                self.last_photo_recovery = recovery_trace
                _reach(capture, recovery_trace=recovery_trace)
            else:
                published = render_commission(commission)
        finally:
            # A judge or revision refusal during rendering moves the run to Flash after the first line.
            if genai_client is not None and run_models.fallback_fired and not fallback_shown:
                print(model_choice_line(run_models, commission.slate_count))
        self.last_published_pulse = published
        # Rendered: the edition's records, state and bundle are assembled until its entry is validated.
        _reach(capture, "assembly", published=published)
        if understanding_v1:
            # After photo recovery: measure the commission that will be published.
            _print_understanding_line(scored_clusters, self.last_commission, original_commission,
                                      extra_group=extra_group)
        # Compatibility DTOs describe the one actual publication. The primary
        # clustering/commissioning path above remains authoritative; no second renderer runs.
        lens_by_tier = {"M1": "ai_builders", "M2": "markets_finops", "M3": "global_politics"}
        observed_documents = [FeedDocument(
            lens_id="personal_orbit" if item.bucket == "ORBIT" else lens_by_tier.get(item.tier, "ai_builders"),
            source_id=item.source_id, title=item.title, summary=item.summary,
        ) for item in raw_signals]
        analysis = analyze_seven_bucket_orbit(config=config, documents=observed_documents, date_iso=date_iso)
        analysis = replace(analysis, word_of_the_day=published.word, explanation_phrase=published.gloss,
                           visual_concrete_anchor=commission.visual_anchor)
        attribution = published.attribution
        credit = str(attribution.get("photographer_name") or {
            "IMAGEN_3": "Gemini editorial synthesis", "FLUX_PRO": "FLUX editorial synthesis",
        }.get(published.visual_medium, "Procedural editorial plate"))
        decision = VisualDecision(
            selected_mode="CURATED_PHOTO" if published.visual_medium == "PHOTOGRAPHY" else published.visual_medium,
            word_of_the_day=published.word, explanation_phrase=published.gloss,
            composite_score=published.composite_critic_score, image_url=str(attribution.get("unsplash_url") or ""),
            local_svg_path=str(self.svg_path), credit_label=credit,
            credit_url=str(attribution.get("unsplash_url") or ""),
            semantic_alignment=published.semantic_alignment_score,
            minimalist_restraint=published.composite_critic_score, safety_pass=True,
            rationale="Single v3 publication; compatibility fields derived from the selected artwork",
        )
        adk_mode = "google-adk-live" if not commission.fallback_used else "deterministic-hybrid-adk"

        # Compute token & monthly budget telemetry
        guardrails = config.get("budget_guardrails", {})
        ceiling = float(guardrails.get("monthly_ceiling_usd_index", 2.50))
        # Provider billing and token usage must be reconciled from actual responses.
        # A configured ceiling is not evidence that this publication was cheap.
        est_in_tokens = None
        est_out_tokens = None
        monthly_cost_index = None

        telemetry = ADKRunTelemetry(
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            adk_mode=adk_mode,
            model_id=run_models.current("director"),
            estimated_input_tokens=est_in_tokens,
            estimated_output_tokens=est_out_tokens,
            estimated_monthly_cost_index=monthly_cost_index,
            monthly_budget_ceiling_index=ceiling,
            within_budget=False,
            spending_status="unknown_pending_billing_reconciliation",
            collection_health=collection_health,
        )

        self._persist_state(analysis, decision, telemetry, commission=commission, published=published)
        if run_record.run_record_enabled():
            self._run_record_inputs = {
                "date_iso": date_iso, "run_models": run_models, "embedding_run": embedding_run,
                "clusters": scored_clusters, "commission": commission, "main_commission": original_commission,
                # Only links an attempt showed are listed; with the packet path off, none (as with links off).
                "links": links_shown(original_commission, night_links),
                "realm_counts": (None if night_links is None else _night_realm_counts(
                    raw_signals, realm_map, scored_clusters, links_shown(original_commission, night_links) or [])),
                "links_on": night_links is not None,
                "reflector_candidate": reflector.candidate_record(reflector_candidate,
                                                                 shown=extra_group_shown(original_commission)),
            }
        if archive_on:
            self._evidence_archive_inputs = {
                "date_iso": date_iso, "rows": loaded_rows, "items": raw_signals, "clusters": scored_clusters,
                "main_commission": original_commission, "commission": commission, "revisions": photo_revisions,
                "photo_phases": photo_phases, "published": published,
                "recovery_trace": getattr(self, "last_photo_recovery", None),
            }
        if run_record.run_record_enabled() and own_group.own_group_enabled():
            self._own_group_inputs = {
                "date_iso": date_iso, "run_models": run_models, "embedding_run": embedding_run,
                "clusters": scored_clusters, "commission": commission, "genai_client": genai_client,
                "links": links_shown(original_commission, night_links), "extra_group": extra_group,
            }
        if run_record.run_record_enabled() and reflector.reflector_enabled():
            self._reflector_inputs = {
                "date_iso": date_iso, "run_models": run_models, "embedding_run": embedding_run,
                "clusters": scored_clusters, "commission": commission, "extra_group": extra_group,
            }
        return analysis, decision, telemetry

    def _persist_state(
        self,
        analysis: WeightedOrbitAnalysis,
        decision: VisualDecision,
        telemetry: ADKRunTelemetry,
        *,
        commission: Optional[DirectorCommission] = None,
        published: Optional[PublishedPulse] = None,
    ) -> None:
        from src.state_store import get_state_store

        store = self._state_store if self._state_store is not None else get_state_store(repo_root=self.repo_root)
        existing = store.load_pulse_state()
        tally = normalize_mood_votes(
            existing.get(
                "visitor_mood_tally",
                {
                    "spot_on": 20,
                    "too_sunny": 3,
                    "too_heavy": 4,
                },
            )
        )
        payload = {
            "date_iso": analysis.date_iso,
            "analysis": asdict(analysis),
            "visual_decision": asdict(decision),
            "telemetry": asdict(telemetry),
            "visitor_mood_tally": tally,
        }
        if commission is not None:
            payload["commission"] = commission.to_dict()
        if published is not None:
            payload["published_pulse"] = published.to_feed_entry()
            payload["curatorial_archetype"] = published.curatorial_archetype
            payload["text_quadrant"] = published.text_quadrant
            prev_arch_hist = [
                str(a)
                for a in (existing.get("archetype_history_30d") or ())
                if str(a).strip()
            ]
            payload["archetype_history_30d"] = (
                [published.curatorial_archetype] + prev_arch_hist
            )[:30]

        # The canonical adapter is only invoked after the staged README gate.
        self._prepared_state = payload


def run_daily_zeitgeist(
    repo_root: Optional[Path] = None,
    date_iso: str | None = None,
    prefer_visual_mode: Optional[str] = None,
    *,
    live_collect: Optional[bool] = None,
    now_epoch: float | None = None,
    offline_demo: bool | None = None,
) -> dict:
    """Convenience entrypoint for CLI, GitHub Actions, and Cloud Run."""
    root = repo_root or Path(__file__).resolve().parents[2]
    orchestrator = ZeitgeistADKOrchestrator(root)
    analysis, decision, telemetry = orchestrator.run(
        date_iso=date_iso,
        prefer_visual_mode=prefer_visual_mode,
        live_collect=live_collect,
        now_epoch=now_epoch,
        offline_demo=offline_demo,
    )
    result = {
        "analysis": asdict(analysis),
        "visual_decision": asdict(decision),
        "telemetry": asdict(telemetry),
    }
    if orchestrator.last_commission is not None:
        result["commission"] = orchestrator.last_commission.to_dict()
    if orchestrator.last_published_pulse is not None:
        result["published_pulse"] = orchestrator.last_published_pulse.to_feed_entry()
    return result


# ---------------------------------------------------------------------------
# WS3 Seam 3: Editorial Director Agent & Latin-Script / 30-Day Deduplication Gate
# ---------------------------------------------------------------------------

from dataclasses import field
import inspect
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence
import unicodedata

# Strictly anchored with \Z (zero literal 0x24 characters in source)
WORD_FORMAT_PATTERN = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿ'-]{2,18}(?: [A-Za-zÀ-ÖØ-öø-ÿ'-]{2,18})?\Z"
)

DEFAULT_PALETTE_HEX: List[str] = [
    "#0F172A",
    "#1E293B",
    "#38BDF8",
    "#10B981",
    "#F8FAFC",
]

FALLBACK_EDITORIAL_VOCABULARY: Tuple[Tuple[str, str, str], ...] = (
    (
        "Tessellation",
        "Geometric tiling aligning independent signals.",
        "Interlocking architectural limestone tiles under directional daylight",
    ),
    (
        "Equilibrium",
        "Balanced tension across builder and macro feeds.",
        "Minimalist suspension bridge cable array in morning mist",
    ),
    (
        "Bulkhead",
        "Structural partition isolating fault domains.",
        "Heavy architectural concrete bulkhead door with brass latches",
    ),
    (
        "Counterpoise",
        "Counterweight stabilizing high-velocity shifts.",
        "Precision brass balance beam on an obsidian plinth",
    ),
    (
        "Laminar Flow",
        "Parallel stream movement without turbulence.",
        "Glass wind-tunnel streamline visualization in cyan light",
    ),
    (
        "Cantilever",
        "Structural projection from a single fulcrum.",
        "Brutalist cantilevered concrete terrace over calm water",
    ),
    (
        "Refraction",
        "Phase shift as signals cross density boundaries.",
        "Optical glass prism splitting monochromatic beam on slate",
    ),
    (
        "Isobaric Framing",
        "Constant-pressure boundary across runtime layers.",
        "Brushed steel barometric manifold with calibrated pressure gauges",
    ),
)


@dataclass
class DirectorCommission:
    """Structured output of the Editorial Director Agent (Stage 3)."""

    word: str
    gloss: str
    commissioned_thesis: str
    visual_anchor: str = "Architectural structural study in natural light"
    unsplash_queries: List[str] = field(
        default_factory=lambda: [
            "minimalist architecture",
            "structural geometry",
            "industrial craft",
        ]
    )
    imagen_commission_prompt: str = (
        "Minimalist editorial architectural study with precise geometric shadows."
    )
    flux_3_commission_prompts: List[str] = field(default_factory=list)
    palette_hex: List[str] | Dict[str, str] = field(
        default_factory=lambda: list(DEFAULT_PALETTE_HEX)
    )
    selected_cluster_id: str = "cluster_0"
    attempts_used: int = 1
    fallback_used: bool = False
    packet_used: bool = False
    echo_rejections: int = 0
    slate_count: int = 0

    def __post_init__(self) -> None:
        if not self.flux_3_commission_prompts:
            base = (self.imagen_commission_prompt or self.visual_anchor or "Editorial architectural study").strip().rstrip(".")
            anchor = (self.visual_anchor or "architectural concrete and natural light").strip().rstrip(".")
            self.flux_3_commission_prompts = [
                f"{base}. Calm negative space in upper-left quadrant, 3:2 editorial photography, 35mm film grain.",
                f"Quiet atmospheric study of {anchor}, minimalist gallery composition, natural directional light, 3:2 aspect ratio.",
                f"Fine-art architectural monograph plate of {anchor}, subtle tonal gradation, zero text overlay, 3:2 format.",
            ]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Understanding-v1 bookkeeping appears only when the switch actually did something.
        if not self.packet_used and not self.echo_rejections:
            data.pop("packet_used", None)
            data.pop("echo_rejections", None)
        if not self.slate_count:
            data.pop("slate_count", None)
        return data


def _find_non_latin_script_char(text: str) -> Optional[Tuple[str, str]]:
    """Return (char, unicode_name) if any letter or mark in text is non-Latin script."""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            uname = unicodedata.name(ch, "UNKNOWN")
            if not uname.startswith("LATIN"):
                return ch, uname
        elif cat.startswith("M"):
            uname = unicodedata.name(ch, "UNKNOWN")
            # Allow standard Combining Diacritical Marks block (U+0300..U+036F) used with Latin
            if ord(ch) > 0x036F:
                return ch, uname
    return None


def validate_director_commission(
    commission: DirectorCommission | Dict[str, Any],
    history_words_30d: Optional[Iterable[str]] = None,
) -> Tuple[bool, str]:
    """Validate word format, 30-day deduplication, and strict Latin-script purity."""
    if isinstance(commission, dict):
        word = str(commission.get("word", "") or "").strip()
        gloss = str(commission.get("gloss", "") or "").strip()
        thesis = str(commission.get("commissioned_thesis", "") or "").strip()
    else:
        word = str(commission.word or "").strip()
        gloss = str(commission.gloss or "").strip()
        thesis = str(commission.commissioned_thesis or "").strip()

    # Check non-Latin script leakage across word, gloss, and commissioned_thesis first
    for field_name, field_val in (
        ("word", word),
        ("gloss", gloss),
        ("commissioned_thesis", thesis),
    ):
        if not field_val:
            return False, f"Field '{field_name}' must be non-empty."
        violation = _find_non_latin_script_char(field_val)
        if violation is not None:
            bad_ch, bad_name = violation
            return (
                False,
                f"Non-Latin script character {bad_ch!r} ({bad_name}) detected in {field_name}.",
            )

    if len(gloss) > 50 or len(gloss.split()) > 12 or any(ch in gloss for ch in '\r\n'):
        return False, (f"Gloss must be a single line of <= 50 characters and <= 12 words "
                       f"(received {len(gloss)} characters, {len(gloss.split())} words).")

    # Check strict 1-2 word regex format anchored with \Z
    if not WORD_FORMAT_PATTERN.match(word):
        return (
            False,
            f"Word '{word}' violates format regex (must be 1-2 Latin words, 2-18 chars each).",
        )

    # Check 30-day deduplication gate (case-insensitive)
    history_lower = {
        str(w).strip().lower() for w in (history_words_30d or ()) if str(w).strip()
    }
    if word.lower() in history_lower:
        return (
            False,
            f"Word '{word}' duplicates a word published in the last 30 days.",
        )

    return True, "valid"


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [
            ln
            for ln in cleaned.splitlines()
            if not ln.strip().startswith("```")
        ]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _take_candidate_slate(raw: Any) -> Tuple[Any, int, List[Dict[str, str]]]:
    """Pop candidate_words from a director reply; return (reply, count, words). Never raises.

    The count is the number of well-formed entries, and only when there are 8 to 12 of
    them; otherwise 0. The words are the well-formed entries; they go only to the private
    run record, kept beside the commission and never on it (`_keep_director_evidence`).
    """
    try:
        payload = raw.text if hasattr(raw, "text") and isinstance(raw.text, str) else raw
        if isinstance(payload, str):
            data = json.loads(_strip_code_fence(payload))
        elif isinstance(payload, dict):
            data = dict(payload)
        else:
            return raw, 0, []
        if not isinstance(data, dict):
            return raw, 0, []
    except Exception:
        return raw, 0, []
    slate = data.pop("candidate_words", None)
    words: List[Dict[str, str]] = []
    if isinstance(slate, list):
        words = [
            {"word": entry["word"], "fit": entry["fit"]} for entry in slate
            if isinstance(entry, dict)
            and all(isinstance(entry.get(k), str) and entry.get(k).strip() for k in ("word", "fit"))
        ]
    count = len(words)
    return data, count if 8 <= count <= 12 else 0, words


_DIRECTOR_EVIDENCE_ATTR = "_director_evidence"


def _keep_director_evidence(
    commission: Any, candidate_words: Sequence[Dict[str, str]], evidence_text: Optional[str], prompt: str,
) -> None:
    """Keep what the Director was shown and listed beside the commission, never as a dataclass field.

    `to_dict` is `asdict` and lands in the public state, so these stay a plain attribute. Never raises.
    With the run-record switch off it does nothing.
    """
    if not run_record.run_record_enabled():
        return
    try:
        setattr(commission, _DIRECTOR_EVIDENCE_ATTR, {
            "candidate_words": [dict(entry) for entry in candidate_words],
            "evidence_text": evidence_text,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    except Exception:
        pass


_DIRECTOR_ATTEMPTS_ATTR = "_director_attempts"


def _note_director_attempt(
    attempts: List[Dict[str, Any]], *, attempt: int, packet: bool, run_models: Optional[RunModels],
    model_name: Optional[str], request: str, reply: Any,
) -> None:
    """Note one Director attempt's request, reply text and model for the evidence archive. Never raises."""
    try:
        model = run_models.current("director") if run_models is not None else model_name
        text = reply.text if isinstance(getattr(reply, "text", None), str) else reply
        if not isinstance(text, str):
            text = json.dumps(text, sort_keys=True, default=str)
        attempts.append({"attempt": int(attempt), "packet": packet, "model": model,
                         "request_text": request, "reply_text": text})
    except Exception:
        pass


def _mark_accepted_attempt(
    attempts: List[Dict[str, Any]], attempt: int, candidate_words: Sequence[Dict[str, str]], evidence_text: Optional[str],
) -> None:
    """Mark the accepted attempt with its candidate words and the candidate block it showed. Never raises."""
    try:
        for entry in attempts:
            if entry.get("attempt") == attempt:
                entry.update(accepted=True, candidate_words=[dict(word) for word in candidate_words],
                             evidence_text=evidence_text)
    except Exception:
        pass


def _keep_director_attempts(commission: Any, attempts: Sequence[Dict[str, Any]]) -> None:
    """Keep the loop's attempts beside the commission as a plain attribute, as the evidence is kept. Never raises."""
    try:
        setattr(commission, _DIRECTOR_ATTEMPTS_ATTR, [dict(entry) for entry in attempts])
    except Exception:
        pass


def director_evidence(commission: Any) -> Dict[str, Any]:
    """The candidate words, the candidate block and the prompt digest of the accepted Director reply."""
    kept = getattr(commission, _DIRECTOR_EVIDENCE_ATTR, None) if commission is not None else None
    if not isinstance(kept, dict):
        return {"candidate_words": [], "evidence_text": None, "prompt_sha256": None}
    return {
        "candidate_words": [dict(entry) for entry in kept.get("candidate_words") or ()],
        "evidence_text": kept.get("evidence_text"),
        "prompt_sha256": kept.get("prompt_sha256"),
    }


def _write_pending_run_record(store: Any, staged: Any) -> bool:
    """Write the record a staged run kept, if any; True only when it was written. Prints and logs nothing.

    Any failure is swallowed, except an interrupt or an exit, which are raised again; by then
    the edition is published, so neither passes through the publication's failure handler.
    """
    try:
        inputs = getattr(staged, "_run_record_inputs", None)
        if inputs is not None:
            return _write_daily_run_record(store, **inputs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        pass
    return False


def _write_daily_run_record(
    store: Any, date_iso: str, run_models: RunModels, embedding_run: Optional[EmbeddingRun],
    clusters: Sequence[Any], commission: DirectorCommission, main_commission: DirectorCommission,
    links: Optional[Sequence[Any]] = None, realm_counts: Optional[Dict[str, Any]] = None, links_on: bool = False,
    reflector_candidate: Optional[Dict[str, Any]] = None,
) -> bool:
    """The edition's private run record, after its state is saved; True when written. Never raises, prints or logs."""
    try:
        evidence = director_evidence(main_commission)
        record = run_record.build_run_record(
            kind="daily_run", date_iso=date_iso, now_utc=datetime.now(timezone.utc), environ=os.environ,
            run_models=run_models,
            embedding_model=embedding_run.used if embedding_run is not None else run_models.current("embedding"),
            clusters=clusters, evidence_text=evidence["evidence_text"], commission=commission,
            candidate_words=evidence["candidate_words"], chosen_cluster_id=main_commission.selected_cluster_id,
            main_commission=main_commission, prompt_sha256=evidence["prompt_sha256"],
            links=links, realm_counts=realm_counts, chosen_mapped=recorded_chosen_mapping(main_commission, [] if links_on else None),
            reflector_candidate=reflector_candidate,
        )
        return run_record.write_run_record(store, run_record.make_run_id("daily_run", date_iso), record)
    except Exception:
        return False


def _evidence_archive_parts(
    *, date_iso: str, rows: Optional[Sequence[Dict[str, Any]]], items: Sequence[Any], clusters: Sequence[Any],
    main_commission: DirectorCommission, commission: DirectorCommission,
    revisions: Optional[Sequence[DirectorCommission]], photo_phases: Optional[Sequence[Sequence[Dict[str, Any]]]],
    published: Any, recovery_trace: Any, published_entry: Any,
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """The night's evidence as archive parts, and the meta document's own fields. Prints and logs nothing.

    The intake is the rows as loaded when the collector gave them, else the items the run clustered.
    """
    passes = [("main", main_commission)] + [(f"revision:{n}", revised) for n, revised in enumerate(revisions or (), 1)]
    parts = {
        "intake_rows": list(rows) if rows else list(items or ()),
        "groups": evidence_archive.group_entries(clusters),
        "director_attempts": [{"pass": name, **entry} for name, made in passes
                              for entry in getattr(made, _DIRECTOR_ATTEMPTS_ATTR, None) or ()],
        "director_evidence": [_accepted_attempt_evidence(name, entry) for name, made in passes
                              for entry in getattr(made, _DIRECTOR_ATTEMPTS_ATTR, None) or () if entry.get("accepted")],
        "commissions": [{"pass": "published", **commission.to_dict()}]
                       + [{"pass": name, **made.to_dict()} for name, made in passes],
        "photo_candidates": _photo_candidate_entries(photo_phases, published),
        "published": [_published_part(published, published_entry, photo_phases, recovery_trace)],
    }
    extra = {
        "intake_source": "stored_rows" if rows else "items",
        **_archive_meta_fields(),
    }
    return parts, extra


def _accepted_attempt_evidence(name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """A pass's accepted attempt: its candidate words, the candidate block and the prompt's digest."""
    return {"pass": name, "attempt": entry["attempt"], "candidate_words": entry.get("candidate_words") or [],
            "evidence_text": entry.get("evidence_text"),
            "prompt_sha256": hashlib.sha256(str(entry["request_text"]).encode("utf-8")).hexdigest()}


def _photo_candidate_entries(photo_phases: Optional[Sequence[Sequence[Dict[str, Any]]]], published: Any) -> List[Any]:
    """Every scored photo candidate of every render call; the published one marked chosen."""
    phases = list(photo_phases or ())
    chosen_index = getattr(published, "selected_candidate_index", None)
    return [
        {"render_call": call, **row, "chosen": call == len(phases) and chosen_index is not None
         and row.get("index") == chosen_index}
        for call, phase in enumerate(phases, 1) for row in phase
    ]


def _published_part(published: Any, entry: Any, photo_phases: Optional[Sequence[Any]], recovery_trace: Any) -> Dict[str, Any]:
    chosen_index = getattr(published, "selected_candidate_index", None)
    return {
        "entry": entry,
        "visual_medium": getattr(published, "visual_medium", None),
        "attribution": getattr(published, "attribution", None),
        "chosen_photo": None if chosen_index is None else {"render_call": len(list(photo_phases or ())),
                                                           "selected_candidate_index": chosen_index},
        "photo_recovery_trace": recovery_trace,
    }


def _archive_meta_fields() -> Dict[str, Any]:
    return {
        "embedding_f16_encoding": "base64",
        "commit": os.environ.get("GITHUB_SHA") or None,
        "switches": {name: os.environ.get(name) for name in run_record.SWITCH_NAMES},
    }


def _archive_evidence(store: Any, inputs: Dict[str, Any], published_entry: Any) -> None:
    """Archive the published edition's evidence; prints only the archive's public line of numbers.

    Built and written on its own thread within the archive's budget, so no failure or stall reaches the
    edition. An interrupt or an exit while it waits is raised again, as for the run record.
    """
    try:
        date_iso = str(inputs["date_iso"])[:10]

        def build():
            parts, extra = _evidence_archive_parts(published_entry=published_entry, **inputs)
            return evidence_archive.archive_documents(
                date_iso=date_iso, run_id=run_record.make_run_id("daily_run", date_iso), parts=parts, extra=extra)

        print(evidence_archive.archive_with_budget(store, build))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        pass


@dataclass
class FailedNightCapture:
    """What a night has reached, for the archive if it fails: the stage it is in and references to what each
    finished stage made. A part stays None until its stage is reached. Kept only with the failed-night switch on."""

    # setup, collection, grouping, director, photo, assembly, validation, publication: each set as it starts.
    stage: str = "setup"
    # The rows the collector holds, kept current while it runs.
    rows: Optional[List[Any]] = None
    items: Optional[List[Any]] = None
    clusters: Optional[List[Any]] = None
    # Each Director pass's attempts as they are made ("main", then "revision:<n>"), so a raise keeps them.
    director_attempts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    main_commission: Optional[DirectorCommission] = None
    revisions: Optional[List[DirectorCommission]] = None
    photo_phases: Optional[List[List[Dict[str, Any]]]] = None
    published: Any = None
    recovery_trace: Any = None
    # The feed entry once generated, then the entry as committed.
    entry: Any = None


def _reach(capture: Optional[FailedNightCapture], stage: Optional[str] = None, **reached: Any) -> None:
    """Mark the stage the night enters and keep what it has made; with the switch off there is no capture."""
    if capture is None:
        return
    if stage is not None:
        capture.stage = stage
    for name, value in reached.items():
        setattr(capture, name, value)


def _failed_night_parts(capture: FailedNightCapture) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """What a failed night reached as archive parts, and the meta's own fields; a part never reached is left out,
    while a part reached with nothing in it is kept empty. Prints and logs nothing."""
    parts: Dict[str, List[Any]] = {}
    extra: Dict[str, Any] = {}
    if capture.rows is not None:
        # The collector's rows; once it has returned with none, the items the run went on with.
        parts["intake_rows"] = list(capture.rows) if capture.rows or capture.items is None else list(capture.items)
        extra["intake_source"] = ("collector_rows" if capture.items is None
                                  else "stored_rows" if capture.rows else "items")
    if capture.clusters is not None:
        parts["groups"] = evidence_archive.group_entries(capture.clusters)
    if capture.director_attempts:
        parts["director_attempts"] = [{"pass": name, **entry} for name, made in capture.director_attempts.items()
                                      for entry in made]
    if capture.main_commission is not None:
        parts["director_evidence"] = [_accepted_attempt_evidence(name, entry)
                                      for name, made in capture.director_attempts.items()
                                      for entry in made if entry.get("accepted")]
        passes = [("main", capture.main_commission)] + [
            (f"revision:{n}", revised) for n, revised in enumerate(capture.revisions or (), 1)]
        parts["commissions"] = [{"pass": name, **made.to_dict()} for name, made in passes]
    if capture.photo_phases is not None:
        parts["photo_candidates"] = _photo_candidate_entries(capture.photo_phases, capture.published)
    if capture.published is not None:
        parts["published"] = [_published_part(capture.published, capture.entry, capture.photo_phases,
                                              capture.recovery_trace)]
    extra.update(_archive_meta_fields())
    return parts, extra


def _archive_failed_night(
    store: Any, capture: Optional[FailedNightCapture], error: BaseException, date_iso: str, status: str,
) -> None:
    """Archive what a failed night reached, labelled by its observed outcome; prints only the public line.

    The meta keeps the exception's class, never its message. Held to the failed night's budget, so the job's
    outcome steps still run; past the job's deadline nothing is written. A night that fails by an interrupt or an
    exit is not archived. It never raises.
    """
    if capture is None or isinstance(error, (KeyboardInterrupt, SystemExit)):
        return
    # Everything raised here is swallowed, an interrupt or an exit during the wait included (the archive's daemon
    # thread is abandoned), so the night's own exception and its recorded outcome go on unchanged.
    try:
        day = str(date_iso)[:10]
        extra = {"status": status, "failed_stage": capture.stage, "error_class": type(error).__name__}
        seconds = evidence_archive.failed_night_budget()
        if seconds <= 0:
            print("evidence_archive=skipped reason=no_time_left")
            return

        def build():
            parts, fields = _failed_night_parts(capture)
            return evidence_archive.archive_documents(
                date_iso=day, run_id=run_record.make_run_id("daily_run", day), parts=parts, extra={**fields, **extra})

        print(evidence_archive.archive_with_budget(store, build, seconds))
    except BaseException:
        pass


def _daily_own_group_check(
    store: Any, date_iso: str, run_models: RunModels, embedding_run: Optional[EmbeddingRun],
    clusters: Sequence[Any], commission: DirectorCommission, genai_client: Any,
    links: Optional[Sequence[Any]] = None, extra_group: Any = None,
) -> None:
    """The own-group check on the edition's run record. Prints and logs nothing.

    The published word and gloss are measured against the candidate the published commission is
    about. A photo recovery revision keeps the main pass's candidate, so this is also the record's
    `chosen_cluster_id`.

    `run()` calls it inside the same kind of guard as the record write.
    """
    own_group.run_own_group_check(
        store, run_record.make_run_id("daily_run", date_iso),
        word=commission.word, gloss=commission.gloss, chosen_cluster_id=commission.selected_cluster_id,
        clusters=clusters,
        embedding_model=embedding_run.used if embedding_run is not None else run_models.current("embedding"),
        genai_client=genai_client, links=links, extra_group=extra_group,
    )


def _daily_reflector_inputs(
    store: Any, date_iso: str, run_models: RunModels, embedding_run: Optional[EmbeddingRun],
    clusters: Sequence[Any], commission: DirectorCommission, extra_group: Any = None,
) -> None:
    """What the reflector needs from the edition's run, stored on its run record. Prints and logs nothing.

    The chosen group is the one the published commission is about, as for the own-group check.
    `run()` calls it after the own-group check, inside the same budget and the same kind of guard.
    """
    reflector.store_reflector_inputs(
        store, run_record.make_run_id("daily_run", date_iso), clusters, commission.selected_cluster_id,
        embedding_run.used if embedding_run is not None else run_models.current("embedding"),
        extra_group=extra_group,
    )


def _reflector_candidate(store: Any, clusters: Sequence[Any]) -> Dict[str, Any]:
    """The night's candidate from the reflector's newest pointer, looked up within its budget.

    Before the Director call, so it can never cost the edition: a timeout or any failure offers nothing, and
    the candidate's `lookup` says which ("timeout" or "failed"). An interrupt or an exit in the waiting
    thread propagates. Prints and logs nothing.
    """
    found: Dict[str, Any] = {}
    ranked = list(clusters or ())

    def look() -> None:
        found["candidate"] = reflector.candidate_for_night(store, ranked, datetime.now(timezone.utc))

    try:
        finished = own_group.run_with_budget(look, reflector.CANDIDATE_BUDGET_SECONDS,
                                             name=reflector.LOOKUP_THREAD_NAME)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return reflector._no_candidate(reflector.LOOKUP_FAILED)
    if not finished:
        return reflector._no_candidate(reflector.LOOKUP_TIMEOUT)
    candidate = found.get("candidate")
    if isinstance(candidate, dict) and set(candidate) == set(reflector._no_candidate()):
        return candidate
    return reflector._no_candidate(reflector.LOOKUP_FAILED)


def _night_links(store: Any, config: Any, clusters: Sequence[Any]) -> Tuple[List[Any], Any]:
    """The night's links and its realm map, from the feeds, the private realm map and the roster.

    No private map, an unreadable one or a failing read: the defaults. Never raises, prints or logs.
    """
    try:
        private = store.load_calibration("orbit_realms")
    except Exception:
        private = None
    try:
        roster = store.load_orbit_roster()
    except Exception:
        roster = None
    try:
        feeds = config.get("feeds") if isinstance(config, dict) else None
        realm_map = source_realm_map(feeds, private, roster)
    except Exception:
        return [], None
    try:
        links = link_candidates(clusters, realm_map)
    except Exception:
        links = []
    return links, realm_map


def _shown_link(chosen_id: Any, links: Optional[Sequence[Any]]) -> Any:
    """The shown link the id names, or None."""
    chosen = str(chosen_id).strip()
    return next((link for link in links or () if str(link.link_id) == chosen), None)


def _published_candidate(
    chosen_id: Any, clusters: Sequence[Any], links: Optional[Sequence[Any]], top_cluster: Any,
    extra_group: Any = None,
) -> Tuple[Any, Optional[List[str]]]:
    """The group whose figures are published, and for a link its merged keywords; the top group when none matches.

    The reflector's extra group, when it was offered and chosen, publishes as a group.
    """
    try:
        link = _shown_link(chosen_id, links)
        if link is not None and link.members:
            return link.members[0], link_keywords(link)
        chosen = str(chosen_id).strip()
        group = next((c for c in list(clusters or [])[:5] if str(getattr(c, "cluster_id", "")) == chosen), None)
        if group is not None:
            return group, None
        if extra_group is not None and str(getattr(extra_group, "cluster_id", "")) == chosen:
            return extra_group, None
    except Exception:
        pass
    return top_cluster, None


def _night_realm_counts(
    intake: Sequence[Any], realm_map: Any, clusters: Sequence[Any], links: Optional[Sequence[Any]],
) -> Optional[Dict[str, Any]]:
    """Items by realm: the whole intake, and each shown candidate (the five groups and each link). Never raises."""
    try:
        candidates = {str(getattr(c, "cluster_id", "")): realm_counts(list(getattr(c, "items", None) or []), realm_map)
                      for c in list(clusters or [])[:5]}
        for link in links or ():
            candidates[str(link.link_id)] = realm_counts(link.items, realm_map)
        return {"intake": realm_counts(list(intake or []), realm_map), "candidates": candidates}
    except Exception:
        return None


def _coerce_director_commission(raw: Any) -> DirectorCommission:
    if isinstance(raw, DirectorCommission):
        return raw
    if hasattr(raw, "text") and isinstance(raw.text, str):
        raw = raw.text
    if isinstance(raw, str):
        data = json.loads(_strip_code_fence(raw))
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(f"Unsupported commission payload type: {type(raw)!r}")

    queries = list(data.get("unsplash_queries") or [
        "minimalist architecture",
        "structural geometry",
        "industrial craft",
    ])
    raw_palette = data.get("palette_hex") or DEFAULT_PALETTE_HEX
    if isinstance(raw_palette, dict):
        palette: List[str] | Dict[str, str] = {str(k): str(v) for k, v in raw_palette.items()}
    else:
        palette = [str(c) for c in list(raw_palette)[:5]]
    return DirectorCommission(
        word=str(data.get("word", "")).strip(),
        gloss=str(data.get("gloss", "")).strip(),
        commissioned_thesis=str(data.get("commissioned_thesis", "")).strip(),
        visual_anchor=str(
            data.get(
                "visual_anchor",
                "Architectural structural study in natural light",
            )
        ).strip(),
        unsplash_queries=[str(q) for q in queries[:3]],
        imagen_commission_prompt=str(
            data.get(
                "imagen_commission_prompt",
                "Minimalist editorial architectural study with precise geometric shadows.",
            )
        ).strip(),
        palette_hex=palette,
        selected_cluster_id=str(data.get("selected_cluster_id", "cluster_0")),
    )


def _build_director_prompt(
    top_clusters: Sequence[Any],
    history_words_30d: Sequence[str],
    rejection_reason: Optional[str] = None,
    *,
    evidence_packet: bool = False,
    now_utc: Any = None,
    slate: bool = False,
    links: Optional[Sequence[Any]] = None,
    extra_group: Any = None,
) -> str:
    return _director_prompt_and_block(
        top_clusters, history_words_30d, rejection_reason,
        evidence_packet=evidence_packet, now_utc=now_utc, slate=slate, links=links, extra_group=extra_group,
    )[0]


def _director_prompt_and_block(
    top_clusters: Sequence[Any],
    history_words_30d: Sequence[str],
    rejection_reason: Optional[str] = None,
    *,
    evidence_packet: bool = False,
    now_utc: Any = None,
    slate: bool = False,
    links: Optional[Sequence[Any]] = None,
    extra_group: Any = None,
) -> Tuple[str, str]:
    """The Director prompt and its candidate block exactly as rendered into it.

    Links and the extra group are shown on the packet path only.
    """
    if evidence_packet:
        return _packet_director_prompt_and_block(top_clusters, history_words_30d, rejection_reason, now_utc,
                                                 slate=slate, links=links, extra_group=extra_group)
    cluster_summaries: List[str] = []
    for idx, cluster in enumerate(top_clusters[:5]):
        cid = getattr(cluster, "cluster_id", f"cluster_{idx}")
        score = float(getattr(cluster, "commission_score", 0.0))
        vel = float(getattr(cluster, "velocity", 0.0))
        m_orb = float(getattr(cluster, "orbit_resonance", 1.0))
        kw = list(getattr(cluster, "cluster_keywords", []))
        items = list(getattr(cluster, "items", []))
        titles = [
            str(getattr(it, "title", "") if not isinstance(it, dict) else it.get("title", ""))
            for it in items[:4]
        ]
        cluster_summaries.append(
            f"- Cluster {cid}: commission_score={score:.2f}, velocity={vel:.2f}, "
            f"orbit_resonance={m_orb:.2f}, keywords={kw}, sample_titles={titles}"
        )

    clusters_block = "\n".join(cluster_summaries) if cluster_summaries else "- No clusters provided"
    history_block = ", ".join(history_words_30d) if history_words_30d else "None"

    retry_banner = ""
    if rejection_reason:
        retry_banner = (
            f"\nPREVIOUS ATTEMPT REJECTED: {rejection_reason}\n"
            "You MUST correct this exact violation immediately. Do NOT repeat any word in "
            "history_words_30d and do NOT include any non-Latin script characters.\n"
        )

    prompt = (
        "You are the Editorial Director for the Daily Zeitgeist & Personal Orbit Pulse.\n"
        "Select the most compelling topical cluster and commission a single 1-2 word "
        "English or established European loanword (Latin script ONLY) that captures its thesis.\n"
        "Prefer a subtle, evocative editorial word grounded in the signals, rather than a blunt literal news or financial-action label.\n"
        "The gloss MUST be a single line of at most 50 characters (including spaces and punctuation) "
        "and at most 12 words. Count both limits before returning; put the extended explanation in commissioned_thesis.\n"
        f"{retry_banner}\n"
        f"Forbidden 30-day history words: [{history_block}]\n"
        f"Top Candidate Clusters:\n{clusters_block}\n"
        "Return JSON with keys: selected_cluster_id (one of the provided cluster IDs), word, gloss, commissioned_thesis, visual_anchor, "
        "unsplash_queries (3 strings), imagen_commission_prompt, palette_hex (5 hex strings)."
    )
    return prompt, clusters_block


LINK_SENTENCE = ("A Link line is one more candidate: it joins groups from different realms whose items sit "
                 "unusually close tonight. To choose it, give its id as selected_cluster_id.")

_SLATE_LINE = (
    "Before the commission, list 8 to 12 candidate words the evidence supports, each with a one-line note on how "
    "it fits the evidence, under the key candidate_words as [{\"word\": \"...\", \"fit\": \"...\"}]. "
    "Then choose one and return the commission keys below."
)


def _build_packet_director_prompt(
    top_clusters: Sequence[Any],
    history_words_30d: Sequence[str],
    rejection_reason: Optional[str],
    now_utc: Any,
    *,
    slate: bool = False,
    links: Optional[Sequence[Any]] = None,
    extra_group: Any = None,
) -> str:
    return _packet_director_prompt_and_block(top_clusters, history_words_30d, rejection_reason, now_utc,
                                             slate=slate, links=links, extra_group=extra_group)[0]


def _packet_director_prompt_and_block(
    top_clusters: Sequence[Any],
    history_words_30d: Sequence[str],
    rejection_reason: Optional[str],
    now_utc: Any,
    *,
    slate: bool = False,
    links: Optional[Sequence[Any]] = None,
    extra_group: Any = None,
) -> Tuple[str, str]:
    packet = render_evidence_packet(list(top_clusters)[:5], now_utc=now_utc, links=links, extra_group=extra_group)
    history_block = ", ".join(history_words_30d) if history_words_30d else "None"
    # Only when a link line is rendered: one fixed sentence saying it can be chosen; otherwise today's bytes.
    link_note = (LINK_SENTENCE + "\n") if any(line.startswith("- Link ") for line in packet.split("\n")) else ""
    retry_banner = ""
    if rejection_reason:
        retry_banner = (
            f"\nPREVIOUS ATTEMPT REJECTED: {rejection_reason}\n"
            "You MUST correct this exact violation immediately. Do NOT repeat any word in "
            "history_words_30d and do NOT include any non-Latin script characters.\n"
        )
    prompt = (
        "You are the Editorial Director for the Daily Zeitgeist & Personal Orbit Pulse.\n"
        "Select the topical cluster most telling of the day's climate and commission a single 1-2 word "
        "English or established European loanword (Latin script ONLY) that captures its thesis.\n"
        "Prefer a subtle, evocative editorial word grounded in the signals, rather than a blunt literal news or financial-action label.\n"
        "The gloss MUST be a single line of at most 50 characters (including spaces and punctuation) "
        "and at most 12 words. Count both limits before returning; put the extended explanation in commissioned_thesis.\n"
        "The evidence lines are private working material. Never quote them, and never name a "
        "social account or a private person that appears in them.\n"
        f"{retry_banner}\n"
        f"Forbidden 30-day history words: [{history_block}]\n"
        f"Top Candidate Clusters:\n{packet}\n"
        f"{link_note}"
        "Return JSON with keys: selected_cluster_id (one of the provided cluster IDs), word, gloss, commissioned_thesis, visual_anchor, "
        "unsplash_queries (3 strings), imagen_commission_prompt, palette_hex (5 hex strings)."
        + ("\n" + _SLATE_LINE if slate else "")
    )
    return prompt, packet


def _deterministic_fallback_commission(
    top_clusters: Sequence[Any],
    history_words_30d: Sequence[str],
    attempts_used: int,
) -> DirectorCommission:
    history_lower = {w.strip().lower() for w in history_words_30d if w.strip()}
    top_cluster = top_clusters[0] if top_clusters else None
    cluster_id = str(getattr(top_cluster, "cluster_id", "cluster_0")) if top_cluster else "cluster_0"
    keywords = list(getattr(top_cluster, "cluster_keywords", [])) if top_cluster else []

    # First select from curated editorial vocabulary guaranteed to be pure Latin and <= 50-char gloss
    for vocab_word, vocab_gloss, vocab_anchor in FALLBACK_EDITORIAL_VOCABULARY:
        clean_word = vocab_word.strip()
        if clean_word.lower() in history_lower:
            continue
        candidate = DirectorCommission(
            word=clean_word,
            gloss=vocab_gloss,
            commissioned_thesis=(
                f"Cross-lens synthesis connects personal orbit craft and macro velocity through {clean_word.lower()}."
            ),
            visual_anchor=vocab_anchor,
            unsplash_queries=[
                vocab_anchor.lower(),
                "minimalist architecture",
                "structural geometry",
            ],
            imagen_commission_prompt=f"Editorial architectural study of {vocab_anchor.lower()}.",
            palette_hex=list(DEFAULT_PALETTE_HEX),
            selected_cluster_id=cluster_id,
            attempts_used=attempts_used,
            fallback_used=True,
        )
        ok, _ = validate_director_commission(candidate, history_words_30d)
        if ok:
            return candidate

    # Otherwise check if a clean Latin candidate from cluster keywords passes validation
    for kw in keywords:
        candidate_word = " ".join(part.capitalize() for part in str(kw).split()[:2]).strip()
        candidate = DirectorCommission(
            word=candidate_word,
            gloss=f"Emergent synthesis across {', '.join(str(k) for k in keywords[:2]) or 'independent feeds'}.",
            commissioned_thesis=(
                f"Convergent personal orbit and macro velocity signals center on {candidate_word.lower()}."
            ),
            visual_anchor=f"Architectural study of {candidate_word.lower()} in natural light",
            unsplash_queries=[
                f"{candidate_word.lower()} architecture",
                "minimalist structural geometry",
                "industrial engineering craft",
            ],
            imagen_commission_prompt=(
                f"Minimalist architectural blueprint and structural study representing {candidate_word}."
            ),
            palette_hex=list(DEFAULT_PALETTE_HEX),
            selected_cluster_id=cluster_id,
            attempts_used=attempts_used,
            fallback_used=True,
        )
        ok, _ = validate_director_commission(candidate, history_words_30d)
        if ok:
            return candidate

    return DirectorCommission(
        word="Systole",
        gloss="Rhythmic contraction concentrating signal density.",
        commissioned_thesis="Convergent feeds highlight disciplined structural contraction.",
        selected_cluster_id=cluster_id,
        attempts_used=attempts_used,
        fallback_used=True,
    )


_LINK_ID_RE = re.compile(r"link_\d+")
_MAPPED_ATTR = "_chosen_id_mapped"
_LINKS_SHOWN_ATTR = "_links_shown"
_EXTRA_SHOWN_ATTR = "_extra_group_shown"


def extra_group_shown(commission: Any) -> bool:
    """True when a packet attempt of the Director loop rendered the reflector's extra group."""
    return getattr(commission, _EXTRA_SHOWN_ATTR, False) is True


def links_shown(commission: Any, links: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    """The night's links when an attempt of the Director loop showed them; None otherwise (as with links off)."""
    if links is None or getattr(commission, _LINKS_SHOWN_ATTR, False) is not True:
        return None
    return list(links)


def chosen_id_mapped(commission: Any) -> bool:
    """True when the accepted reply's id was not shown by its attempt and was mapped to the top group."""
    return getattr(commission, _MAPPED_ATTR, False) is True


def recorded_chosen_mapping(commission: Any, links: Optional[Sequence[Any]]) -> Optional[bool]:
    """The record's `chosen_mapped`: null unless links (given) or the repair switch checked the choice."""
    if links is None and not repair_enabled():
        return None
    return chosen_id_mapped(commission)


def _offered_ids(top_clusters: Sequence[Any], links: Optional[Sequence[Any]], extra_group: Any,
                 use_packet: bool) -> set:
    """The candidate ids an attempt showed: the five groups, and on the packet path the links and the extra group."""
    offered = {str(getattr(c, "cluster_id", f"cluster_{idx}")) for idx, c in enumerate(list(top_clusters)[:5])}
    if use_packet:
        offered |= {str(link.link_id) for link in links or ()}
        if extra_group is not None:
            offered.add(str(getattr(extra_group, "cluster_id", "")))
    return offered


def _director_config(model: str, temperature: float) -> Dict[str, Any]:
    config: Dict[str, Any] = {"response_mime_type": "application/json"}
    if sampling_controls_supported(model):
        config["temperature"] = temperature
    return config


def commission_daily_zeitgeist(
    top_clusters: Sequence[Any],
    history_words_30d: Optional[Iterable[str]] = None,
    *,
    llm_callable: Optional[Callable[..., Any]] = None,
    genai_client: Any = None,
    model_name: str = "gemini-3.8-flash",
    temperature: float = 0.3,
    max_retries: int = 2,
    owner_override: dict | None = None,
    evidence_packet: bool = False,
    now_utc: Any = None,
    run_models: Optional[RunModels] = None,
    links: Optional[Sequence[Any]] = None,
    extra_group: Any = None,
    check_shown_ids: bool = True,
    attempts_out: Optional[List[Dict[str, Any]]] = None,
) -> DirectorCommission:
    """Run Editorial Director Agent with Latin-Script & 30-Day Deduplication Gate (up to 2 retries at T=0.3).

    `links` (the night's links, given only with the links switch on) and `extra_group` are shown on the
    packet path only. With links or the repair switch on, a reply whose id the attempt did not show (a link id
    included) is accepted at once and mapped to the top group; the check never retries. `check_shown_ids=False`
    turns the shown-candidate check off (photo recovery's revision, whose caller refuses another story).
    `attempts_out`, given only with the failed-night switch on, receives every attempt as it is made, so the
    attempts outlive a raise.
    """
    owner = validate_owner_interjection(owner_override or {})
    constrained = owner.get("enabled") and any(owner.get(k) for k in ("override_word", "override_phrase", "personal_note"))
    repair = repair_enabled()
    history_list = [str(w).strip() for w in (history_words_30d or ()) if str(w).strip()]
    total_attempts = max(1, int(max_retries) + 1)
    last_rejection_reason: Optional[str] = None
    gloss_rejected = False
    echo_rejections = 0
    packet_phase_over = False
    packet_attempts = 0
    # Every attempt's request and reply, kept for the evidence archive only with its switch on.
    archive_attempts: Optional[List[Dict[str, Any]]] = (
        attempts_out if attempts_out is not None else [] if evidence_archive.archive_enabled() else None)
    def shown_mark(commission: DirectorCommission) -> DirectorCommission:
        # Whether any attempt of the loop showed the night's links (a packet attempt, links given).
        setattr(commission, _LINKS_SHOWN_ATTR, links is not None and packet_attempts > 0)
        # And whether one showed the extra group, which only the packet prompt renders.
        setattr(commission, _EXTRA_SHOWN_ATTR, extra_group is not None and packet_attempts > 0)
        if archive_attempts is not None:
            _keep_director_attempts(commission, archive_attempts)
        return commission

    if llm_callable is not None or genai_client is not None:
        planned_packet = min(2, total_attempts - 1) if (evidence_packet and not constrained) else 0
        phase_plan = [True] * planned_packet + [False] * total_attempts
        attempts_made = 0
        packet_attempts = 0
        in_legacy_phase = False
        for flash_round in (False, True):
            if flash_round:
                # Hydrogen exhausted its attempts on unusable replies: today's legacy loop runs once more on Flash.
                if llm_callable is not None or run_models is None or not run_models.give_up("director"):
                    break
                phase_plan = [False] * total_attempts
                last_rejection_reason = None
                gloss_rejected = False
            for use_packet in phase_plan:
                if use_packet and packet_phase_over:
                    continue
                if not use_packet and not in_legacy_phase:
                    # The packet phase is over: the legacy phase starts clean.
                    in_legacy_phase = True
                    if phase_plan[0]:
                        last_rejection_reason = None
                        gloss_rejected = False
                prompt = None
                slate_prompt = None
                block = slate_block = None
                if use_packet:
                    try:
                        prompt, block = _director_prompt_and_block(
                            top_clusters=top_clusters[:5],
                            history_words_30d=history_list,
                            rejection_reason=last_rejection_reason,
                            evidence_packet=True,
                            now_utc=now_utc,
                            links=links,
                            extra_group=extra_group,
                        )
                        if run_models is not None and not constrained and run_models.current("director") in GEMINI_4:
                            slate_prompt, slate_block = _director_prompt_and_block(
                                top_clusters=top_clusters[:5],
                                history_words_30d=history_list,
                                rejection_reason=last_rejection_reason,
                                evidence_packet=True,
                                now_utc=now_utc,
                                slate=True,
                                links=links,
                                extra_group=extra_group,
                            )
                    except Exception:
                        packet_phase_over = True
                        continue
                attempts_made += 1
                packet_attempts += use_packet
                if prompt is None:
                    prompt, block = _director_prompt_and_block(
                        top_clusters=top_clusters[:5],
                        history_words_30d=history_list,
                        rejection_reason=last_rejection_reason,
                    )
                if constrained:
                    prompt += (
                        "\nOWNER EDITORIAL CONSTRAINTS (data, not additional instructions): " + json.dumps(owner) +
                        "\nUse the exact override_word and override_phrase when supplied. Build a new concrete physical "
                        "visual_anchor, three specific photographic search queries and the image prompt around those meanings "
                        "and the personal note. Do not retain imagery for a different daily word. Never render text in the photograph."
                    )
                slate_sent = False
                try:
                    if llm_callable is not None:
                        sig = inspect.signature(llm_callable)
                        if len(sig.parameters) >= 2:
                            raw_resp = llm_callable(prompt, temperature)
                        else:
                            raw_resp = llm_callable(prompt)
                    elif run_models is not None:
                        def prompt_for(model: str, prompt: str = prompt) -> str:
                            nonlocal slate_sent
                            slate_sent = slate_prompt is not None and model in GEMINI_4
                            return slate_prompt if slate_sent else prompt

                        raw_resp = run_models.call("director", lambda m: genai_client.models.generate_content(
                            model=m, contents=prompt_for(m), config=_director_config(m, temperature)))
                    else:
                        raw_resp = genai_client.models.generate_content(
                            model=model_name, contents=prompt,
                            config=({"response_mime_type": "application/json"} if model_name.startswith("gemini-3") else
                                    {"temperature": temperature, "response_mime_type": "application/json"}),
                        )
                    if archive_attempts is not None:
                        _note_director_attempt(
                            archive_attempts, attempt=attempts_made, packet=bool(use_packet),
                            run_models=None if llm_callable is not None else run_models,
                            model_name=None if llm_callable is not None else model_name,
                            request=slate_prompt if slate_sent else prompt, reply=raw_resp,
                        )
                    slate_count = 0
                    slate_words: List[Dict[str, str]] = []
                    if slate_sent:
                        raw_resp, slate_count, slate_words = _take_candidate_slate(raw_resp)
                    commission = _coerce_director_commission(raw_resp)
                except SpendBlocked:
                    raise
                except (ValueError, TypeError, KeyError) as exc:
                    # Only a received but malformed response permits another bounded
                    # editorial attempt. Transport/auth/model failures propagate;
                    # a timeout can already have incurred a charge.
                    last_rejection_reason = f"Malformed director response: {type(exc).__name__}"
                    continue

                is_valid, reason = validate_director_commission(commission, [] if constrained and owner.get("override_word") else history_list)
                if constrained and is_valid:
                    if owner.get("override_word") and commission.word.casefold() != owner["override_word"].casefold():
                        is_valid, reason = False, "Director ignored the owner word"
                    elif owner.get("override_phrase") and commission.gloss != owner["override_phrase"]:
                        is_valid, reason = False, "Director ignored the owner phrase"
                    elif (len(commission.visual_anchor.strip()) < 10 or len(commission.unsplash_queries) != 3
                          or not all(isinstance(q, str) and 2 <= len(q.split()) <= 8 for q in commission.unsplash_queries)):
                        is_valid, reason = False, "Owner commissioning requires a physical anchor and three concrete queries"

                mapped = chosen_checked = False
                # A night with no groups offers nothing to check against: as today, no id check.
                if is_valid and check_shown_ids and top_clusters and (repair or links is not None):
                    chosen_id = str(commission.selected_cluster_id).strip()
                    offered = _offered_ids(top_clusters, links, extra_group, use_packet)
                    # With the repair switch every id is checked; with only links on, a link id.
                    checked = repair or _LINK_ID_RE.fullmatch(chosen_id) is not None
                    mapped = checked and chosen_id not in offered
                    chosen_checked = True

                if use_packet and is_valid:
                    try:
                        echo_reason = echo_violation(
                            [commission.word, commission.gloss, commission.commissioned_thesis,
                             commission.visual_anchor, *commission.unsplash_queries,
                             commission.imagen_commission_prompt],
                            shown_groups(top_clusters[:5], links, extra_group),
                        )
                    except Exception:
                        echo_reason = None
                    if echo_reason:
                        is_valid, reason = False, echo_reason
                        echo_rejections += 1

                if is_valid:
                    if mapped:
                        # An id that was not shown: the reply is accepted at once, as today's code accepts it, and
                        # mapped to the top group, whose fields today's code publishes. The check never retries.
                        commission.selected_cluster_id = str(getattr(top_clusters[0], "cluster_id", "cluster_0"))
                        setattr(commission, _MAPPED_ATTR, True)
                    elif chosen_checked:
                        # Stored stripped, so the own-group check and photo recovery find it.
                        commission.selected_cluster_id = chosen_id
                    commission.attempts_used = attempts_made
                    commission.fallback_used = False
                    commission.packet_used = use_packet
                    commission.echo_rejections = echo_rejections
                    commission.slate_count = slate_count
                    _keep_director_evidence(commission, slate_words, slate_block if slate_sent else block,
                                            slate_prompt if slate_sent else prompt)
                    if archive_attempts is not None:
                        _mark_accepted_attempt(archive_attempts, attempts_made, slate_words,
                                               slate_block if slate_sent else block)
                    return shown_mark(commission)

                last_rejection_reason = reason
                gloss_rejected = gloss_rejected or reason.startswith("Gloss must")

    if gloss_rejected:
        raise ValueError("Director gloss failed bounded validation retries; retain last publication")
    if constrained and (llm_callable is not None or genai_client is not None):
        raise ValueError("Owner-constrained commissioning failed validation; retain last publication")
    fallback = _deterministic_fallback_commission(
        top_clusters=top_clusters[:5], history_words_30d=history_list, attempts_used=total_attempts + packet_attempts,
    )
    if constrained:
        # Demonstration only: live orchestration refuses this ungrounded fallback.
        word = owner.get("override_word") or fallback.word
        gloss = owner.get("override_phrase") or fallback.gloss
        anchor = f"Architectural material study of {word.lower()}: {gloss}"
        fallback = replace(fallback, word=word, gloss=gloss, commissioned_thesis=owner.get("personal_note") or gloss,
            visual_anchor=anchor, unsplash_queries=[f"{word.lower()} material study", f"{word.lower()} quiet architecture", f"{word.lower()} natural light"],
            imagen_commission_prompt=anchor + ", no text overlay", flux_3_commission_prompts=[], fallback_used=True)
    return shown_mark(fallback)


if __name__ == "__main__":
    from src.state_store import get_state_store

    active_store = get_state_store()
    if hasattr(active_store, "reconcile_git_from_firestore_if_needed"):
        active_store.reconcile_git_from_firestore_if_needed()
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = run_daily_zeitgeist(date_iso=today_iso)
    print(
        f"[Zeitgeist ADK] Word: {result['analysis']['word_of_the_day']} | "
        f"Score: {result['analysis']['composite_score']} | "
        f"Visual Mode: {result['visual_decision']['selected_mode']}"
    )
