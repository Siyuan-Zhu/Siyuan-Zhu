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
from .methodology import (
    CachedItem,
    FeedDocument,
    ScoredCluster,
    WeightedOrbitAnalysis,
    analyze_seven_bucket_orbit,
    cluster_and_score,
    default_curated_corpus,
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

from .spend_guard import SpendBlocked
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
    )


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
) -> List[CachedItem]:
    """Load, collect, or synthesize 96h Macro + 336h Orbit items with 768-dim L2-normalized embeddings."""
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
        if active_rows:
            from src.state_store import pack_ingestion_cache_pages
            pack_ingestion_cache_pages(date_str, active_rows, reserve_embedding=True)
            cache_before_embedding = dict(existing_cache_map)
            embedded_live = embed_missing_items(
                list(active_rows.values()),
                existing_cache=existing_cache_map,
                genai_client=active_genai_client,
                require_live=True,
            )
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
            if day_shard and hasattr(active_store, "save_ingestion_cache_day"):
                active_store.save_ingestion_cache_day(date_str, day_shard)
        else:
            existing_cache_map = {}

    items: List[CachedItem] = []
    for idx, row in enumerate(existing_cache_map.values()):
        coerced = _row_to_cached_item(row, idx, ref_dt_str)
        if coerced is not None:
            items.append(coerced)

    if should_live_collect:
        rows=[r for r in existing_cache_map.values() if r.get("origin") != "offline_demo"]
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
            raise IngestionUnavailable("Required diverse ORBIT, MACRO and Instagram coverage unavailable; retain last publication")
        health["publishable"]=True
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

        store = (
            self._state_store
            if self._state_store is not None
            else get_state_store(repo_root=self.repo_root)
        )
        if (self.repo_root / "data/publication_reconcile.json").exists():
            raise RuntimeError(
                "Resolve the pending canonical publication reconciliation before generating another publication"
            )
        expected = store.load_pulse_state()
        workspace = PublicationWorkspace(self.repo_root)
        committed = False
        installed = False
        journal = self.repo_root / "data/publication_reconcile.json"
        journal_created = False

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
            state = publish_bundle(
                store,
                payload,
                staged.last_published_pulse.to_feed_entry(),
                expected=expected,
                local_feed=feed,
                prepare=prepare,
            )
            committed = True
            workspace.install(state)
            installed = True
            from .visual_router import confirm_published_photo_download
            published_photo = staged.last_published_pulse
            if (published_photo.visual_medium == "PHOTOGRAPHY"
                    and published_photo.attribution.get("download_location")
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
            return (
                analysis,
                replace(decision, local_svg_path=str(self.svg_path)),
                telemetry,
            )
        except BaseException as exc:
            if committed or isinstance(exc, PublicationOutcomeUnknown):
                store.set_needs_git_reconcile(True)
                try:
                    marker = json.loads(journal.read_text())
                    marker["status"] = ("canonical_committed_download_tracking_pending" if installed else "canonical_committed_local_install_failed" if committed
                                        else "canonical_commit_outcome_unknown")
                    journal.write_text(json.dumps(marker) + "\n")
                except (OSError, ValueError):
                    pass  # The pre-dispatch journal remains the authoritative recovery signal.
                from .publication_commit import PublicationMirrorError

                raise PublicationMirrorError(
                    "Canonical publication committed or outcome unknown; reconcile retained bundle before retry"
                ) from exc
            if journal_created:
                journal.unlink(missing_ok=True)
            raise
        finally:
            workspace.close()

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
        from src.state_store import get_state_store

        date_iso = str(date_iso or datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10]
        config = load_orbit_config(self.config_path)
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
        collection_health = {}
        try:
            raw_signals = collect_96h_signals(
                self.repo_root,
                date_iso=date_iso,
                store=store,
                live_collect=live_collect,
                health=collection_health,
                now_epoch=now_epoch,
                offline_demo=offline_demo,
            )
        except IngestionUnavailable as exc:
            exc.public_collection_health = public_collection_health({**exc.public_collection_health, **collection_health})
            raise
        reference_time = _reference_clock(date_iso, now_epoch,
            _offline_demo_enabled(offline_demo) and not _live_collection_enabled(live_collect))
        scored_clusters = score_and_cluster_signals(
            raw_signals,
            now_utc=reference_time,
        )

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
            m = entry.get("visual_tier_used") or entry.get("visual_medium")
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

        commission = commission_daily_zeitgeist(
            scored_clusters,
            history_words_30d=history_words_30d,
            genai_client=genai_client,
            owner_override=owner,
            model_name=str(config.get("budget_guardrails", {}).get("primary_text_model", "gemini-3.8-flash")),
        )
        if (os.environ.get("ZEITGEIST_LIVE_LLM") == "1"
                and not _offline_demo_enabled(offline_demo) and commission.fallback_used):
            raise ValueError("Live Director commissioning unavailable; retain last publication")
        if (owner.get("enabled") and not _offline_demo_enabled(offline_demo)
                and any(owner.get(k) for k in ("override_word", "override_phrase", "personal_note"))
                and commission.fallback_used):
            raise ValueError("Owner-constrained live commissioning unavailable; retain last publication")
        self.last_commission = commission

        top_cluster = scored_clusters[0] if scored_clusters else None
        if (top_cluster is not None and collection_health.get("apify_batch_succeeded") is True
                and hasattr(store, "load_orbit_roster") and hasattr(store, "save_orbit_roster")):
            roster_before_bonus = store.load_orbit_roster()
            roster_with_bonus = apply_winning_cluster_bonus(roster_before_bonus, top_cluster.items,
                reference_time, apify_batch_succeeded=True)
            if roster_with_bonus != roster_before_bonus:
                store.save_orbit_roster(sanitize_orbit_roster_state(roster_with_bonus))
        orbit_res = float(top_cluster.orbit_resonance) if top_cluster is not None else 1.50
        vel_score = float(top_cluster.velocity) if top_cluster is not None else 2.50
        is_breakout = bool(getattr(top_cluster, "breakout_surge", False)) if top_cluster is not None else False
        c_keywords = list(top_cluster.cluster_keywords) if top_cluster is not None else [commission.word]
        c_sil = float(top_cluster.silhouette_score) if top_cluster is not None else 0.75

        # Stage 4: Multimodal Visual Critic Funnel & Dual-Resolution WebP/SVG Renderer
        curated_candidates = None
        live_unsplash_enabled = os.environ.get("ZEITGEIST_LIVE_UNSPLASH", "1").strip() != "0"
        has_live_unsplash_key = (
            bool(os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()) and live_unsplash_enabled
        )
        if requested_medium != "blueprint" and live_unsplash_enabled and not has_live_unsplash_key:
            from .visual_router import fetch_curated_library_candidates

            curated_candidates = fetch_curated_library_candidates(word=commission.word)

        def render_commission(current, *, photo_only=None):
            return select_and_render_pulse_assets(
                commission=current,
                medium_history_30d=medium_history_30d,
                archetype_history_30d=archetype_history_30d,
                breakout_surge=is_breakout,
                repo_root=self.repo_root,
                date_str=str(date_iso)[:10],
                candidates=[] if photo_only is False else curated_candidates,
                orbit_resonance_score=orbit_res,
                velocity_score=vel_score,
                cluster_keywords=c_keywords,
                cluster_silhouette=c_sil,
                mood_votes=tally,
                requested_medium="photo" if photo_only is True else requested_medium,
                require_semantic_rescore=bool(owner.get("enabled") and not _offline_demo_enabled(offline_demo)),
                photo_history_30d=feed_entries,
                judge_model=str(config.get("budget_guardrails", {}).get("multimodal_judge_model", "gemini-3.8-flash")),
                strict_photo_errors=genai_client is not None,
            )
        if genai_client is not None and requested_medium == "blueprint":
            from .photo_recovery import ai_exception_allowed
            modes = [str(entry.get("visual_medium") or entry.get("visual_tier_used") or "UNKNOWN") for entry in feed_entries if isinstance(entry,dict)]
            if not ai_exception_allowed(modes) or any(str(entry.get("date", ""))[:10]==str(date_iso)[:10] for entry in feed_entries if isinstance(entry,dict)):
                raise ValueError("Non-photo cadence does not permit this requested edition")
        if genai_client is not None and requested_medium != "blueprint":
            from .photo_recovery import recover_visual, revision_owner_constraints
            def revise_for_photo(current, *, change_word):
                selected = [cluster for cluster in scored_clusters if getattr(cluster, "cluster_id", None)==current.selected_cluster_id]
                if not selected:
                    raise ValueError("Original story unavailable; cannot revise unrelated signals")
                locks = revision_owner_constraints(owner, current, change_word=change_word)
                revised = commission_daily_zeitgeist(selected,
                    history_words_30d=history_words_30d+([current.word] if change_word and not locks.get("override_word") else []),
                    genai_client=genai_client, model_name=str(config.get("budget_guardrails",{}).get("primary_text_model","gemini-3.8-flash")),
                    max_retries=0, owner_override=locks)
                if revised.fallback_used or revised.selected_cluster_id!=current.selected_cluster_id:
                    raise ValueError("Same-story photo recovery did not produce a grounded commission")
                return revised
            cadence_history = [str(entry.get("visual_medium") or entry.get("visual_tier_used") or "UNKNOWN") for entry in feed_entries[:30] if isinstance(entry,dict)]
            commission, published, recovery_trace = recover_visual(commission,
                render=render_commission, revise=revise_for_photo, history=cadence_history,
                photo_required=requested_medium=="photo" or any(str(entry.get("date", ""))[:10]==str(date_iso)[:10] for entry in feed_entries if isinstance(entry,dict)))
            self.last_commission = commission
            self.last_photo_recovery = recovery_trace
        else:
            published = render_commission(commission)
        self.last_published_pulse = published
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
            model_id=str(guardrails.get("primary_text_model", "gemini-3.8-flash")),
            estimated_input_tokens=est_in_tokens,
            estimated_output_tokens=est_out_tokens,
            estimated_monthly_cost_index=monthly_cost_index,
            monthly_budget_ceiling_index=ceiling,
            within_budget=False,
            spending_status="unknown_pending_billing_reconciliation",
            collection_health=collection_health,
        )

        self._persist_state(analysis, decision, telemetry, commission=commission, published=published)
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
        return asdict(self)


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


def _coerce_director_commission(raw: Any) -> DirectorCommission:
    if isinstance(raw, DirectorCommission):
        return raw
    if hasattr(raw, "text") and isinstance(raw.text, str):
        raw = raw.text
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [
                ln
                for ln in cleaned.splitlines()
                if not ln.strip().startswith("```")
            ]
            cleaned = "\n".join(lines).strip()
        data = json.loads(cleaned)
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
) -> str:
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

    return (
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
) -> DirectorCommission:
    """Run Editorial Director Agent with Latin-Script & 30-Day Deduplication Gate (up to 2 retries at T=0.3)."""
    owner = validate_owner_interjection(owner_override or {})
    constrained = owner.get("enabled") and any(owner.get(k) for k in ("override_word", "override_phrase", "personal_note"))
    history_list = [str(w).strip() for w in (history_words_30d or ()) if str(w).strip()]
    total_attempts = max(1, int(max_retries) + 1)
    last_rejection_reason: Optional[str] = None
    gloss_rejected = False

    if llm_callable is not None or genai_client is not None:
        for attempt_idx in range(total_attempts):
            prompt = _build_director_prompt(
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
            try:
                if llm_callable is not None:
                    sig = inspect.signature(llm_callable)
                    if len(sig.parameters) >= 2:
                        raw_resp = llm_callable(prompt, temperature)
                    else:
                        raw_resp = llm_callable(prompt)
                else:
                    try:
                        raw_resp = genai_client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=({"response_mime_type": "application/json"} if model_name.startswith("gemini-3") else
                                    {"temperature": temperature, "response_mime_type": "application/json"}),
                        )
                    except Exception as model_exc:
                        err_str = str(model_exc)
                        if "404" in err_str or "NOT_FOUND" in err_str:
                            raw_resp = genai_client.models.generate_content(
                                model="gemini-3-flash-preview",
                                contents=prompt,
                                config={"response_mime_type": "application/json"},
                            )
                        else:
                            raise
                commission = _coerce_director_commission(raw_resp)
            except SpendBlocked:
                raise
            except Exception as exc:
                last_rejection_reason = f"Malformed director response: {exc}"
                if llm_callable is None and any(code in str(exc) for code in ("402", "403", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED")):
                    break
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

            if is_valid:
                commission.attempts_used = attempt_idx + 1
                commission.fallback_used = False
                return commission

            last_rejection_reason = reason
            gloss_rejected = gloss_rejected or reason.startswith("Gloss must")

    if gloss_rejected:
        raise ValueError("Director gloss failed bounded validation retries; retain last publication")
    if constrained and (llm_callable is not None or genai_client is not None):
        raise ValueError("Owner-constrained commissioning failed validation; retain last publication")
    fallback = _deterministic_fallback_commission(
        top_clusters=top_clusters[:5], history_words_30d=history_list, attempts_used=total_attempts,
    )
    if constrained:
        # Demonstration only: live orchestration refuses this ungrounded fallback.
        word = owner.get("override_word") or fallback.word
        gloss = owner.get("override_phrase") or fallback.gloss
        anchor = f"Architectural material study of {word.lower()}: {gloss}"
        fallback = replace(fallback, word=word, gloss=gloss, commissioned_thesis=owner.get("personal_note") or gloss,
            visual_anchor=anchor, unsplash_queries=[f"{word.lower()} material study", f"{word.lower()} quiet architecture", f"{word.lower()} natural light"],
            imagen_commission_prompt=anchor + ", no text overlay", flux_3_commission_prompts=[], fallback_used=True)
    return fallback


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
