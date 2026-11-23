"""The reflector: a group that keeps standing in the window without leading, and whether it has been named.

This part finds the standing group while the night's groups are in memory and stores what the
reflector needs from the run on its record. Arithmetic only: it calls no provider and decides nothing.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .methodology import _extract_unit_embedding_f32, _get_item_field

SWITCH_ENV = "ZEITGEIST_REFLECTOR"
INPUTS_KEY = "reflector_inputs"
MIN_DAYS = 5


def reflector_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


def _utc_date(value: Any) -> Optional[date]:
    """The UTC date of a datetime or an ISO string; a time without a zone is read as UTC. None when unreadable."""
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    return None


def _items(cluster: Any) -> List[Any]:
    return list(getattr(cluster, "items", None) or ())


def _distinct_days(cluster: Any) -> int:
    days = {_utc_date(_get_item_field(item, "published_at")) for item in _items(cluster)}
    days.discard(None)
    return len(days)


def _score(cluster: Any) -> float:
    try:
        score = float(getattr(cluster, "commission_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def standing_groups(clusters: Sequence[Any], *, min_days: int = MIN_DAYS) -> list:
    """Every cluster after the first whose items fall on at least `min_days` distinct UTC dates.

    Most distinct days first, then the higher `commission_score`, then the smaller `cluster_id`.
    Items without a readable `published_at` are ignored. Pure.
    """
    counted = [(cluster, _distinct_days(cluster)) for cluster in list(clusters or ())[1:]]
    qualifying = [(cluster, days) for cluster, days in counted if days >= int(min_days)]
    qualifying.sort(key=lambda pair: (-pair[1], -_score(pair[0]), str(getattr(pair[0], "cluster_id", ""))))
    return [cluster for cluster, _ in qualifying]


def centre(cluster: Any) -> Optional[List[float]]:
    """The unit mean of the items' unit vectors, read as understanding._vectors reads them; None without any."""
    vectors = []
    for item in _items(cluster):
        try:
            vector = np.asarray(_extract_unit_embedding_f32(item), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if vector.size and np.all(np.isfinite(vector)) and float(np.linalg.norm(vector)) > 1e-12:
            vectors.append(vector)
    if not vectors:
        return None
    dims = [vector.shape[0] for vector in vectors]
    dim = max(set(dims), key=dims.count)
    mean = np.mean([vector for vector in vectors if vector.shape[0] == dim], axis=0)
    norm = float(np.linalg.norm(mean))
    if not norm > 1e-12:
        return None
    return [float(value) for value in mean / norm]


def reflector_inputs(clusters: Sequence[Any], chosen_cluster_id: Any, embedding_model: Any) -> Dict[str, Any]:
    """The standing group and the chosen candidate as the reflector needs them: block, centre, item hashes.

    `standing` is the first standing group that is not the chosen one; `chosen` is the cluster among the
    first five whose id is `chosen_cluster_id`. Either is None when there is none. Pure.
    """
    from . import run_record
    from .understanding import _render_cluster

    ranked = list(clusters or ())
    chosen_id = None if chosen_cluster_id is None else str(chosen_cluster_id)

    def cluster_id(cluster: Any) -> str:
        return str(getattr(cluster, "cluster_id", ""))

    standing = next((cluster for cluster in standing_groups(ranked) if cluster_id(cluster) != chosen_id), None)
    chosen = next((cluster for cluster in ranked[:run_record.MAX_GROUPS] if cluster_id(cluster) == chosen_id), None)
    standing_entry = None
    if standing is not None:
        standing_entry = {
            "cluster_id": cluster_id(standing),
            "rank": next(rank for rank, cluster in enumerate(ranked, 1) if cluster is standing),
            "days": _distinct_days(standing),
            "item_hashes": [str(value) for value in (_get_item_field(item, "url_hash") for item in _items(standing))
                            if value],
            "block": _render_cluster(standing),
            "centre": centre(standing),
        }
    chosen_entry = None
    if chosen is not None:
        chosen_entry = {"cluster_id": cluster_id(chosen), "block": _render_cluster(chosen), "centre": centre(chosen)}
    return {"standing": standing_entry, "chosen": chosen_entry, "embedding_model": str(embedding_model or "")}


def store_reflector_inputs(
    store: Any, run_id: str, clusters: Sequence[Any], chosen_cluster_id: Any, embedding_model: Any,
) -> bool:
    """Store `reflector_inputs` under tests["reflector_inputs"] of a stored record; True when it was stored.

    Without a stored record nothing is written. Never raises but KeyboardInterrupt and SystemExit;
    prints and logs nothing.
    """
    from . import run_record

    try:
        inputs = reflector_inputs(clusters, chosen_cluster_id, embedding_model)
        return bool(run_record.add_test_numbers(store, run_id, INPUTS_KEY, inputs))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False


# ---- has it been named? arithmetic, then the pointing task ----

ANY_DAY_GLOSSES = (
    "A quiet tension beneath ordinary days.",
    "Change arriving slower than expected.",
    "What we carry and what we set down.",
)
# The standing group counts as named when its closest past gloss is this close to tonight's reference, or closer.
NAMED_TOLERANCE = 0.02
NONE_ANSWER = "none"


def _unit(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(array)) if array.size else 0.0
    if not array.size or not np.all(np.isfinite(array)) or not norm > 1e-12:
        raise ValueError("vector is not usable")
    return array / norm


def named_by_arithmetic(
    standing_centre: Any, gloss_vectors: Sequence[Any], own_gloss_vector: Any, chosen_centre: Any,
) -> Dict[str, Any]:
    """Whether a past gloss sits as close to the standing group as tonight's gloss sits to tonight's candidate.

    The reference is the cosine of `own_gloss_vector` to `chosen_centre`; the best is the highest cosine of
    a past gloss to `standing_centre`. `margin` is best minus reference; `named` is best >= reference - 0.02.
    Raises ValueError when there is no past gloss or a vector is unusable. Pure.
    """
    standing = _unit(standing_centre)
    reference_vectors = [_unit(own_gloss_vector), _unit(chosen_centre)]
    glosses = [_unit(vector) for vector in gloss_vectors or ()]
    if not glosses:
        raise ValueError("no past gloss")
    if any(vector.shape != standing.shape for vector in glosses + reference_vectors):
        raise ValueError("vector dimensions differ")
    reference = float(reference_vectors[0] @ reference_vectors[1])
    cosines = [float(vector @ standing) for vector in glosses]
    best_index = int(np.argmax(cosines))
    best = cosines[best_index]
    return {"named": bool(best >= reference - NAMED_TOLERANCE), "margin": float(best - reference),
            "best_index": best_index}


PROMPT_HEAD = (
    "Below is one group of items and a numbered list of short glosses. Reply with JSON of the form "
    '{"answer": N}, where N is the number of the gloss that describes what this group is about, or "none" if '
    "no gloss describes it."
)


def build_question(block: str, glosses: Sequence[str], order: Sequence[int]) -> str:
    """The head, the block, then the glosses numbered from 1: gloss `order[n - 1]` is shown as number n."""
    order = [int(index) for index in order]
    if sorted(order) != list(range(len(glosses))):
        raise ValueError("order is not a permutation of the glosses")
    lines = ["%d. %s" % (number, " ".join(str(glosses[index]).split())) for number, index in enumerate(order, 1)]
    return PROMPT_HEAD + "\n\n" + str(block) + "\n\n" + "\n".join(lines)


def parse_answer(text: Any, count: int) -> Any:
    """An integer from 1 to `count`, the string "none", or None for anything else. Never raises.

    The whole reply must be one JSON object with an "answer" key; text around it is not read.
    """
    try:
        if not isinstance(text, str):
            return None
        reply = json.loads(text.strip())
        if not isinstance(reply, dict) or "answer" not in reply:
            return None
        answer = reply["answer"]
        if isinstance(answer, str):
            return NONE_ANSWER if answer.strip().lower() == NONE_ANSWER else None
        if isinstance(answer, int) and not isinstance(answer, bool) and 1 <= answer <= int(count):
            return int(answer)
    except Exception:
        return None
    return None


def answer_identity(answer: Any, order: Sequence[int]) -> Any:
    """The gloss an answer pointed at, as its index before the shuffle; "none" and None stay as they are."""
    if isinstance(answer, int) and not isinstance(answer, bool) and 1 <= answer <= len(order):
        return int(order[answer - 1])
    return answer if answer == NONE_ANSWER else None


def verdict(
    control_answers: Sequence[Any], control_expected: Sequence[Any], real_answers: Sequence[Any],
    plant_indexes: Sequence[Any],
) -> Dict[str, Any]:
    """The pointing task's verdict over its runs. Every answer and index is a gloss identity (`answer_identity`).

    `plants_ok`: every control answer is its expected gloss and no real answer is its run's any-day gloss.
    `agreement`: every real answer was read and all are the same. `status` is "void" unless both hold,
    then "pointer" when the answers are all "none" and "named" when they are all the same gloss. Pure.
    """
    control_answers, control_expected = list(control_answers), list(control_expected)
    real_answers, plant_indexes = list(real_answers), list(plant_indexes)
    runs = len(real_answers)
    complete = runs > 0 and len(control_answers) == len(control_expected) == len(plant_indexes) == runs
    plants_ok = bool(complete
                     and all(answer is not None and answer == expected
                             for answer, expected in zip(control_answers, control_expected))
                     and all(answer is None or answer != plant for answer, plant in zip(real_answers, plant_indexes)))
    agreement = bool(runs > 0 and all(answer is not None for answer in real_answers)
                     and all(answer == real_answers[0] for answer in real_answers))
    status = "void"
    if plants_ok and agreement:
        status = "pointer" if real_answers[0] == NONE_ANSWER else "named"
    return {"plants_ok": plants_ok, "agreement": agreement, "status": status}


# ---- the next night: the pointed group is offered to the Director ----

CANDIDATE_ENV = "ZEITGEIST_REFLECTOR_CANDIDATE"
# The edition's lookup runs before the Director call, so it is held to this budget.
CANDIDATE_BUDGET_SECONDS = 20.0
POINTER_KINDS = ("daily_run", "private_run")
POINTER_MAX_AGE_HOURS = 36
POINTER_LOAD_LIMIT = 200
MIN_OVERLAP = 0.5
# The setting's result on a record; a trial's (tests["reflector_trial"]) is never read.
KEY = "reflector"


def candidate_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """The switch, and the keyword repair: without it a word about the extra group would be published with
    the top group's keywords, so the candidate is never offered."""
    from .understanding import repair_enabled

    env = os.environ if environ is None else environ
    return env.get(CANDIDATE_ENV) == "1" and repair_enabled(env)


def _pointer_hashes(result: Any) -> Optional[List[str]]:
    hashes = result.get("pointer_item_hashes") if isinstance(result, Mapping) else None
    if not isinstance(hashes, list) or not hashes:
        return None
    if not all(isinstance(value, str) and value for value in hashes):
        return None
    return list(dict.fromkeys(hashes))


def newest_pointer(store: Any, now: datetime, *, max_age_hours: float = POINTER_MAX_AGE_HOURS) -> Optional[Dict[str, Any]]:
    """The newest pointer of a daily or private run record written in the window; None when there is none.

    Only tests["reflector"] with status "pointer" counts, never tests["reflector_trial"]. The window is
    read from the record's own written_at. Never raises; prints and logs nothing.
    """
    from . import run_record

    try:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        window = timedelta(hours=float(max_age_hours))
        found = []
        for kind in POINTER_KINDS:
            try:
                records = store.load_run_records(kind=kind, limit=POINTER_LOAD_LIMIT)
            except Exception:
                continue
            for record in records or ():
                if not isinstance(record, dict) or record.get("kind") != kind:
                    continue
                tests, run_id = record.get("tests"), record.get("run_id")
                if not isinstance(tests, dict) or not isinstance(run_id, str):
                    continue
                result = tests.get(KEY)
                if not isinstance(result, Mapping) or result.get("status") != "pointer":
                    continue
                hashes = _pointer_hashes(result)
                moment = run_record._written_at(record)
                if hashes is None or moment is None or not timedelta(0) <= now - moment <= window:
                    continue
                found.append((moment, run_id, hashes))
        if not found:
            return None
        found.sort(key=lambda entry: entry[1])
        found.sort(key=lambda entry: entry[0], reverse=True)
        return {"run_id": found[0][1], "item_hashes": found[0][2]}
    except Exception:
        return None


def match_pointer(pointer: Mapping[str, Any], clusters: Sequence[Any]) -> tuple:
    """The ranked cluster holding the largest share of the pointer's items, and that share.

    A match needs at least half of them; ties go to the higher rank. Without a match the cluster is None.
    """
    hashes = set(pointer.get("item_hashes") or ())
    best, best_share = None, 0.0
    if not hashes:
        return None, 0.0
    for cluster in list(clusters or ()):
        present = {str(_get_item_field(item, "url_hash") or "") for item in _items(cluster)}
        share = len(hashes & present) / len(hashes)
        if share > best_share:
            best, best_share = cluster, share
    return (best if best_share >= MIN_OVERLAP else None), float(best_share)


def _no_candidate() -> Dict[str, Any]:
    return {"pointer_run_id": None, "matched_cluster_id": None, "overlap": None, "already_shown": False,
            "offered": False, "group": None}


def candidate_for_night(store: Any, clusters: Sequence[Any], now: datetime) -> Dict[str, Any]:
    """The newest pointer matched to tonight's ranked groups. Never raises; prints and logs nothing.

    `already_shown` is true when the match is one of the first five groups; then nothing extra is offered.
    Otherwise a match is offered, and `group` is that cluster.
    """
    from . import run_record

    try:
        pointer = newest_pointer(store, now)
        if pointer is None:
            return _no_candidate()
        ranked = list(clusters or ())
        group, overlap = match_pointer(pointer, ranked)
        found = dict(_no_candidate(), pointer_run_id=pointer["run_id"], overlap=overlap)
        if group is None:
            return found
        rank = next(index for index, cluster in enumerate(ranked) if cluster is group)
        shown = rank < run_record.MAX_GROUPS
        found.update(matched_cluster_id=str(getattr(group, "cluster_id", "")), already_shown=shown,
                     offered=not shown, group=None if shown else group)
        return found
    except Exception:
        return _no_candidate()


def candidate_record(found: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the run record keeps of the night's candidate: everything but the group itself."""
    if found is None:
        return None
    return {key: value for key, value in found.items() if key != "group"}
