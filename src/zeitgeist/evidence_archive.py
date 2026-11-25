"""The private evidence archive: everything one published edition loaded, weighed and chose, kept forever.

One night is a meta document `<date>:<run_id>` and chunk documents `<date>:<run_id>:<part>:<n>` in the
`evidence_archive` collection, each under MAX_DOC_BYTES of encoded JSON (Firestore refuses a document over
1 MiB). The same run id always gives the same document ids, so a second write overwrites the first; the meta
document is written last and lists the chunks of its night. Nothing in the nightly run reads the archive.

Image bytes and data URIs are left out: only the links and text the rows already carry are kept. The one
outside effect is the store write, held to BUDGET_SECONDS; the one output is a public line of numbers.

With ZEITGEIST_FAILED_NIGHT_ARCHIVE on as well, a night that fails is archived the same way, with only the parts
it reached; its meta document adds the observed status, the stage it failed in and the exception's class.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import os
import threading
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .own_group import run_with_budget
from .run_record import _without_surrogates

ARCHIVE_SCHEMA = 1
SWITCH_ENV = "ZEITGEIST_EVIDENCE_ARCHIVE"
COLLECTION = "evidence_archive"
MAX_DOC_BYTES = 800_000
BUDGET_SECONDS = 120.0
# Room left in each document for its own keys, so the entries alone are measured against the rest.
_HEADROOM_BYTES = 4_096
# A JSON string grows at most sixfold when written (a control character becomes a six-character escape).
_FRAGMENT_CHARS = (MAX_DOC_BYTES - 2 * _HEADROOM_BYTES) // 6
_DROPPED = object()


def archive_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH_ENV) == "1"


FAILED_NIGHT_SWITCH_ENV = "ZEITGEIST_FAILED_NIGHT_ARCHIVE"
RUN_STARTED_ENV = "ZEITGEIST_RUN_STARTED_AT"
# The edition job stops at 15 minutes and its outcome steps run after the edition. The run's recorded start comes
# after checkout and install: on the 25 Nov edition the job started 05:24:29Z and the pipeline step 05:24:41Z
# (12 s of setup), and the steps after the pipeline took about 7 s. Twelve minutes from the recorded start leaves
# about three minutes for those steps.
FAILED_NIGHT_DEADLINE_SECONDS = 12 * 60
UNKNOWN_START_SECONDS = 60.0


def failed_night_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """A night that fails is archived only with this switch and the evidence archive's both on."""
    env = os.environ if environ is None else environ
    return archive_enabled(env) and env.get(FAILED_NIGHT_SWITCH_ENV) == "1"


def failed_night_budget(environ: Optional[Mapping[str, str]] = None, *, now: Optional[datetime] = None) -> float:
    """BUDGET_SECONDS, capped to the time left before the deadline counted from the run's start; never below zero.

    A start that is missing, unreadable or without a time zone gives UNKNOWN_START_SECONDS.
    """
    env = os.environ if environ is None else environ
    raw = str(env.get(RUN_STARTED_ENV) or "")
    try:
        started = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
    except ValueError:
        started = None
    if started is None or started.tzinfo is None:
        return min(BUDGET_SECONDS, UNKNOWN_START_SECONDS)
    elapsed = ((now or datetime.now(timezone.utc)) - started).total_seconds()
    return max(0.0, min(BUDGET_SECONDS, FAILED_NIGHT_DEADLINE_SECONDS - elapsed))


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _fields(value: Any) -> Dict[str, Any]:
    """An object's own fields: a dataclass's fields, or its public attributes."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    return {key: item for key, item in vars(value).items() if not str(key).startswith("_")}


def _plain(value: Any, key: str = "") -> Any:
    """Plain JSON values; image bytes and data URIs become _DROPPED, embedding bytes base64 text."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _DROPPED if value.startswith("data:image/") else value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii") if key.startswith("embedding") else _DROPPED
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), key)
    if isinstance(value, np.generic):
        return _plain(value.item(), key)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        out = {}
        for name, item in value.items():
            name = str(name)
            # The float32 vector is decoded from the stored float16 one; the stores never keep both.
            if name == "embedding_f32" and value.get("embedding_f16") is not None:
                continue
            plain = _plain(item, name)
            if plain is not _DROPPED:
                out[name] = plain
        return out
    if isinstance(value, (list, tuple)):
        return [plain for plain in (_plain(item, key) for item in value) if plain is not _DROPPED]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item, key) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "__dict__") or dataclasses.is_dataclass(value):
        return _plain(_fields(value), key)
    return str(value)


def plain_entries(entries: Sequence[Any]) -> List[Any]:
    """Each entry as plain JSON, every string free of lone surrogates (Firestore refuses them)."""
    out = []
    for entry in entries or ():
        plain = _plain(entry)
        if plain is not _DROPPED:
            out.append(_without_surrogates(plain))
    return out


def group_entries(clusters: Sequence[Any]) -> List[Dict[str, Any]]:
    """Every scored group, ranked, with all its own fields and its items by url_hash."""
    groups = []
    for rank, cluster in enumerate(clusters or (), 1):
        fields = dict(cluster) if isinstance(cluster, Mapping) else _fields(cluster)
        items = list(fields.pop("items", None) or ())
        hashes = [str((item.get("url_hash") if isinstance(item, Mapping) else getattr(item, "url_hash", "")) or "")
                  for item in items]
        groups.append({"rank": rank, **fields, "item_count": len(items), "item_hashes": hashes})
    return groups


def _fragments(entry: Any, index: int) -> List[Dict[str, Any]]:
    text = json.dumps(entry, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    pieces = [text[at:at + _FRAGMENT_CHARS] for at in range(0, len(text), _FRAGMENT_CHARS)]
    return [{"_fragment": {"entry": index, "k": k, "of": len(pieces), "json": piece}} for k, piece in enumerate(pieces)]


def _chunks(entries: List[Any], budget: int) -> List[List[Any]]:
    """The entries packed in order into lists whose encoded size stays within budget; a larger entry is fragmented."""
    chunks: List[List[Any]] = []
    current: List[Any] = []
    size = 0
    for index, entry in enumerate(entries):
        entry_size = _encoded_size(entry)
        pieces = _fragments(entry, index) if entry_size + 1 > budget else [entry]
        for piece in pieces:
            piece_size = entry_size if piece is entry else _encoded_size(piece)
            if current and size + piece_size + 1 > budget:
                chunks.append(current)
                current, size = [], 0
            current.append(piece)
            size += piece_size + 1
    if current:
        chunks.append(current)
    return chunks


def archive_documents(
    *, date_iso: str, run_id: str, parts: Mapping[str, Sequence[Any]],
    extra: Optional[Mapping[str, Any]] = None, now_utc: Optional[datetime] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """The night's documents, chunks first and the meta document last; each under MAX_DOC_BYTES.

    An entry too large for one document is kept as fragments of its JSON text, which `assemble` joins.
    """
    day = str(date_iso)[:10]
    meta_id = f"{day}:{run_id}"
    documents: List[Tuple[str, Dict[str, Any]]] = []
    summary: Dict[str, Dict[str, int]] = {}
    for part, raw_entries in parts.items():
        part = str(part)
        entries = plain_entries(raw_entries)
        header = {"archive_schema": ARCHIVE_SCHEMA, "date": day, "run_id": run_id, "part": part, "n": 10**6, "entries": []}
        chunks = _chunks(entries, MAX_DOC_BYTES - _HEADROOM_BYTES - _encoded_size(header))
        for n, chunk in enumerate(chunks):
            documents.append((f"{meta_id}:{part}:{n}", {**header, "n": n, "entries": chunk}))
        summary[part] = {"entries": len(entries), "chunks": len(chunks)}
    moment = now_utc or datetime.now(timezone.utc)
    meta = {
        "archive_schema": ARCHIVE_SCHEMA, "date": day, "run_id": run_id,
        "written_at": moment.astimezone(timezone.utc).isoformat(),
        **_without_surrogates(_plain(dict(extra or {}))),
        "parts": summary, "chunk_ids": [doc_id for doc_id, _ in documents],
    }
    documents.append((meta_id, meta))
    if any(_encoded_size(document) >= MAX_DOC_BYTES for _, document in documents):
        raise ValueError("archive document over the size limit")
    return documents


def assemble(documents: Mapping[str, Mapping[str, Any]], meta_id: str) -> Dict[str, List[Any]]:
    """The night's parts read back from its documents, fragments joined: for offline tools and tests."""
    meta = documents[meta_id]
    parts: Dict[str, List[Any]] = {name: [] for name in meta["parts"]}
    pending: Dict[Tuple[str, int], Dict[int, str]] = {}
    for chunk_id in meta["chunk_ids"]:
        chunk = documents[chunk_id]
        part = chunk["part"]
        for entry in chunk["entries"]:
            fragment = entry.get("_fragment") if isinstance(entry, dict) and len(entry) == 1 else None
            if fragment is None:
                parts[part].append(entry)
                continue
            pieces = pending.setdefault((part, fragment["entry"]), {})
            pieces[fragment["k"]] = fragment["json"]
            if len(pieces) == fragment["of"]:
                parts[part].append(json.loads("".join(pieces[k] for k in range(fragment["of"]))))
                del pending[(part, fragment["entry"])]
    return parts


def write_archive(
    store: Any, documents: Sequence[Tuple[str, Dict[str, Any]]], *, closed: Optional[threading.Event] = None,
) -> Tuple[int, int]:
    """Save each document in order; (documents, encoded bytes). Raises on any failure; prints nothing.

    Once `closed` is set no further document is saved.
    """
    count = total = 0
    for doc_id, document in documents:
        if closed is not None and closed.is_set():
            raise TimeoutError("evidence archive budget spent")
        store.save_evidence_archive_doc(doc_id, document)
        count += 1
        total += _encoded_size(document)
    return count, total


def archive_with_budget(
    store: Any, build: Callable[[], Sequence[Tuple[str, Dict[str, Any]]]], seconds: Optional[float] = None,
) -> str:
    """Build and write the night's documents within the budget; the public line, numbers only. Never raises.

    The work runs on its own thread, so whatever it raises (a store error, SpendBlocked) stays there and is
    named only by its class. Past the budget the thread is abandoned and saves nothing more.
    """
    closed = threading.Event()
    outcome: Dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["written"] = write_archive(store, build(), closed=closed)
        except BaseException as exc:
            outcome["error"] = type(exc).__name__

    finished = run_with_budget(work, BUDGET_SECONDS if seconds is None else seconds, name="evidence-archive")
    if not finished:
        closed.set()
        return "evidence_archive=failed error_class=TimeoutError"
    if "written" in outcome:
        docs, size = outcome["written"]
        return f"evidence_archive=ok docs={int(docs)} bytes={int(size)}"
    return f"evidence_archive=failed error_class={outcome.get('error', 'Exception')}"
