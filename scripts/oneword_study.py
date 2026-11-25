"""One-word study (25 Nov, private, study branch only; delete after).

Replays the Director's archived request for the 25 Nov edition through the guarded Gemini client with one change:
"a single 1-2 word" becomes "a single one-word". Three tries. Replies go to the private store collection
study_runs as 2026-11-25:oneword:<n>; stdout carries numbers only (public log).
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state_store import get_state_store  # noqa: E402
from src.zeitgeist import evidence_archive as ea  # noqa: E402
from src.zeitgeist.adk_pipeline import _director_config  # noqa: E402

OLD = "a single 1-2 word"
NEW = "a single one-word"
MODEL = "gemini-4-neon"


def main() -> int:
    store = get_state_store(repo_root=os.getcwd())
    db = store._get_client()
    col = db.collection(store._col("evidence_archive"))
    docs = {d.id: d.to_dict() for d in col.stream() if d.id.startswith("2026-11-25:daily_run")}
    metas = sorted(k for k in docs if k.count(":") == 4)
    if len(metas) != 1:
        print(json.dumps({"status": "error", "metas": len(metas)}))
        return 1
    night = ea.assemble(docs, metas[0])
    attempts = night.get("director_attempts") or []
    accepted = [a for a in attempts if a.get("accepted")] or attempts
    request = str(accepted[-1].get("request_text") or "")
    if request.count(OLD) != 1:
        print(json.dumps({"status": "error", "phrase_count": request.count(OLD)}))
        return 1
    prompt = request.replace(OLD, NEW)
    from google import genai
    from src.zeitgeist.spend_guard import guard_genai_client

    key = os.environ["GEMINI_API_KEY"].strip()
    client = guard_genai_client(genai.Client(api_key=key), credential_key=key)
    out = db.collection(store._col("study_runs"))
    ok = 0
    for n in (1, 2, 3):
        doc = {"night": "2026-11-25", "kind": "oneword", "try": n, "model": MODEL,
               "written_at": datetime.now(timezone.utc).isoformat()}
        try:
            reply = client.models.generate_content(model=MODEL, contents=prompt, config=_director_config(MODEL, 0.3))
            doc["reply_text"] = reply.text if isinstance(getattr(reply, "text", None), str) else str(reply)
            ok += 1
        except Exception as exc:  # the class only, never the message, in the public log
            doc["error_class"] = type(exc).__name__
        out.document(f"2026-11-25:oneword:{n}").set(doc)
        print(json.dumps({"try": n, "ok": "reply_text" in doc, "error_class": doc.get("error_class")}))
    print(json.dumps({"status": "done", "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
