"""scripts/reflect.py: the reflector on a stored run. Synthetic records, fake clients, no network, no paid call."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import reflect
from src.state_store import JsonFileStateStore
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import model_choice as mc
from src.zeitgeist import reflector as rf
from src.zeitgeist import run_record
from src.zeitgeist.spend_guard import SpendBlocked

NOW = datetime(2026, 11, 22, 4, 0, tzinfo=timezone.utc)
WRITTEN = NOW - timedelta(hours=2)
DATE = "2026-11-21"
OWN_GLOSS = "Zqown gloss of tonight."
CHOSEN_BLOCK = "- Cluster c_chosen: Zqchosen block"
STANDING_BLOCK = "- Cluster c_standing: Zqstanding block"
STANDING_HASHES = ["h_s1", "h_s2", "h_s3"]
PAST = [("2026-11-%02d" % day, "Zqpast gloss %d." % day) for day in range(20, 9, -1)]
# Every synthetic private string: the glosses and the blocks.
PRIVATE = ("Zqown", "Zqchosen", "Zqstanding", "Zqpast", "Zqsecret")
LINE = re.compile(r"\[reflector\] status=\d+ plants_ok=[01] agreement=[01] arithmetic_named=[01] calls=\d+")
EMBED = mc.EMBED_2


def _axis(index):
    vector = np.zeros(768)
    vector[index] = 1.0
    return [float(value) for value in vector]


def _inputs(standing=True, chosen=True, model=EMBED):
    return {
        "standing": {"cluster_id": "c_standing", "rank": 4, "days": 6, "item_hashes": list(STANDING_HASHES),
                     "block": STANDING_BLOCK, "centre": _axis(1)} if standing else None,
        "chosen": {"cluster_id": "c_chosen", "block": CHOSEN_BLOCK, "centre": _axis(0)} if chosen else None,
        "embedding_model": model,
    }


def _record(kind="daily_run", *, written=WRITTEN, director=mc.HYDROGEN, gloss=OWN_GLOSS, inputs=None, arm=None,
            repeat=None, date=DATE):
    tests = {"own_group": {"status": "ok"}}
    if inputs is not False:
        tests["reflector_inputs"] = _inputs() if inputs is None else inputs
    return {"record_schema": 1, "kind": kind, "date": date, "written_at": written.isoformat(),
            "models": {"director": director, "judge": director, "embedding": EMBED, "fallback_fired": False,
                       "unrecognized": False},
            "word": "Zqword", "gloss": gloss, "arm": arm, "repeat": repeat, "tests": tests,
            "prompt_sha256": "a" * 64, "evidence_text": CHOSEN_BLOCK, "candidate_words": []}


class Store(JsonFileStateStore):
    """The JSON store with a feed history; every read and write is noted."""

    def __init__(self, root, history=None, fail=None):
        super().__init__(repo_root=root)
        self.history = [{"date": day, "gloss": gloss, "word": "w"} for day, gloss in PAST] if history is None else history
        self.fail = fail
        self.calls = []

    def put(self, record, run_id=None):
        run_id = run_id or run_record.make_run_id(record["kind"], record["date"], {}, record.get("arm"),
                                                  record.get("repeat"))
        assert run_record.write_run_record(self, run_id, record)
        self.calls.clear()
        return run_id

    def load_run_records(self, **kwargs):
        self.calls.append("load_run_records")
        if self.fail is not None:
            raise self.fail
        return super().load_run_records(**kwargs)

    def save_run_record(self, run_id, document):
        self.calls.append("save_run_record")
        super().save_run_record(run_id, document)

    def load_pulse_feed_history(self, days=30):
        self.calls.append(("load_pulse_feed_history", days))
        return list(self.history)


class NoHistoryStore(Store):
    def __getattribute__(self, name):
        if name == "load_pulse_feed_history":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _lines(question):
    """The numbered glosses of a question, by number."""
    numbered = {}
    for line in question.split("\n\n", 2)[2].split("\n"):
        number, text = line.split(". ", 1)
        numbered[text] = int(number)
    return numbered


class Client:
    """A fake genai client: control questions are answered by `control`, real ones by `real`."""

    def __init__(self, control="own", real="none", embed_error=None, generate_error=None, vectors=None):
        self.control, self.real = control, real
        self.embed_error, self.generate_error = embed_error, generate_error
        # By default tonight's gloss sits on its candidate's axis and no past gloss near the standing group,
        # so the arithmetic says "not named" unless a test says otherwise.
        self.vectors = {OWN_GLOSS: _axis(0)} if vectors is None else vectors
        self.generates, self.embeds = [], []
        self.models = self

    def embed_content(self, *, model, contents, config):
        texts = [content.parts[0].text if hasattr(content, "parts") else str(content) for content in contents]
        self.embeds.append((model, texts, config))
        if self.embed_error is not None:
            raise self.embed_error
        values = []
        for text in texts:
            match = next((vector for key, vector in self.vectors.items() if key in text), None)
            values.append(match if match is not None else _axis(700))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector) for vector in values])

    def generate_content(self, *, model, contents, config):
        self.generates.append((model, contents, config))
        run = (len(self.generates) + 1) // 2
        if self.generate_error is not None:
            error = self.generate_error(len(self.generates)) if callable(self.generate_error) else self.generate_error
            if error is not None:
                raise error
        numbered = _lines(contents)
        is_control = CHOSEN_BLOCK in contents
        policy = self.control if is_control else self.real
        if callable(policy):
            policy = policy(run, numbered)
        if policy == "own":
            answer = numbered[OWN_GLOSS]
        elif policy == "plant":
            answer = numbered[rf.ANY_DAY_GLOSSES[run - 1]]
        elif isinstance(policy, str) and policy.startswith("gloss:"):
            answer = numbered[policy[len("gloss:"):]]
        elif policy == "garbage":
            return SimpleNamespace(text="I think it is number 2")
        else:
            answer = policy
        return SimpleNamespace(text=json.dumps({"answer": answer}))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("ZEITGEIST_REFLECTOR_MODEL", raising=False)


def _reflect(store, client, **kwargs):
    kwargs.setdefault("now", NOW)
    return reflect.reflect(store, genai_client=client, **kwargs)


def _stored(store, run_id):
    return JsonFileStateStore(repo_root=store.repo_root).load_run_record(run_id)["tests"].get("reflector")


def _no_paid_call(client):
    assert client.generates == [] and client.embeds == []


# ---- statuses that end the run before any paid call ----


def test_no_record_stores_nothing_and_calls_nothing(tmp_path):
    store, client = Store(tmp_path), Client()
    outcome = _reflect(store, client)
    assert outcome["status"] == "no_record" and outcome["calls"] == 0
    _no_paid_call(client)
    assert "save_run_record" not in store.calls


def test_a_record_without_the_inputs_is_not_used(tmp_path):
    store, client = Store(tmp_path), Client()
    store.put(_record(inputs=False))
    assert _reflect(store, client)["status"] == "no_record"
    _no_paid_call(client)
    assert "save_run_record" not in store.calls


@pytest.mark.parametrize("hours, found", [(35, True), (37, False)])
def test_only_a_record_written_in_the_last_36_hours_is_used(tmp_path, hours, found):
    store, client = Store(tmp_path), Client()
    store.put(_record(written=NOW - timedelta(hours=hours), inputs=_inputs(standing=False)))
    assert _reflect(store, client)["status"] == ("no_group" if found else "no_record")


def test_the_newest_record_of_the_kind_and_arm_is_used(tmp_path):
    store, client = Store(tmp_path), Client()
    store.put(_record(written=NOW - timedelta(hours=1), inputs=_inputs(standing=False)))
    old = store.put(_record("private_run", written=NOW - timedelta(hours=5), arm="current", repeat=1,
                            inputs=_inputs(standing=False)))
    new = store.put(_record("private_run", written=NOW - timedelta(hours=3), arm="current", repeat=2,
                            inputs=_inputs(standing=False)))
    other = store.put(_record("private_run", written=NOW - timedelta(hours=2), arm="previous", repeat=1,
                              inputs=_inputs(standing=False)))
    _reflect(store, client, kind="private_run", arm="current")
    assert _stored(store, new)["status"] == "no_group"
    assert _stored(store, old) is None and _stored(store, other) is None


def test_no_standing_group_stores_no_group(tmp_path):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record(inputs=_inputs(standing=False)))
    outcome = _reflect(store, client)
    assert outcome["status"] == "no_group" and outcome["calls"] == 0
    _no_paid_call(client)
    stored = _stored(store, run_id)
    assert stored["status"] == "no_group" and stored["pointer_item_hashes"] is None
    assert stored["model"] == mc.FLASH and stored["written_at"] == NOW.isoformat()


@pytest.mark.parametrize("record", [_record(inputs=_inputs(chosen=False)), _record(gloss=None), _record(gloss=" ")],
                         ids=["no-chosen", "no-gloss", "blank-gloss"])
def test_no_chosen_entry_or_no_gloss_stores_no_control(tmp_path, record):
    store, client = Store(tmp_path), Client()
    run_id = store.put(record)
    assert _reflect(store, client)["status"] == "no_control"
    _no_paid_call(client)
    assert _stored(store, run_id)["status"] == "no_control"


def test_the_directors_own_model_is_skipped_unless_allowed(tmp_path):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record(director=mc.FLASH))
    outcome = _reflect(store, client)
    assert (outcome["status"], outcome["reason"]) == ("skipped", "same_model")
    _no_paid_call(client)
    stored = _stored(store, run_id)
    assert (stored["status"], stored["reason"]) == ("skipped", "same_model")
    allowed = _reflect(store, client, allow_same_model=True)
    assert allowed["status"] == "pointer" and len(client.generates) == 6


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-embedding-2", "unknown"])
def test_a_model_outside_the_two_is_skipped_before_anything(tmp_path, model):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record())
    outcome = _reflect(store, client, model=model)
    assert (outcome["status"], outcome["reason"]) == ("skipped", "model")
    _no_paid_call(client)
    assert store.calls == [] and _stored(store, run_id) is None


def test_the_model_comes_from_the_environment_when_not_given(tmp_path, monkeypatch):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record(director=mc.FLASH))
    monkeypatch.setenv("ZEITGEIST_REFLECTOR_MODEL", mc.HYDROGEN)
    assert _reflect(store, client)["status"] == "pointer"
    assert {model for model, _, _ in client.generates} == {mc.HYDROGEN}
    assert _stored(store, run_id)["model"] == mc.HYDROGEN
    monkeypatch.setenv("ZEITGEIST_REFLECTOR_MODEL", "gemini-1.0")
    assert _reflect(store, Client())["reason"] == "model"


@pytest.mark.parametrize("history", [[], [{"date": DATE, "gloss": "Zqpast same night."}],
                                     [{"date": "2026-11-20", "gloss": ""}, {"date": "", "gloss": "Zqpast undated"}]],
                         ids=["empty", "own-date-only", "unusable"])
def test_no_gloss_from_another_date_stores_no_glosses(tmp_path, history):
    store, client = Store(tmp_path, history=history), Client()
    run_id = store.put(_record())
    assert _reflect(store, client)["status"] == "no_glosses"
    _no_paid_call(client)
    assert _stored(store, run_id)["status"] == "no_glosses"


def test_a_store_without_feed_history_stores_no_glosses(tmp_path):
    store, client = NoHistoryStore(tmp_path), Client()
    run_id = store.put(_record())
    assert _reflect(store, client)["status"] == "no_glosses"
    _no_paid_call(client)
    assert _stored(store, run_id)["status"] == "no_glosses"


# ---- a full run ----


def test_a_clean_pointer(tmp_path):
    store = Store(tmp_path, history=[{"date": DATE, "gloss": "Zqpast same night."}]
                  + [{"date": day, "gloss": gloss} for day, gloss in PAST])
    run_id = store.put(_record())
    before = JsonFileStateStore(repo_root=tmp_path).load_run_record(run_id)
    client = Client()
    outcome = _reflect(store, client)
    assert outcome["status"] == "pointer" and outcome["calls"] == 7
    stored = _stored(store, run_id)
    assert stored == {k: v for k, v in outcome.items() if k != "calls"}
    assert stored["plants_ok"] is True and stored["agreement"] is True
    assert stored["control_answers"] == [1, 1, 1] and stored["answers"] == ["none", "none", "none"]
    assert stored["pointer_item_hashes"] == STANDING_HASHES
    assert stored["model"] == mc.FLASH and stored["written_at"] == NOW.isoformat()
    assert set(stored) == {"status", "arithmetic_named", "arithmetic_margin", "control_answers", "answers",
                           "plants_ok", "agreement", "pointer_item_hashes", "model", "same_model", "written_at"}
    assert stored["same_model"] is False
    after = JsonFileStateStore(repo_root=tmp_path).load_run_record(run_id)
    assert {k: v for k, v in after.items() if k != "tests"} == {k: v for k, v in before.items() if k != "tests"}
    assert after["tests"]["reflector_inputs"] == before["tests"]["reflector_inputs"]
    assert after["tests"]["own_group"] == {"status": "ok"}
    assert ("load_pulse_feed_history", 30) in store.calls


def test_the_one_embedding_call_and_the_six_question_calls(tmp_path):
    store, client = Store(tmp_path), Client()
    store.put(_record())
    _reflect(store, client)
    [(model, texts, config)] = client.embeds
    assert model == EMBED and config == {"output_dimensionality": 768}
    assert len(texts) == len(PAST) + 1
    assert all(gloss in text for (_, gloss), text in zip(PAST, texts)) and OWN_GLOSS in texts[-1]
    assert len(client.generates) == 6
    for index, (model, question, config) in enumerate(client.generates):
        assert model == mc.FLASH and config == pipeline._director_config(mc.FLASH, 0.3)
        assert question.startswith(rf.PROMPT_HEAD + "\n\n")
        numbered = _lines(question)
        assert set(numbered) >= {gloss for _, gloss in PAST}
        if index % 2 == 0:
            assert CHOSEN_BLOCK in question and OWN_GLOSS in numbered
            assert not any(plant in numbered for plant in rf.ANY_DAY_GLOSSES)
        else:
            assert STANDING_BLOCK in question and OWN_GLOSS not in numbered
            assert rf.ANY_DAY_GLOSSES[index // 2] in numbered and len(numbered) == len(PAST) + 1


def test_at_most_25_past_glosses_newest_first(tmp_path):
    history = [{"date": (datetime(2026, 11, 20) - timedelta(days=day)).strftime("%Y-%m-%d"),
                "gloss": "Zqpast many %d." % day} for day in range(30)]
    store, client = Store(tmp_path, history=list(reversed(history))), Client()
    store.put(_record())
    _reflect(store, client)
    [(_, texts, _)] = client.embeds
    assert len(texts) == 26
    assert ["Zqpast many %d." % day in text for day, text in zip(range(25), texts)] == [True] * 25


def test_the_orders_are_shuffled_by_run_and_fixed_by_the_run_id(tmp_path):
    store = Store(tmp_path)
    store.put(_record())
    first, second = Client(), Client()
    _reflect(store, first)
    _reflect(store, second)
    assert [q for _, q, _ in first.generates] == [q for _, q, _ in second.generates]
    controls = [_lines(q) for _, q, _ in first.generates[0::2]]
    assert len({json.dumps(sorted(order.items())) for order in controls}) == 3


def test_a_clean_named(tmp_path):
    store, client = Store(tmp_path), Client(real="gloss:" + PAST[3][1])
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    stored = _stored(store, run_id)
    assert outcome["status"] == stored["status"] == "named"
    assert stored["answers"] == [PAST[3][0]] * 3 and stored["pointer_item_hashes"] is None
    assert stored["plants_ok"] is True and stored["agreement"] is True


def test_a_wrong_control_gives_void(tmp_path):
    store = Store(tmp_path)
    client = Client(control=lambda run, numbered: numbered[PAST[0][1]] if run == 2 else numbered[OWN_GLOSS])
    run_id = store.put(_record())
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["status"] == "void" and stored["plants_ok"] is False
    assert stored["control_answers"] == [1, 0, 1] and stored["pointer_item_hashes"] is None


def test_a_chosen_plant_gives_void(tmp_path):
    store, client = Store(tmp_path), Client(real=lambda run, numbered: "plant" if run == 3 else "none")
    run_id = store.put(_record())
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["status"] == "void" and stored["plants_ok"] is False
    assert stored["answers"] == ["none", "none", "plant"]


def test_three_answers_that_differ_give_void(tmp_path):
    store = Store(tmp_path)
    client = Client(real=lambda run, numbered: "none" if run < 3 else numbered[PAST[0][1]])
    run_id = store.put(_record())
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["status"] == "void" and stored["agreement"] is False and stored["plants_ok"] is True
    assert stored["answers"] == ["none", "none", PAST[0][0]]


def test_an_unreadable_or_out_of_range_reply_is_wrong_for_a_control_and_void_for_the_real_question(tmp_path):
    store = Store(tmp_path)
    client = Client(control=lambda run, numbered: "garbage" if run == 1 else numbered[OWN_GLOSS],
                    real=lambda run, numbered: 99 if run == 2 else "none")
    run_id = store.put(_record())
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["status"] == "void"
    assert stored["control_answers"] == [0, 1, 1] and stored["answers"] == ["none", None, "none"]


def test_the_arithmetic_answer_is_stored_beside_the_pointing_one(tmp_path):
    # Tonight's own gloss sits on its candidate's axis; one past gloss sits on the standing group's.
    store = Store(tmp_path)
    client = Client(vectors={OWN_GLOSS: _axis(0), PAST[2][1]: _axis(1)})
    run_id = store.put(_record())
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["arithmetic_named"] is True and stored["arithmetic_margin"] == pytest.approx(0.0, abs=1e-3)
    far = Client(vectors={OWN_GLOSS: _axis(0)})
    _reflect(store, far)
    stored = _stored(store, run_id)
    assert stored["arithmetic_named"] is False and stored["arithmetic_margin"] == pytest.approx(-1.0, abs=1e-3)


@pytest.mark.parametrize("model", ["mixed", "unknown", ""])
def test_a_record_without_one_embedding_model_has_no_arithmetic_answer_and_no_embedding_call(tmp_path, model):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record(inputs=_inputs(model=model)))
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert client.embeds == [] and len(client.generates) == 6
    assert stored["arithmetic_named"] is None and stored["arithmetic_margin"] is None
    # Without an arithmetic answer a clean "none" is not enough for a pointer.
    assert stored["status"] == "disputed" and stored["pointer_item_hashes"] is None


# ---- spend and failures ----


def _uncertain():
    return SpendBlocked("Zqsecret uncertain")


def _refused():
    error = SpendBlocked("Unregistered paid SDK method")
    error.refused_before_dispatch = True
    return error


def test_an_uncertain_spend_block_on_a_question_ends_the_run_at_once(tmp_path):
    store, client = Store(tmp_path), Client(generate_error=lambda n: _uncertain() if n == 2 else None)
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    assert len(client.generates) == 2 and outcome["status"] == "failed" and outcome["calls"] == 3
    assert _stored(store, run_id)["status"] == "failed"


def test_an_uncertain_spend_block_on_the_embedding_ends_the_run_before_any_question(tmp_path):
    store, client = Store(tmp_path), Client(embed_error=_uncertain())
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    assert len(client.embeds) == 1 and client.generates == [] and outcome["status"] == "failed"
    assert outcome["calls"] == 1 and _stored(store, run_id)["status"] == "failed"


def test_a_question_refused_before_dispatch_is_not_counted_and_voids_the_run(tmp_path):
    store, client = Store(tmp_path), Client(generate_error=lambda n: _refused() if n == 4 else None)
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    assert len(client.generates) == 6 and outcome["calls"] == 6
    stored = _stored(store, run_id)
    assert stored["status"] == "void" and stored["answers"] == ["none", None, "none"]


def test_an_embedding_refused_before_dispatch_leaves_the_pointing_task(tmp_path):
    store, client = Store(tmp_path), Client(embed_error=_refused())
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    assert outcome["calls"] == 6 and outcome["status"] == "disputed"
    assert _stored(store, run_id)["arithmetic_named"] is None


def test_a_raising_client_voids_the_run_and_never_raises(tmp_path):
    store = Store(tmp_path)
    client = Client(embed_error=RuntimeError("Zqsecret embed"), generate_error=RuntimeError("Zqsecret generate"))
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    assert outcome["status"] == "void" and outcome["calls"] == 7
    assert _stored(store, run_id)["answers"] == [None, None, None]


def test_without_a_client_the_run_is_skipped_before_any_call(tmp_path):
    store = Store(tmp_path)
    run_id = store.put(_record())
    outcome = _reflect(store, None)
    assert (outcome["status"], outcome["reason"]) == ("skipped", "no_client")
    assert _stored(store, run_id)["status"] == "skipped"


def test_a_raising_store_fails_without_raising(tmp_path):
    store, client = Store(tmp_path, fail=RuntimeError("Zqsecret store")), Client()
    assert _reflect(store, client)["status"] == "failed"
    _no_paid_call(client)


def test_never_more_than_six_question_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(reflect, "RUNS", 5)
    store, client = Store(tmp_path), Client()
    store.put(_record())
    _reflect(store, client)
    assert len(client.generates) == 6 and len(client.embeds) == 1


# ---- the command line ----


def _main(monkeypatch, capsys, argv, store, client):
    monkeypatch.setattr(reflect, "_store", lambda: store if not isinstance(store, BaseException) else (_ for _ in ()).throw(store))
    monkeypatch.setattr(reflect, "_genai_client", lambda: client)
    monkeypatch.setattr(reflect, "_now", lambda: NOW)
    code = reflect.main(argv)
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.out + out.err


def test_main_prints_one_line_of_numbers(monkeypatch, capsys, tmp_path):
    store = Store(tmp_path)
    store.put(_record())
    code, lines, everything = _main(monkeypatch, capsys, [], store, Client(vectors={OWN_GLOSS: _axis(0)}))
    assert code == 0
    assert lines == ["[reflector] status=%d plants_ok=1 agreement=1 arithmetic_named=0 calls=7"
                     % reflect.STATUS_NUMBERS["pointer"]]
    assert not any(text in everything for text in PRIVATE)


def test_main_reads_kind_arm_and_the_same_model_override(monkeypatch, capsys, tmp_path):
    store = Store(tmp_path)
    run_id = store.put(_record("private_run", arm="current", repeat=1, director=mc.FLASH))
    store.put(_record("private_run", arm="previous", repeat=1, written=NOW - timedelta(hours=1),
                      inputs=_inputs(standing=False)))
    code, lines, _ = _main(monkeypatch, capsys, ["--kind", "private_run", "--arm", "current"], store, Client())
    assert code == 0 and _stored(store, run_id)["reason"] == "same_model"
    code, lines, _ = _main(monkeypatch, capsys, ["--kind", "private_run", "--arm", "current", "--allow-same-model"],
                           store, Client())
    assert code == 0 and _stored(store, run_id)["status"] == "pointer"


def test_the_status_numbers_are_fixed_and_distinct():
    assert reflect.STATUS_NUMBERS == {"pointer": 1, "named": 2, "void": 3, "skipped": 4, "no_group": 5,
                                      "no_glosses": 6, "no_control": 7, "no_record": 8, "failed": 9,
                                      "disputed": 10}


@pytest.mark.parametrize("case", ["raising-store", "store-setup", "raising-client", "interrupt", "bad-argument"])
def test_main_exits_0_and_prints_only_numbers_whatever_fails(monkeypatch, capsys, tmp_path, case):
    store, client, argv = Store(tmp_path), Client(), []
    store.put(_record())
    if case == "raising-store":
        store.fail = RuntimeError("Zqsecret store")
    elif case == "store-setup":
        store = RuntimeError("Zqsecret setup")
    elif case == "raising-client":
        client = Client(embed_error=RuntimeError("Zqsecret"), generate_error=RuntimeError("Zqsecret"))
    elif case == "interrupt":
        client = Client(generate_error=KeyboardInterrupt())
    else:
        argv = ["--kind", "Zqsecret", "--no-such-flag"]
    code, lines, everything = _main(monkeypatch, capsys, argv, store, client)
    assert code == 0
    assert len(lines) == 1 and LINE.fullmatch(lines[0]), lines
    assert not any(text in everything for text in PRIVATE)
    expected = "void" if case == "raising-client" else "failed"
    assert lines[0].startswith("[reflector] status=%d " % reflect.STATUS_NUMBERS[expected])


# ---- a named model, and whether it is the Director's ----


def test_the_stored_result_says_whether_the_reflector_answered_on_the_directors_model(tmp_path):
    store = Store(tmp_path)
    run_id = store.put(_record(director=mc.FLASH))
    assert _reflect(store, Client())["same_model"] is True
    assert _stored(store, run_id)["same_model"] is True
    _reflect(store, Client(), allow_same_model=True)
    assert _stored(store, run_id)["same_model"] is True
    _reflect(store, Client(), model=mc.HYDROGEN)
    assert _stored(store, run_id)["same_model"] is False


def test_a_skip_before_the_record_is_read_has_no_same_model_answer(tmp_path):
    store = Store(tmp_path)
    assert _reflect(store, Client())["same_model"] is None
    assert _reflect(store, Client(), model="gemini-1.0")["same_model"] is None


def test_main_runs_on_a_named_model(monkeypatch, capsys, tmp_path):
    store = Store(tmp_path)
    run_id = store.put(_record("private_run", arm="current", repeat=1, director=mc.HYDROGEN))
    client = Client()
    code, lines, _ = _main(monkeypatch, capsys, ["--kind", "private_run", "--arm", "current", "--model", mc.HYDROGEN,
                                                 "--allow-same-model"], store, client)
    stored = _stored(store, run_id)
    assert code == 0 and stored["status"] == "pointer" and stored["model"] == mc.HYDROGEN
    assert stored["same_model"] is True
    assert {model for model, _, _ in client.generates} == {mc.HYDROGEN}


def test_main_still_refuses_a_model_outside_the_two(monkeypatch, capsys, tmp_path):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record())
    code, lines, _ = _main(monkeypatch, capsys, ["--model", "gemini-9-unknown", "--allow-same-model"], store, client)
    assert code == 0 and lines[0].startswith("[reflector] status=%d " % reflect.STATUS_NUMBERS["skipped"])
    assert client.generates == [] and _stored(store, run_id) is None


# ---- a run whose Director fell back ----


def _fell_back(**kwargs):
    record = _record(**kwargs)
    record["models"]["fallback_fired"] = True
    return record


@pytest.mark.parametrize("kwargs", [{}, {"allow_same_model": True}, {"model": mc.HYDROGEN}])
def test_a_record_whose_director_fell_back_is_skipped_without_a_paid_call(tmp_path, kwargs):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_fell_back(director=mc.FLASH))
    outcome = _reflect(store, client, **kwargs)
    assert (outcome["status"], outcome["reason"], outcome["calls"]) == ("skipped", "fallback", 0)
    _no_paid_call(client)
    stored = _stored(store, run_id)
    assert (stored["status"], stored["reason"]) == ("skipped", "fallback")
    assert ("load_pulse_feed_history", 30) not in store.calls


def test_the_fallback_skip_comes_before_the_other_early_statuses(tmp_path):
    store, client = Store(tmp_path, history=[]), Client()
    store.put(_fell_back(inputs=_inputs(standing=False, chosen=False)))
    assert _reflect(store, client)["reason"] == "fallback"


def test_a_record_without_a_models_block_is_not_a_fallback(tmp_path):
    store, client = Store(tmp_path), Client()
    record = _record()
    record["models"] = None
    store.put(record)
    assert _reflect(store, client)["status"] == "pointer"


# ---- a pointer needs the arithmetic to agree ----


def test_a_clean_none_with_arithmetic_saying_not_named_is_a_pointer(tmp_path):
    store, client = Store(tmp_path), Client(vectors={OWN_GLOSS: _axis(0)})
    run_id = store.put(_record())
    _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["arithmetic_named"] is False
    assert stored["status"] == "pointer" and stored["pointer_item_hashes"] == STANDING_HASHES


def test_a_clean_none_with_arithmetic_saying_named_is_disputed(tmp_path):
    store, client = Store(tmp_path), Client(vectors={OWN_GLOSS: _axis(0), PAST[2][1]: _axis(1)})
    run_id = store.put(_record())
    outcome = _reflect(store, client)
    stored = _stored(store, run_id)
    assert stored["arithmetic_named"] is True and stored["plants_ok"] is True and stored["agreement"] is True
    assert outcome["status"] == stored["status"] == "disputed" and stored["pointer_item_hashes"] is None


@pytest.mark.parametrize("client", [lambda: Client(embed_error=RuntimeError("Zqsecret")),
                                    lambda: Client(embed_error=_refused())], ids=["failed", "refused"])
def test_a_clean_none_without_an_arithmetic_answer_is_disputed(tmp_path, client):
    store = Store(tmp_path)
    run_id = store.put(_record())
    _reflect(store, client())
    stored = _stored(store, run_id)
    assert stored["arithmetic_named"] is None
    assert stored["status"] == "disputed" and stored["pointer_item_hashes"] is None


def test_the_arithmetic_does_not_change_a_named_or_void_verdict(tmp_path):
    store = Store(tmp_path)
    run_id = store.put(_record())
    _reflect(store, Client(real="gloss:" + PAST[3][1], vectors={OWN_GLOSS: _axis(0), PAST[2][1]: _axis(1)}))
    assert _stored(store, run_id)["status"] == "named"
    _reflect(store, Client(real="plant", vectors={OWN_GLOSS: _axis(0)}))
    assert _stored(store, run_id)["status"] == "void"


def test_main_prints_the_disputed_number(monkeypatch, capsys, tmp_path):
    store = Store(tmp_path)
    store.put(_record())
    code, lines, _ = _main(monkeypatch, capsys, [], store, Client(embed_error=_refused()))
    assert code == 0 and lines == ["[reflector] status=10 plants_ok=1 agreement=1 arithmetic_named=0 calls=6"]


# ---- a run whose Director gave no accepted reply ----


@pytest.mark.parametrize("digest", [None, "", "  ", "missing"])
def test_a_record_without_a_director_reply_is_skipped_without_a_paid_call(tmp_path, digest):
    store, client = Store(tmp_path), Client()
    record = _record()
    if digest == "missing":
        del record["prompt_sha256"]
    else:
        record["prompt_sha256"] = digest
    run_id = store.put(record)
    outcome = _reflect(store, client, allow_same_model=True)
    assert (outcome["status"], outcome["reason"], outcome["calls"]) == ("skipped", "director_failed", 0)
    _no_paid_call(client)
    stored = _stored(store, run_id)
    assert (stored["status"], stored["reason"]) == ("skipped", "director_failed")
    assert ("load_pulse_feed_history", 30) not in store.calls


def test_a_reply_on_the_older_prompt_with_no_candidate_words_still_runs(tmp_path):
    store, client = Store(tmp_path), Client()
    run_id = store.put(_record())
    assert _reflect(store, client)["status"] == "pointer"
    assert _stored(store, run_id)["status"] == "pointer"
