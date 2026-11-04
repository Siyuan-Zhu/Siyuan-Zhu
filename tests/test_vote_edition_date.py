"""The real canonical visitor CLI votes on the accepted edition, not UTC today."""
import copy
from datetime import datetime, timezone
import json

import pytest

from scripts import process_visitor_event as visitor
from src import state_store as state
from tests.test_canonical_runtime_factory import fixtures, ZERO
from tests.test_staging_application_rehearsal import ApplicationWire


def canonical(tmp_path, monkeypatch, *, edition='2026-11-03'):
    root, _, _, initial = fixtures(tmp_path)
    wire = ApplicationWire(monkeypatch)
    wire.docs.update(copy.deepcopy(initial.docs))
    wire.versions.update({key: 1 for key in wire.docs})
    wire.docs['profile_state/daily_pulse']['date'] = edition
    monkeypatch.setenv('PULSE_VOTE_SALT', 'synthetic-edition-vote-salt')
    store = state.FirestoreStateStore(client=wire.client, repo_root=root, canonical_only=True)
    return root, wire, store


def freeze(monkeypatch, instant):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)
    monkeypatch.setattr(visitor, 'datetime', Clock)


def arguments(root, mood='spot_on'):
    return ['--repo-root', str(root), '--login', 'edition-voter',
            '--title', 'pulse:vote:' + mood, '--issue-number', '701', '--require-accepted']


@pytest.mark.parametrize('instant', [
    datetime(2026, 11, 4, 1, 30, tzinfo=timezone.utc),  # UTC crossed; Eastern has not.
    datetime(2026, 11, 4, 5, 30, tzinfo=timezone.utc),  # Eastern crossed; 02:00 has not.
    datetime(2026, 11, 6, 12, 30, tzinfo=timezone.utc),  # Retained last accepted edition.
])
def test_real_cli_votes_once_on_retained_edition_across_clock_boundaries(tmp_path, monkeypatch, capsys, instant):
    root, wire, store = canonical(tmp_path, monkeypatch)
    freeze(monkeypatch, instant)
    game = copy.deepcopy(wire.docs['profile_state/active_game'])
    assert visitor.main(arguments(root), store=store) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'recorded_vote'
    assert result['tally'] == dict(ZERO, spot_on=1)
    assert wire.docs['pulse_votes/2026-11-03'] == result['tally']
    assert 'pulse_votes/' + instant.date().isoformat() not in wire.docs
    assert visitor.main(arguments(root, 'too_sunny'), store=store) == 3
    duplicate = json.loads(capsys.readouterr().out)
    assert duplicate['status'] == 'ignored' and duplicate['tally'] == result['tally']
    rate_keys = [key for key in wire.docs if key.startswith('vote_rate_limits/')]
    assert len(rate_keys) == 1 and rate_keys[0].startswith('vote_rate_limits/2026-11-03_')
    assert wire.docs['profile_state/active_game'] == game
    pulse = wire.docs['profile_state/daily_pulse']
    assert pulse['date'] == '2026-11-03' and pulse['word'] == 'CANONICAL'


@pytest.mark.parametrize('field', ['date_iso', 'analysis'])
def test_existing_canonical_edition_fallback_fields_remain_supported(tmp_path, monkeypatch, capsys, field):
    root, wire, store = canonical(tmp_path, monkeypatch)
    pulse = wire.docs['profile_state/daily_pulse']
    edition = pulse.pop('date')
    pulse[field] = {'date_iso': edition} if field == 'analysis' else edition
    freeze(monkeypatch, datetime(2026, 11, 4, 1, tzinfo=timezone.utc))
    assert visitor.main(arguments(root), store=store) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'recorded_vote'
    assert wire.docs['pulse_votes/' + edition]['spot_on'] == 1


@pytest.mark.parametrize('edition', [None, '', 'not-a-date', '2026-02-30', '2026-1-01'])
def test_missing_or_malformed_canonical_edition_never_guesses_or_records_vote(tmp_path, monkeypatch, edition):
    root, wire, store = canonical(tmp_path, monkeypatch, edition=edition)
    before = copy.deepcopy(wire.docs)
    with pytest.raises(RuntimeError, match='edition date'):
        visitor.main(arguments(root), store=store)
    assert wire.docs == before
    assert not any(method == 'POST' for method, _ in wire.touched)


def test_publication_racing_actual_cli_vote_rejects_old_target_without_retargeting(tmp_path, monkeypatch, capsys):
    root, wire, store = canonical(tmp_path, monkeypatch)
    freeze(monkeypatch, datetime(2026, 11, 4, 1, tzinfo=timezone.utc))
    def publish(server):
        server.docs['profile_state/daily_pulse'].update(
            date='2026-11-04', word='NEW-EDITION', visitor_mood_tally=dict(ZERO), mood_votes=dict(ZERO))
        server.versions['profile_state/daily_pulse'] += 1
    wire.before_commit = publish
    assert visitor.main(arguments(root), store=store) == 3
    assert json.loads(capsys.readouterr().out)['status'] == 'ignored'
    assert wire.docs['profile_state/daily_pulse']['date'] == '2026-11-04'
    assert wire.docs['profile_state/daily_pulse']['visitor_mood_tally'] == ZERO
    assert not any(key.startswith(('vote_rate_limits/', 'pulse_votes/')) for key in wire.docs)
    assert len(wire.commits) == 1  # Failed CAS; no replay against the newer edition.


def test_local_legacy_vote_date_behavior_is_unchanged(tmp_path, monkeypatch):
    instant = datetime(2026, 11, 4, 1, tzinfo=timezone.utc)
    freeze(monkeypatch, instant)
    store = state.InMemoryStateStore()
    seen = []
    original = store.record_mood_vote
    def record(**values):
        seen.append(values['date_str'])
        return original(**values)
    monkeypatch.setattr(store, 'record_mood_vote', record)
    result = visitor.process_event(tmp_path, 'local-voter', 'pulse:vote:spot_on', store=store)
    assert result['status'] == 'recorded_vote'
    assert seen == ['2026-11-04']
