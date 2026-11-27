"""A client may narrow the guard's request timeout for its own calls, never widen it (SIY-284)."""
import pytest

from src.zeitgeist.spend_guard import SpendBlocked, guard_genai_client
from tests.test_hydrogen_guard import OP, guard, request


def test_default_timeout_unchanged():
    bounded = guard().bounded_genai_request(OP, request())
    assert bounded['config']['http_options'] == {'timeout': 300000, 'retry_options': {'attempts': 1}}


def test_shorter_timeout_is_applied():
    bounded = guard().bounded_genai_request(OP, request(), max_request_seconds=100)
    assert bounded['config']['http_options'] == {'timeout': 100000, 'retry_options': {'attempts': 1}}


def test_timeout_equal_to_the_policy_duration_is_allowed():
    bounded = guard().bounded_genai_request(OP, request(), max_request_seconds=300)
    assert bounded['config']['http_options']['timeout'] == 300000


def test_longer_timeout_never_widens():
    with pytest.raises(SpendBlocked):
        guard().bounded_genai_request(OP, request(), max_request_seconds=301)


def test_the_shortest_timeout_google_accepts_is_allowed():
    bounded = guard().bounded_genai_request(OP, request(), max_request_seconds=10)
    assert bounded['config']['http_options']['timeout'] == 10000


@pytest.mark.parametrize('value', [0, -5, 9, True, 1.5, '100'])
def test_timeout_must_be_a_whole_number_of_seconds_google_accepts(value):
    # Google refuses a deadline under about 10 s with a 400 (live probe, 27 Nov).
    with pytest.raises(SpendBlocked):
        guard().bounded_genai_request(OP, request(), max_request_seconds=value)


def test_caller_http_options_still_refused():
    with pytest.raises(SpendBlocked):
        guard().bounded_genai_request(OP, request(config={'http_options': {'timeout': 1}}), max_request_seconds=100)


class _Reached(Exception):
    pass


class _RecordingGuard:
    def __init__(self):
        self.seen = None

    def bounded_genai_request(self, operation, kwargs, **options):
        self.seen = options
        raise _Reached()


class _Client:
    class models:
        @staticmethod
        def generate_content(**kwargs):
            raise AssertionError('never dispatched in this test')


def test_client_wrapper_passes_the_timeout_to_the_guard():
    recording = _RecordingGuard()
    with pytest.raises(_Reached):
        guard_genai_client(_Client(), guard=recording, request_timeout_seconds=100).models.generate_content(
            model='gemini-4-neon', contents='x')
    assert recording.seen == {'max_request_seconds': 100}


def test_client_wrapper_without_a_timeout_passes_nothing():
    recording = _RecordingGuard()
    with pytest.raises(_Reached):
        guard_genai_client(_Client(), guard=recording).models.generate_content(model='gemini-4-neon', contents='x')
    assert recording.seen == {}
