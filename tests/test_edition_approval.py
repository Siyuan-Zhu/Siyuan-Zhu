import pytest
from src.zeitgeist.edition_approval import require_edition_approval
from src.zeitgeist.publication_commit import publish_bundle
from src.state_store import FirestoreStateStore


def test_november_three_hold_is_not_a_wall_clock_expiry():
    with pytest.raises(PermissionError):
        require_edition_approval({'date':'2026-11-03','publication_id':'unapproved'})
    require_edition_approval({'date':'2026-11-04','publication_id':'future'})


def test_canonical_publication_stops_before_client_or_rendering():
    store=object.__new__(FirestoreStateStore)
    store.collection_prefix=""
    with pytest.raises(PermissionError):
        publish_bundle(store,{}, {'date':'2026-11-03','publication_id':'unapproved'},expected={},local_feed={},prepare=lambda *a:pytest.fail('must not render'))


def test_direct_production_generation_is_held_before_storage_or_providers(tmp_path):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    store=object.__new__(FirestoreStateStore);store.collection_prefix=''
    with pytest.raises(PermissionError):
        ZeitgeistADKOrchestrator(tmp_path,store).run('2026-11-03')


def test_exact_owner_approved_publication_identity_is_accepted():
    require_edition_approval({'date':'2026-11-03','publication_id':'ff043f9a96df9c9a42ec'})
    with pytest.raises(PermissionError):
        require_edition_approval({'date':'2026-11-03','publication_id':'other'})
