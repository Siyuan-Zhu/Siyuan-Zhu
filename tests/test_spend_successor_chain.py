import pytest
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore, SpendBlocked
from tests.test_apify_build_repair import receipts

OLD = {
    'price_repair_20261103': {'month': '2026-11', 'old_policy_sha256': 'a', 'new_policy_sha256': 'b'},
    'apify_build_repair_20261106': {'month': '2026-11', 'old_policy_sha256': 'b', 'new_policy_sha256': 'c'},
    'price_repair_20261113': {'month': '2026-11', 'old_policy_sha256': 'c', 'new_policy_sha256': 'd'},
}
SUCCESSOR = {'month': '2026-11', 'old_policy_sha256': 'd', 'new_policy_sha256': 'e'}
ROLLBACK = {'month': '2026-11', 'old_policy_sha256': 'e', 'new_policy_sha256': 'd'}


def test_receipt_ids_and_order():
    assert runtime.SUCCESSOR_RECEIPT_ID == 'spend_successor_20261122'
    assert runtime.ROLLBACK_RECEIPT_ID == 'spend_rollback_20261122'
    assert runtime.POLICY_REPAIR_RECEIPTS == (
        'price_repair_20261103', 'apify_build_repair_20261106', 'price_repair_20261113',
        'spend_successor_20261122', 'spend_rollback_20261122', 'spend_successor_20261125', 'spend_rollback_20261125',
        'spend_successor_20261128', 'spend_rollback_20261128',
        'spend_successor_20261128_cushion', 'spend_rollback_20261128_cushion',
        'spend_successor_december_2026', 'spend_rollback_december_2026')


def test_successor_receipt_extends_the_chain(monkeypatch):
    store = receipts(monkeypatch, {**OLD, 'spend_successor_20261122': SUCCESSOR})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'e'


def test_rollback_receipt_returns_to_the_previous_policy(monkeypatch):
    store = receipts(monkeypatch, {**OLD, 'spend_successor_20261122': SUCCESSOR,
                                   'spend_rollback_20261122': ROLLBACK})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'd'


def test_absent_new_receipts_are_skipped(monkeypatch):
    store = receipts(monkeypatch, dict(OLD))
    assert runtime._successor_policy(store, 'a', '2026-11') == 'd'


def test_successor_receipt_for_another_month_is_ignored(monkeypatch):
    store = receipts(monkeypatch, {**OLD, 'spend_successor_20261122': {**SUCCESSOR, 'month': '2026-12'}})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'd'


class Doc:
    def get(self):
        return None


class Collection:
    def document(self, document_id):
        return Doc()


class Client:
    def collection(self, name):
        return Collection()


@pytest.mark.parametrize('document_id', ['spend_successor_20261122', 'spend_rollback_20261122'])
def test_store_reads_the_approved_receipt_ids(document_id):
    PrivateFirestoreStore(Client()).read(document_id)


def test_store_still_refuses_an_unapproved_receipt_id():
    with pytest.raises(SpendBlocked, match='Unapproved private spend document'):
        PrivateFirestoreStore(Client()).read('spend_successor_20261123')
