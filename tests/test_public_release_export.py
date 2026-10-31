"""Release republication provenance remains public without private source notes."""
import pytest


def test_release_day_projection_keeps_safe_republication_provenance():
    from scripts.export_public_release import _editorial
    record={'kind':'initial_release_republication','original_publication_id':'a'*20,
            'original_publication_date':'2026-10-30','original_published_at':'2026-10-30T15:02:01Z',
            'new_inference_performed':False,'artwork_reused_byte_exact':True}
    public=_editorial({'date':'2026-10-31','publication_id':'b'*20,
                      'release_republication':dict(record,private_source_note='must stay private')})
    assert public['release_republication']==record
    assert 'private_source_note' not in str(public)


@pytest.mark.parametrize('key,value',[
    ('original_publication_id','not-an-id'),('original_publication_date','2026-99-99'),
    ('original_published_at','private note'),('original_published_at','2026-10-30T15:02:01'),
    ('original_published_at','2026-10-31T15:02:01Z'),
    ('new_inference_performed','false'),('new_inference_performed',True),
    ('artwork_reused_byte_exact',False),
])
def test_untrusted_republication_metadata_cannot_become_public_claims(key,value):
    from scripts.export_public_release import _editorial
    record={'kind':'initial_release_republication','original_publication_id':'a'*20,
            'original_publication_date':'2026-10-30','original_published_at':'2026-10-30T15:02:01Z',
            'new_inference_performed':False,'artwork_reused_byte_exact':True}
    record[key]=value
    with pytest.raises((ValueError,TypeError)):
        _editorial({'release_republication':record})
