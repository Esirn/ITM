import hashlib
import pytest
from restore_git_archive_snapshots import restore


def entry():
    return dict(canonical_path='source.py',archive_path='snapshot/source.py',sha256=hashlib.sha256(b'content').hexdigest())


def test_dry_run_then_restore_does_not_overwrite(tmp_path):
    (tmp_path/'source.py').write_bytes(b'content')
    assert restore(tmp_path,[entry()])[0]['action']=='would_copy'
    assert not (tmp_path/'snapshot').exists()
    assert restore(tmp_path,[entry()],apply=True)[0]['action']=='copy'
    assert restore(tmp_path,[entry()],apply=True)[0]['action']=='already_present'
    (tmp_path/'snapshot/source.py').write_bytes(b'other')
    with pytest.raises(ValueError):restore(tmp_path,[entry()],apply=True)
    assert (tmp_path/'snapshot/source.py').read_bytes()==b'other'


def test_all_entries_validate_before_any_writes(tmp_path):
    (tmp_path/'source.py').write_bytes(b'content')
    bad=dict(entry(),archive_path='snapshot/other.py',sha256='bad')
    with pytest.raises(ValueError):restore(tmp_path,[entry(),bad],apply=True)
    assert not (tmp_path/'snapshot').exists()


def test_path_outside_checkout_rejected(tmp_path):
    bad=dict(entry(),archive_path='../outside.py')
    with pytest.raises(ValueError):restore(tmp_path,[bad],apply=True)
