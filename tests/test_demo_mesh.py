"""SMPL preview cache and input boundaries, without running a body fit."""

import json

import pytest
from fastapi import HTTPException

from itm.demo import app as demo


def test_mesh_cache_reused_and_invalid_panel_rejected(tmp_path, monkeypatch):
    model = tmp_path / 'body_models/smpl/SMPL_NEUTRAL.pkl'
    model.parent.mkdir(parents=True)
    model.touch()
    monkeypatch.setattr(demo, 'MDM_ROOT', tmp_path)
    monkeypatch.setattr(demo, 'MESH_ROOT', tmp_path / 'cache')
    motion = [[[0., 0., 0.]] * 22] * 3
    monkeypatch.setattr(demo, 'get_run', lambda _: {'panels': [{'motion': motion}]})
    calls = []

    def fit(output, model, device):
        (output / 'metadata.json').write_text(json.dumps({'frames': 3}))
        calls.append(device)

    monkeypatch.setattr(demo, '_run_mesh_job', fit)
    request = demo.MeshRequest(run_id='example', panel_index=0, device='cpu')
    first = demo.prepare_mesh(request)
    assert first == demo.prepare_mesh(request)
    assert len(calls) == 1
    with pytest.raises(HTTPException) as exc:
        demo.prepare_mesh(demo.MeshRequest(run_id='example', panel_index=1))
    assert exc.value.status_code == 400
    unrelated = demo.MESH_ROOT / ('b' * 64)
    unrelated.mkdir()
    key = demo._mesh_identity({'motion': motion})[2]
    demo.MESH_ACTIVE.add(key)
    try:
        with pytest.raises(HTTPException) as exc:
            demo.clear_mesh(demo.MeshClearRequest(run_id='example'))
        assert exc.value.status_code == 409
    finally:
        demo.MESH_ACTIVE.remove(key)
    assert demo.clear_mesh(demo.MeshClearRequest(run_id='example'))['removed'] == 1
    assert unrelated.exists()
    assert not (demo.MESH_ROOT / key).exists()
    demo.prepare_mesh(request)
    assert len(calls) == 2


@pytest.mark.parametrize('cache_id, filename', [('..', 'vertices.bin'), ('a' * 64, 'motion.npy')])
def test_mesh_download_rejects_nonpublic_assets(cache_id, filename):
    with pytest.raises(HTTPException) as exc:
        demo.mesh_cache(cache_id, filename)
    assert exc.value.status_code == 400


def test_device_load_reservations_and_generation(monkeypatch):
    monkeypatch.setattr(demo, '_gpu_load', lambda: [(0, 20000, 0), (1, 23000, 0)])
    monkeypatch.setattr(demo, 'MESH_DEVICES', set())
    monkeypatch.setattr(demo, 'MESH_GPU_FINISHED', {})
    assert demo._mesh_device('auto') == 'cuda:1'
    demo.MESH_DEVICES.add('cuda:1')
    assert demo._mesh_device('auto') == 'cuda:0'
    assert demo._mesh_device('cuda:1') is None
    monkeypatch.setattr(demo, '_gpu_load', lambda: [(0, 2000, 0), (1, 23000, 80)])
    assert demo._mesh_device('auto') is None
    demo.MESH_DEVICES.clear()
    assert demo._mesh_device('auto') == 'cpu'
    demo.MESH_DEVICES.add('cpu')
    assert demo._mesh_device('auto') is None
    demo.MESH_DEVICES.clear()
    demo.GENERATION_LOCK.acquire()
    try:
        assert demo._mesh_device('cuda:0') is None
        assert demo._mesh_device('auto') == 'cpu'
    finally:
        demo.GENERATION_LOCK.release()


def test_gpu_all_uses_every_gpu_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(demo, '_gpu_load', lambda: [(0, 20000, 0), (1, 19000, 0), (2, 18000, 0)])
    monkeypatch.setattr(demo, 'MESH_DEVICES', set())
    assert demo.mesh_devices()['gpus'] == [0, 1, 2]
    for expected in ('cuda:0', 'cuda:1', 'cuda:2'):
        assert demo._mesh_device('gpu_all') == expected
        demo.MESH_DEVICES.add(expected)
    assert demo._mesh_device('gpu_all') is None
    demo.MESH_DEVICES.clear()
    monkeypatch.setattr(demo, '_gpu_load', lambda: [(0, 20000, 90)])
    assert demo._mesh_device('gpu_all') is None
    monkeypatch.setattr(demo, '_gpu_load', lambda: [])
    assert demo._mesh_device('gpu_all') is None
