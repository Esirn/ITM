import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.models.mdm_caption_conditions import prepare_caption_input, require_caption_training_admission

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    text = tmp_path/'caption.txt'
    text.write_text('bad caption#tokens#nan#0\nselected caption#tokens#1#2\n')
    motion = tmp_path/'motion.npy'
    np.save(motion, np.broadcast_to(np.arange(60)[:,None],(60,263)).copy())
    cache = tmp_path/'imu.npz'
    np.savez(cache, acceleration=np.broadcast_to((np.arange(60)/20)[:,None,None],(60,6,3)).copy(),
             orientation=np.broadcast_to(np.eye(3),(60,6,3,3)), joints=np.zeros((60,24,3)),
             fps=20., motion_id=np.array('fixture'), sensor_mask=np.ones(6,dtype=bool))
    mr=dict(motion_id='fixture',split='train',text_path=str(text),joint_vec_path=str(motion),joints_path=str(motion))
    ir=dict(motion_id='fixture',split='train',imu_path=str(cache),schema='standard_imu_v1',fps=20.,num_frames=60)
    plan=plan_caption_crop(parse_timed_caption('selected caption#tokens#1#2'),60,np.arange(60)/20)
    plan={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in plan.items()}
    plan.update(motion_id='fixture',split='train',caption='selected caption',caption_index=1,
                sensor_configs=['head','wrists'],status='time_interval_plan_passed_not_alignment_certified',
                artifacts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [text,motion,cache]])
    norm=dict(acceleration_mean=np.zeros((6,3)).tolist(),acceleration_std=np.ones((6,3)).tolist())
    return plan,mr,ir,norm,tmp_path


def test_crop_uses_selected_index_source_seconds_and_rebased_model_time(inputs):
    plan,mr,ir,norm,root=inputs
    item=prepare_caption_input(plan,mr,ir,norm,workspace_root=root)
    assert item['caption']=='selected caption'
    np.testing.assert_array_equal(item['motion'][:,0],np.arange(20,40))
    np.testing.assert_allclose(item['imu'][:,4,0],np.arange(20,40)/20)
    np.testing.assert_allclose(item['source_times_seconds'],np.arange(20,40)/20)
    np.testing.assert_allclose(item['model_times_seconds'],np.arange(20)/20)
    assert item['input_metadata']['training_admissible'] is False


def test_caption_sampling_and_training_loader_agree(inputs):
    plan,mr,ir,norm,root=inputs
    sampler=script('sample_mdm_imu_control');trainer=script('train_mdm_imu_control')
    mpath=root/'manifest.jsonl';mpath.write_text(json.dumps(mr)+'\n')
    ipath=root/'cache.jsonl';ipath.write_text(json.dumps(ir)+'\n')
    ds=trainer._PairedDataset(mpath,ipath,np.zeros(263),np.ones(263),{'head':(4,)},None,root,
                             caption_plans={'fixture':plan})
    ds.set_normalization(norm);item=ds[0]
    sample=sampler._load_case({'motion_id':'fixture'}, {'fixture':ir}, norm,
                             caption_plan=plan,motion_records={'fixture':mr})
    np.testing.assert_array_equal(item['imu'],sample['imu'])
    assert sample['text']==item['caption']=='selected caption'
    _,y=trainer._collate([item])
    np.testing.assert_allclose(y['imu_timestamps'].numpy()[0],np.arange(20)/20)
    stats=ds.compute_normalization()
    np.testing.assert_allclose(stats['acceleration_mean'],np.full((6,3),1.475))


def test_overrides_hash_drift_and_modified_time_plan_rejected(inputs):
    plan,mr,ir,norm,root=inputs
    helper=script('sample_mdm_imu_control')
    for override in ({'length':19},{'text':'other caption'},{'imu_mode':'shuffled'}):
        with pytest.raises(ValueError):
            helper._load_case(dict(motion_id='fixture',**override),{'fixture':ir},norm,
                              caption_plan=plan,motion_records={'fixture':mr})
    with pytest.raises(ValueError,match='time axis'):
        prepare_caption_input(dict(plan,source_times_seconds=plan['model_times_seconds']),mr,ir,norm,workspace_root=root)
    Path(mr['text_path']).write_text('changed caption#tokens#1#2\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        prepare_caption_input(plan,mr,ir,norm,workspace_root=root)


def test_blocked_plan_and_false_admission_are_not_bypassed(inputs):
    plan,mr,ir,norm,root=inputs
    with pytest.raises(ValueError,match='blocked'):
        prepare_caption_input(dict(plan,status='blocked',reason='199 frames'),mr,ir,norm,workspace_root=root)
    with pytest.raises(ValueError,match='not admitted'):
        require_caption_training_admission({'fixture':plan})
    with pytest.raises(ValueError,match='not admitted'):
        require_caption_training_admission({'fixture':dict(plan,training_admissible=True)})


def test_nonzero_crop_does_not_reuse_prefix_response_witness(inputs):
    plan,mr,ir,norm,root=inputs
    item=prepare_caption_input(plan,mr,ir,norm,workspace_root=root)
    with pytest.raises(ValueError,match='separate witness'):
        script('train_mdm_imu_control').validate_response_intervals([{}],[item['input_metadata']])


def test_training_cli_rejects_diagnostic_sidecar_before_torch_setup(inputs,monkeypatch):
    from types import SimpleNamespace
    plan,mr,ir,norm,root=inputs
    path=root/'plans.jsonl';path.write_text(json.dumps(plan)+'\n')
    trainer=script('train_mdm_imu_control')
    monkeypatch.setattr(trainer,'parse_args',lambda:SimpleNamespace(
        allow_unsynchronized_diagnostic=True,text_drop_probability=.1,caption_plan=str(path)))
    monkeypatch.setattr(trainer,'require_torch',lambda:pytest.fail('Must reject before model/device setup'))
    with pytest.raises(ValueError,match='not admitted'):
        trainer.main()
