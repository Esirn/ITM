"""Regression tests for B0. Tensor-level correctness is not generation evidence."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from itm.models.mdm_imu_control import (MDMIMUControlConfig, make_imu_control_encoder,
    install_imu_control, make_text_imu_guidance_model, configure_control_training,
    evaluating_preserving_modes)
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm.models.semantic_response import relative_response_loss, semantic_response_step


def config(**kwargs):
    return MDMIMUControlConfig(latent_dim=16, encoder_layers=1, encoder_heads=4,
                              dropout=0, imu_drop_probability=0, **kwargs)


class TinyMDM(torch.nn.Module):
    rot2xyz = None
    translation = True
    njoints, nfeats = 16, 1
    data_rep, cond_mode = "hml_vec", "text"

    def __init__(self):
        super().__init__()
        self.seqTransEncoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(16, 4, dropout=0), 2)
        self.text = torch.nn.Parameter(torch.randn(1, 1, 16))

    def encode_text(self, texts):
        return self.text.expand(1, len(texts), -1) * (1 + torch.tensor([len(s) for s in texts])[None, :, None])

    def forward(self, x, t, y=None):
        if y.get("raise_before_encoder"):
            raise RuntimeError("intentional")
        text = y.get("text_embed")
        if text is None:
            text = self.encode_text(y.get("text", ["a"] * len(x)))
        if y.get("uncond", False):
            text = torch.zeros_like(text)
        src = torch.cat((text, x[:, :, 0].permute(2, 0, 1)))
        valid = y.get("imu_frame_mask", torch.ones(len(x), x.shape[-1], dtype=torch.bool))
        mask = torch.cat((torch.zeros(len(x), 1, dtype=torch.bool), ~valid), 1)
        return self.seqTransEncoder(src, src_key_padding_mask=mask)[1:].permute(1, 2, 0)[:, :, None]


def setup():
    torch.manual_seed(12)
    original = TinyMDM().eval()
    model = deepcopy(original)
    controlled, encoder, hook = install_imu_control(model, config(encoder_variant="ordered_concat", null_mode="strict", time_encoding="physical_sinusoidal"))
    for adapter in controlled.adapters:
        torch.nn.init.normal_(adapter.weight, std=0.1)
        torch.nn.init.normal_(adapter.bias, std=0.3)
    model.eval()
    x = torch.randn(2, 16, 1, 5)
    y = {"imu": torch.randn(2, 5, 6, 12), "sensor_mask": torch.tensor([[1,1,0,0,0,0],[0,0,0,0,1,0]]).bool(),
         "imu_frame_mask": torch.ones(2, 5, dtype=torch.bool), "text": ["left", "right"]}
    return original, model, x, y


def test_legacy_normalized_wrist_swap_invariance_and_state_loading():
    torch.manual_seed(1)
    encoder = make_imu_control_encoder(config()).eval()
    copy = make_imu_control_encoder(config()).eval()
    copy.load_state_dict(encoder.state_dict())
    imu = torch.randn(2, 8, 6, 12)
    mask = torch.tensor([[1,1,0,0,0,0]] * 2).bool()
    swapped = imu.clone(); swapped[:, :, [0,1]] = imu[:, :, [1,0]]
    torch.testing.assert_close(encoder(imu, mask), copy(swapped, mask), atol=1e-6, rtol=1e-5)


def test_ordered_identity_padding_time_and_empty_conditions():
    torch.manual_seed(1)
    encoder = make_imu_control_encoder(config(encoder_variant="ordered_concat", null_mode="strict", time_encoding="physical_sinusoidal")).eval()
    imu = torch.randn(2, 8, 6, 12)
    mask = torch.tensor([[1,1,0,0,0,0]] * 2).bool()
    valid = torch.tensor([[1,1,1,1,0,0,0,0], [0,0,0,0,0,0,0,0]]).bool()
    baseline = encoder(imu, mask, valid)
    swapped = imu.clone(); swapped[:, :, [0,1]] = imu[:, :, [1,0]]
    assert not torch.allclose(baseline[0,:4], encoder(swapped, mask, valid)[0,:4])
    changed = imu.clone(); changed[:, :, 2:] = float("nan"); changed[:, 4:] = float("nan")
    torch.testing.assert_close(baseline, encoder(changed, mask, valid))
    torch.testing.assert_close(baseline[:1,:4], encoder(imu[:1,:4], mask[:1]), atol=1e-6, rtol=1e-5)
    assert baseline[1].count_nonzero() == 0
    assert encoder(imu, mask * 0, valid).count_nonzero() == 0
    times = torch.arange(8)[None].expand(2,-1) / 10
    assert not torch.allclose(baseline[0,:4], encoder(imu, mask, valid, timestamps=times)[0,:4])


def test_strict_null_nonzero_bias_mixed_cache_and_exception_cleanup():
    original, model, x, y = setup()
    t = torch.zeros(2)
    reference = original(x, t, y)
    torch.testing.assert_close(reference, model(x,t,dict(y, imu_uncond=True)), atol=1e-6, rtol=1e-5)
    mixed = model(x,t,dict(y, imu_uncond=torch.tensor([True,False])))
    torch.testing.assert_close(reference[0], mixed[0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(reference[1], mixed[1])
    control, presence = model.imu_control_encoder(y['imu'],y['sensor_mask'],y['imu_frame_mask'],return_presence=True)
    cached = dict(y, imu_control=control, imu_condition_present=presence)
    torch.testing.assert_close(model(x,t,y), model(x,t,cached))
    torch.testing.assert_close(reference, model(x,t,dict(cached,imu_uncond=True)))
    with pytest.raises(RuntimeError):
        model(x,t,dict(y,raise_before_encoder=True))
    assert model.seqTransEncoder.control is None
    with pytest.raises(ValueError):
        model(x,t,dict(cached,imu_control=control[:,:2]))
    assert model.seqTransEncoder.control is None
    torch.testing.assert_close(reference,model(x,t,{'text':y['text']}))


def test_strict_cfg_sequential_batched_uncached_and_identities():
    _, model, x, y = setup(); t=torch.zeros(2)
    control,presence=model.imu_control_encoder(y['imu'],y['sensor_mask'],y['imu_frame_mask'],return_presence=True)
    cached=dict(y,text_embed=model.encode_text(y['text']),imu_control=control,imu_condition_present=presence)
    seq=make_text_imu_guidance_model(model,mode='factorized').eval()
    bat=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched').eval()
    for ts,ss,js in [(1.,1.,1.),(2.,0.,0.),(0.,1.,0.),(torch.tensor([1.,2.]),torch.tensor([0.,3.]),.5)]:
        cond=dict(cached,text_scale=ts,imu_scale=ss,joint_scale=js)
        torch.testing.assert_close(seq(x,t,cond),bat(x,t,cond),atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(seq(x,t,dict(y,text_scale=ts,imu_scale=ss,joint_scale=js)),seq(x,t,cond))
    torch.testing.assert_close(seq(x,t,y),model(x,t,y),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(seq(x,t,dict(y,text_scale=0.,imu_scale=1.,joint_scale=0.)),model(x,t,dict(y,uncond=True)),atol=2e-6,rtol=1e-5)


def test_frozen_modes_gradients_and_dropout():
    _, model, x,y=setup(); configure_control_training(model)
    before=[m.training for m in model.modules()]
    with pytest.raises(RuntimeError):
        with evaluating_preserving_modes(model):
            assert not any(m.training for m in model.modules())
            raise RuntimeError()
    assert before==[m.training for m in model.modules()]
    model(x,torch.zeros(2),y).square().mean().backward()
    for name,param in model.named_parameters():
        trainable=name.startswith(('imu_control_encoder.','seqTransEncoder.adapters.'))
        assert param.requires_grad==trainable
        assert (param.grad is not None)==trainable
    cfg=replace(model.imu_control_encoder.config,imu_drop_probability=1.)
    dropped=make_imu_control_encoder(cfg).train()
    _,present=dropped(y['imu'],y['sensor_mask'],return_presence=True)
    assert not present.any()


def test_addressed_full_loop_is_batch_order_and_global_rng_independent():
    class Posterior:
        num_timesteps=4
        def p_mean_variance(self,model,x,t,**kwargs):
            return {'mean':x*.5,'log_variance':torch.zeros_like(x)}
    model=TinyMDM().eval(); lengths=torch.tensor([3,5])
    sample,plans=sample_addressed_ddpm(model,Posterior(),y={},lengths=lengths,case_ids=['a','b'],sample_seed=19,backbone_id='tiny')
    torch.randn(100)
    reverse,other=sample_addressed_ddpm(model,Posterior(),y={},lengths=lengths.flip(0),case_ids=['b','a'],sample_seed=19,backbone_id='tiny')
    torch.testing.assert_close(sample,reverse.flip(0))
    assert plans[0]['random_plan_id']==other[1]['random_plan_id']
    assert [r['step'] for r in plans[0]['noise_draws']]==[-1,3,2,1]


def test_response_loss_teacher_detached_and_confidence_accounted():
    a=torch.zeros(2,1,requires_grad=True); b=torch.zeros(2,1,requires_grad=True)
    teacher=torch.tensor([[.1],[.001]],requires_grad=True)
    loss,diag=relative_response_loss(a,b,torch.zeros_like(a),teacher,compatibility_weight=torch.tensor([1.,.5]))
    loss.backward()
    assert teacher.grad is None and b.grad[0]<0 and b.grad[1]==0
    assert diag['accepted_attributes']==1 and diag['total_attributes']==2


def test_semantic_response_step_keeps_teacher_frozen_and_modes():
    _,model,x,y=setup(); configure_control_training(model)
    before=[m.training for m in model.modules()]
    def decode(value):
        return value[:,:,0].transpose(1,2)[...,:3,None].transpose(-1,-2).expand(-1,-1,22,-1)
    witnesses=[dict(split='train',compatibility_evidence='synthetic fixture only',attribute_spec='arm_spread_m')]*2
    losses,diag=semantic_response_step(model,x,torch.zeros(2),y,caption0=['a','b'],caption1=['wide arms','spread'],witnesses=witnesses,
        decode_joints=decode,target_proxy_joints=torch.randn(2,5,22,3),compatibility_weight=torch.ones(2))
    sum(losses.values()).backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(('imu_control_encoder.','seqTransEncoder.adapters.')))
    assert before==[m.training for m in model.modules()]
    assert diag['teacher_forwards']==2


def test_film_baseline_strict_null_with_trained_bias():
    torch.manual_seed(12)
    original=TinyMDM().eval();model=deepcopy(original)
    controlled,encoder,_=install_imu_control(model,config(encoder_variant='ordered_concat',null_mode='strict',adapter_variant='film'))
    for adapter in controlled.adapters:
        torch.nn.init.normal_(adapter.weight,std=.1);torch.nn.init.normal_(adapter.bias,std=.1)
    model.eval(); x=torch.randn(1,16,1,4);t=torch.zeros(1)
    y={'text':['hello'],'imu':torch.randn(1,4,6,12),'sensor_mask':torch.ones(1,6,dtype=torch.bool),'imu_uncond':True}
    torch.testing.assert_close(original(x,t,y),model(x,t,y),atol=1e-6,rtol=1e-5)


def test_physical_crop_never_stretches_donor(tmp_path):
    import numpy as np
    spec=importlib.util.spec_from_file_location('sample_helpers',Path(__file__).parents[1]/'scripts/sample_mdm_imu_control.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    def write(name,n):
        path=tmp_path/name
        acc=np.broadcast_to(np.arange(n)[:,None,None],(n,6,3)).astype(np.float32).copy()
        np.savez(path,acceleration=acc,orientation=np.broadcast_to(np.eye(3),(n,6,3,3)),joints=np.zeros((n,24,3)),fps=20.)
        return {'imu_path':str(path)}
    records={'a':write('a.npz',8),'b':write('b.npz',4)}
    norm={'acceleration_mean':np.zeros((6,3)),'acceleration_std':np.ones((6,3))}
    case=helpers._load_case({'motion_id':'a','length':4},records,norm)
    np.testing.assert_allclose(case['raw_acceleration'][:,0,0],[0,1,2,3])
    with pytest.raises(ValueError,match='too short'):
        helpers._load_case({'motion_id':'a','imu_motion_id':'b','imu_mode':'shuffled'},records,norm)


def test_training_dropout_combinations_and_missing_witness_guard(tmp_path):
    spec=importlib.util.spec_from_file_location('train_helpers',Path(__file__).parents[1]/'scripts/train_mdm_imu_control.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    _,model,x,y=setup(); configure_control_training(model)
    dropped,counts=helpers.training_condition_dropout(model,y,1.,1.)
    assert counts=={'00':2,'01':0,'10':0,'11':0}
    assert dropped['text_embed'].count_nonzero()==0
    assert dropped['imu_condition_present'].count_nonzero()==0
    kept,counts=helpers.training_condition_dropout(model,y,0.,0.)
    assert counts['11']==2
    empty=tmp_path/'empty.jsonl';empty.write_text('')
    with pytest.raises(ValueError,match='No verified'):
        helpers.load_response_manifest(empty)
