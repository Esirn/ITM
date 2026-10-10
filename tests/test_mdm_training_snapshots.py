"""CPU tensor identity refusal tests; no neural module is constructed."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from itm.models.mdm_correction import tensor_hashes
from itm.models.mdm_feature_balance import parameter_contract, ARM, SCHEMA

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_snapshots', ROOT/'scripts/audit_b0_training_snapshots.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def fixture():
    initial = dict(imu_encoder={'weight':torch.ones(2)},
                   adapters={'0.weight':torch.zeros(2),'0.bias':torch.zeros(2)},
                   control_config=dict(reference_mode='none'))
    initial['tensor_hashes']=tensor_hashes(initial)
    optimizer=dict(name='AdamW',lr=1e-4,betas=[.9,.999])
    package=SimpleNamespace(common=dict(optimizer=optimizer))
    identity=dict(package_sha256='fixture',implementation_sha256='fixture')
    state=deepcopy(initial);state['adapters']['0.weight']+=1
    state.update(identity,checkpoint_schema=SCHEMA,arm=ARM,
                 control_config=dict(reference_mode='static_delta'),
                 execution=dict(updates_completed=50,updates_attempted=50,final_update_200=False,
                                exposures=150,mdm_examples=150,encoder_examples=300,caption_calls=150),
                 training_run_directory=str(audit.RUN),
                 training_audit=dict(trainable=parameter_contract(initial),frozen_unchanged=True),
                 optimizer_defaults=dict(lr=1e-4,betas=(.9,.999)),
                 optimizer=dict(param_groups=[dict(params=[0,1])],state={0:dict(step=torch.tensor(50.)),1:dict(step=torch.tensor(50.))}))
    state['tensor_hashes']=tensor_hashes(state)
    return state,package,initial,identity


def test_fixed_intermediate_snapshot_is_accepted_only_with_matching_step():
    state,package,initial,identity=fixture()
    audit.validate_f_snapshot(state,package,initial,50,identity)
    with pytest.raises(ValueError,match='update/exposure'):
        audit.validate_f_snapshot(state,package,initial,200,identity)
    with pytest.raises(ValueError,match='predeclared'):
        audit.validate_f_snapshot(state,package,initial,51,identity)


@pytest.mark.parametrize('change,match', [
    ('provenance','provenance'),('configuration','config'),('run','another run'),
    ('step','optimizer step'),('missing_optimizer','Incomplete'),('identity','tensor identity'),
    ('frozen_bias','bias'),('nan','finite'),('unchanged','initialization')])
def test_corrupt_or_relabelled_checkpoint_is_rejected(change,match):
    state,package,initial,identity=fixture()
    if change=='provenance':state['implementation_sha256']='wrong'
    if change=='configuration':state['control_config']['reference_mode']='none'
    if change=='run':state['training_run_directory']='/tmp/wrong_run'
    if change=='step':state['optimizer']['state'][0]['step']=torch.tensor(49.)
    if change=='missing_optimizer':del state['optimizer']['state'][0]
    if change=='identity':state['adapters']['0.weight']+=1
    if change=='frozen_bias':
        state['adapters']['0.bias']+=1;state['tensor_hashes']=tensor_hashes(state)
    if change=='nan':
        state['imu_encoder']['weight'][0]=float('nan');state['tensor_hashes']=tensor_hashes(state)
    if change=='unchanged':
        state['adapters']=deepcopy(initial['adapters']);state['tensor_hashes']=tensor_hashes(state)
    with pytest.raises(ValueError,match=match):
        audit.validate_f_snapshot(state,package,initial,50,identity)


def test_distinct_weight_drift_is_not_convergence_measure():
    state,_,initial,_=fixture()
    result=audit.drift(state,initial)
    assert result['imu_encoder']['rms_from_initial']==0
    assert result['adapters']['rms_from_initial']==pytest.approx(2**-.5)
    assert result['adapters']['initial_rms']==0


def test_real_audit_has_six_unique_files_and_seven_CPU_loads(monkeypatch):
    # Count existing-file CPU loads. Forbid model construction and GPU queries.
    original=torch.load;loads=[]
    def counted(path,*args,**kwargs):
        assert kwargs.get('map_location')=='cpu' and kwargs.get('weights_only') is True
        loads.append(str(Path(path).resolve()))
        return original(path,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError('No model construction or GPU query in CPU snapshot audit')
    monkeypatch.setattr(torch,'load',counted)
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    for name in ('is_available','device_count','get_device_properties','current_device'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    result=audit.audit()
    assert result['tensor_deserializations']==len(loads)==7
    assert result['unique_checkpoint_files']==len(set(loads))==6
    assert result['model_constructions']==result['model_forwards']==result['GPU_queries']==0
