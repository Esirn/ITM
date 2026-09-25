"""Catch same-path data replacement and unverified checkpoint normalization."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.models.mdm_conditions import INPUT_CONTRACT_VERSION, INPUT_PIPELINE_VERSION
from itm.models.mdm_imu_control import MDMIMUControlConfig
from itm.models.mdm_training_provenance import build_training_input_provenance, validate_training_resume


@pytest.fixture
def run(tmp_path):
    spec = importlib.util.spec_from_file_location('trainer', Path(__file__).resolve().parents[1] /
                                                 'scripts/train_mdm_imu_control.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    text = tmp_path/'text.txt'; text.write_text('a person stands#tokens#0#0\n')
    motion = tmp_path/'motion.npy'; np.save(motion, np.zeros((4, 263)))
    imu = tmp_path/'imu.npz'
    np.savez(imu, acceleration=np.zeros((4, 6, 3)), orientation=np.broadcast_to(np.eye(3), (4, 6, 3, 3)),
             motion_id='fixture', fps=20., sensor_mask=np.ones(6, dtype=bool))
    mr = dict(motion_id='fixture', split='train', text_path=str(text), joint_vec_path=str(motion),
              joints_path=str(motion))
    ir = dict(motion_id='fixture', split='train', imu_path=str(imu), num_frames=4, fps=20., schema='standard_imu_v1')
    artifacts = {key: tmp_path/key for key in ('motion_manifest', 'imu_manifest', 'motion_mean', 'motion_std',
                                              'mdm_checkpoint', 'mdm_args')}
    artifacts['motion_manifest'].write_text(json.dumps(mr)+'\n')
    artifacts['imu_manifest'].write_text(json.dumps(ir)+'\n')
    for key, array in [('motion_mean', np.zeros(263)), ('motion_std', np.ones(263))]:
        with artifacts[key].open('wb') as stream:
            np.save(stream, array)
    artifacts['mdm_checkpoint'].write_bytes(b'backbone fixture')
    artifacts['mdm_args'].write_text('{"predict_xstart":true,"latent_dim":512}')
    ds = trainer._PairedDataset(artifacts['motion_manifest'], artifacts['imu_manifest'], np.zeros(263),
                               np.ones(263), {'head': (4,)}, None, tmp_path)
    norm = ds.compute_normalization()
    return SimpleNamespace(trainer=trainer, ds=ds, norm=norm, artifacts=artifacts, text=text,
                           motion=motion, imu=imu, root=tmp_path)


def provenance(run, **changes):
    options = dict(artifacts=run.artifacts, pipeline_version=INPUT_PIPELINE_VERSION,
                   contract_version=INPUT_CONTRACT_VERSION)
    options.update(changes)
    return build_training_input_provenance(run.ds, run.norm, **options)


def checkpoint(run):
    return dict(training_input_provenance=provenance(run), imu_normalization=copy.deepcopy(run.norm))


@pytest.mark.parametrize('source', ['text', 'motion', 'imu', 'motion_manifest', 'imu_manifest',
                                    'motion_mean', 'motion_std', 'mdm_checkpoint', 'mdm_args'])
def test_same_path_replacement_rejected_even_when_statistics_unchanged(run, source):
    state = checkpoint(run)
    path = getattr(run, source) if source in ('text', 'motion', 'imu') else run.artifacts[source]
    # Appending bytes suffices to change content identity, without changing the
    # in-memory statistics: checking just the path or mean/std would miss this.
    with path.open('ab') as stream:
        stream.write(b'\nchanged')
    with pytest.raises(ValueError, match='provenance differs'):
        validate_training_resume(state, provenance(run), run.norm)


def test_selected_config_pipeline_contract_and_caption_plan_are_bound(run):
    state = checkpoint(run)
    for changes in ({'pipeline_version': 'itm-method-caption-seconds-v3'},
                    {'contract_version': 'future-contract'}):
        with pytest.raises(ValueError, match='provenance differs'):
            validate_training_resume(state, provenance(run, **changes), run.norm)
    plan = run.root/'plan.jsonl'; plan.write_text('{}\n')
    run.artifacts['caption_plan'] = plan
    with pytest.raises(ValueError, match='provenance differs'):
        validate_training_resume(state, provenance(run), run.norm)
    state = checkpoint(run)
    plan.write_text('{"caption_index":1}\n')
    with pytest.raises(ValueError, match='provenance differs'):
        validate_training_resume(state, provenance(run), run.norm)
    state = checkpoint(run)
    sample, record, _, _ = run.ds.items[0]
    run.ds.items[0] = sample, record, 'wrists', (0, 1)
    with pytest.raises(ValueError, match='provenance differs'):
        validate_training_resume(state, provenance(run), run.norm)


def test_legacy_or_altered_checkpoint_statistics_rejected(run):
    with pytest.raises(ValueError, match='lacks training input provenance'):
        validate_training_resume({'imu_normalization': run.norm}, provenance(run), run.norm)
    state = checkpoint(run)
    state['imu_normalization']['acceleration_mean'][0][0] = 123.
    with pytest.raises(ValueError, match='IMU normalization differs'):
        validate_training_resume(state, provenance(run), run.norm)


def test_checkpoint_serializes_provenance_and_allows_unchanged_warm_start(run):
    encoder = torch.nn.Linear(1, 1)
    controlled = SimpleNamespace(adapters=torch.nn.Linear(1, 1))
    optimizer = torch.optim.AdamW(encoder.parameters())
    path = run.root/'control.pt'
    run.trainer._save_checkpoint(path, torch, encoder, controlled, optimizer, MDMIMUControlConfig(),
                                {'head': (4,)}, run.artifacts['mdm_checkpoint'], run.norm, [], 1,
                                SimpleNamespace(caption_plan=None, contract_version=INPUT_CONTRACT_VERSION),
                                provenance(run))
    state = torch.load(path, weights_only=True)
    validate_training_resume(state, provenance(run), run.norm)
    assert state['resume_semantics'] == 'input_bound_weights_verified_normalization_fresh_optimizer'
    with pytest.raises(FileExistsError):
        run.trainer._save_checkpoint(path, None, None, None, None, None, None, None, None, None, None, None, None)


@pytest.mark.parametrize('legacy', [False, True])
def test_cli_rejects_bad_resume_before_model_import_or_device_setup(run, monkeypatch, legacy):
    state = checkpoint(run)
    if legacy:
        del state['training_input_provenance']
    else:
        state['imu_normalization']['acceleration_std'][0][0] = 999.
    resume = run.root/'resume.pt'; torch.save(state, resume)
    argv = ['train', '--allow-unsynchronized-diagnostic', '--resume', str(resume),
            '--mdm-root', str(run.root/'absent-mdm'), '--output', str(run.root/'new.pt'), '--sensor-configs', 'head']
    flags = {'motion_manifest': '--manifest', 'imu_manifest': '--standard-imu-manifest',
             'motion_mean': '--mean', 'motion_std': '--std', 'mdm_checkpoint': '--mdm-checkpoint', 'mdm_args': '--mdm-args'}
    for name, flag in flags.items():
        argv.extend([flag, str(run.artifacts[name])])
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(run.trainer, '_legacy_compatibility', lambda: None)
    monkeypatch.setattr(run.trainer.os, 'chdir', lambda _: pytest.fail('Must reject before MDM import'))
    monkeypatch.setattr(run.trainer, '_check_device', lambda *_: pytest.fail('Must reject before device setup'))
    with pytest.raises(ValueError, match='lacks training input provenance' if legacy else 'IMU normalization differs'):
        run.trainer.main()
    assert not (run.root/'new.pt').exists()
