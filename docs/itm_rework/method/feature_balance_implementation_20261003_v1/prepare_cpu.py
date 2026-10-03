"""Guarded immutable preparation and independent train-statistics audit."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
from run_mdm_feature_balance_bounded import prepare
from itm.models.mdm_feature_balance import FeaturePackage
from itm.models.mdm_correction import array_sha
from itm.models.mdm_run_limits import sha256


def forbidden(*args,**kwargs):
    raise AssertionError('CPU preparation attempted model construction/forward/GPU/launch')


def main(output):
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module,'__init__',forbidden))
        stack.enter_context(patch.object(torch.nn.Module,'_call_impl',forbidden))
        for name in ('init','_lazy_init','is_available','device_count','get_device_properties','synchronize'):
            stack.enter_context(patch.object(torch.cuda,name,forbidden))
        for name in ('preflight_gpu','supervise'):
            stack.enter_context(patch('run_mdm_feature_balance_bounded.'+name,forbidden))
        manifest=prepare(output)
        package=FeaturePackage()
        clips={}
        for row in package.schedule:
            if row['motion_id'] not in clips:clips[row['motion_id']]=package.inputs(row)['raw_acceleration']
        raw=np.concatenate([clips[k] for k in sorted(clips)])
        assert len(raw)==398 and sorted(clips)==['000027','000506','004728']
        norm=package.common['imu_normalization']
        mean=raw.astype(np.float64).mean(0).astype(np.float32)
        std=raw.astype(np.float64).std(0).astype(np.float32)
        mean_error=float(np.max(np.abs(mean-np.asarray(norm['acceleration_mean']))))
        std_error=float(np.max(np.abs(std-np.asarray(norm['acceleration_std']))))
        assert mean_error<=2e-7 and std_error<=2e-7
        for entry in manifest['expected_artifacts']:
            assert sha256(entry['path'])==entry['sha256']
        parameters=sum(int(np.prod(r['shape'])) for r in manifest['parameter_contract'])
    print(json.dumps(dict(status='passed',implementation_sha256=sha256(output/'implementation.json'),
        bound_files=len(manifest['expected_artifacts']),sampling_inputs=len(manifest['inputs']),training_inputs=len(manifest['training_inputs']),
        unique_training_frames=398,train_mean_max_abs=mean_error,train_std_max_abs=std_error,
        trainable_parameter_tensors=len(manifest['parameter_contract']),trainable_parameters=parameters,
        loss_weights_sha256=package.loss['weights_sha256'],model_constructions=0,real_forwards=0,GPU_queries=0,
        new_initializations=0,new_checkpoints=0),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
