"""Frozen D/L/R inputs, addressed training and shared-reference control.

This module never grants compute authorization. Preparing initialization is CPU
parameter construction only; the execution worker requires a bounded context.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from itm.data.random_plan import sampling_noise
from itm.data.standard_imu import sensor_mask
from itm.models.mdm_canonical_conditions import CanonicalB0Package
from itm.models.mdm_conditions import normalize_imu
from itm.models.mdm_imu_control import MDMIMUControlConfig, make_imu_control_encoder
from itm.models.mdm_run_limits import atomic_artifact, sha256

PACKAGE_SHA = 'e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a'
ARMS = ('L_exact_length', 'R_static_reference_delta')
ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def verify(entry, root=ROOT):
    path = (root / entry['path']).resolve()
    if sha256(path) != entry['sha256']:
        raise ValueError('Correction artifact identity mismatch: '+str(path))
    return path


class CorrectionPackage:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if sha256(self.path) != PACKAGE_SHA:
            raise ValueError('Use the exact owner-02 frozen correction package')
        self.package = read(self.path)
        self.files = {Path(a['path']).stem: verify(a) for a in self.package['artifacts']}
        self.design = read(self.files['design'])
        self.budget = read(self.files['budget'])
        self.schedule = read(self.files['training_schedule'])['rows']
        self.probes = read(self.files['probe_slots'])
        self.noise = read(self.files['probe_noise'])
        self.samples = read(self.files['sampling_slots'])
        self.common = self.design['common']
        for entry in self.budget['dependencies']:
            verify(entry)
        self.data = CanonicalB0Package(verify(self.budget['dependencies'][0]), workspace_root=ROOT)
        self.mean, self.std = [np.load(verify(a)).astype(np.float32) for a in self.common['motion_normalization']]
        if np.__version__ != read(self.files['training_schedule'])['numpy_version']:
            raise ValueError('Frozen RNG requires the bound numpy version')

    def inputs(self, row):
        split = row.get('data_split', row.get('split', 'train'))
        item = self.data.prepare(split, row['motion_id'], self.common['imu_normalization'])
        if len(item['motion']) != row['frames']:
            raise ValueError('Frozen length mismatch')
        if row.get('caption_index', item['input_metadata']['caption_index']) != item['input_metadata']['caption_index']:
            raise ValueError('Frozen caption mismatch')
        # Original observations, before any diagnostic transform, define reference.
        a, o = item['acceleration'], item['orientation']
        condition = row.get('condition', 'paired')
        observed, reference, a, o = transform_inputs(a, o, self.common['imu_normalization'], condition)
        slots = [4] if row['sensor_config'] == 'head' else [0, 1]
        return dict(motion=((item['motion']-self.mean)/self.std).astype(np.float32),
                    imu=observed, reference=reference, raw_acceleration=a, raw_orientation=o,
                    original_orientation0=item['orientation'][0].copy(),
                    sensor_mask=sensor_mask(slots, sensor_count=6),
                    timestamps=item['model_times_seconds'], gt=item['gt'], text=item['caption'],
                    metadata=item['input_metadata'])


def transform_inputs(acceleration, orientation, normalization, condition):
    a, o = acceleration.copy(), orientation.copy()
    static_a = np.zeros_like(a)
    static_o = np.broadcast_to(o[:1], o.shape).copy()
    reference = normalize_imu(static_a, static_o, normalization)
    if condition == 'physical_static':
        a, o = static_a, static_o
    elif condition == 'tail_reversed':
        index = np.r_[0, np.arange(len(a)-1, 0, -1)]
        a, o = a[index].copy(), o[index].copy()
    elif condition not in ('paired', 'text_only', 'imu_only', 'backbone_bypass', 'numeric_zero'):
        raise ValueError('Unknown frozen input condition')
    observed = normalize_imu(a, o, normalization)
    if condition == 'numeric_zero':
        observed = np.zeros_like(observed)
    return observed, reference, a, o


def training_noise(row):
    spec = row['noise']
    value = np.random.Generator(np.random.PCG64(int(spec['address_sha256'], 16))).standard_normal(spec['shape']).astype(np.float32)
    if array_sha(value) != spec['noise_sha256']:
        raise ValueError('Training noise identity mismatch')
    return value


def probe_noise(spec, padded=False):
    values = []
    for entry in [spec['valid_noise']] + ([spec['padding_noise']] if padded and spec['padding_noise'] else []):
        value, metadata = sampling_noise(entry['shape'], **{k: entry[k] for k in ('backbone_id', 'case_id', 'sample_seed', 'step')})
        if metadata != entry:
            raise ValueError('Probe noise identity mismatch')
        values.append(value)
    return np.concatenate(values, axis=-1)


@contextmanager
def isolated_rng(seed, device):
    device = torch.device(device)
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        yield


def encode_control(encoder, y, *, mode='none', seed=None, guard=None):
    if mode not in ('none', 'static_delta'):
        raise ValueError('Unknown reference mode')
    if encoder.training and seed is None:
        raise ValueError('Training requires a frozen encoder dropout seed')
    def branch(value):
        if guard:
            guard.reserve('encoder_examples', len(value))
        kwargs = dict(condition_present=y['imu_condition_present'], timestamps=y['imu_timestamps'], return_presence=True)
        if seed is None:
            result = encoder(value, y['sensor_mask'], y['imu_frame_mask'], **kwargs)
        else:
            with isolated_rng(seed, value.device):
                result = encoder(value, y['sensor_mask'], y['imu_frame_mask'], **kwargs)
        if guard:
            guard.ensure_finite(result[0], 'encoded control')
            guard.check()
        return result
    control, present = branch(y['imu'])
    if mode == 'static_delta':
        reference, _ = branch(y['imu_reference'])
        control = control-reference
    return control, present


class ConditionCache:
    """Worker-local cache; fail closed on any source/model/mode identity change."""
    def __init__(self):
        self.identity = self.value = None

    def get(self, encoder, y, *, checkpoint_sha, mode, guard=None):
        if encoder.training:
            raise ValueError('Condition cache requires eval')
        key = (checkpoint_sha, mode, *(array_sha(y[k]) for k in (
            'imu', 'imu_reference', 'sensor_mask', 'imu_frame_mask', 'imu_timestamps', 'imu_condition_present')))
        if self.identity is not None and self.identity != key:
            raise ValueError('Condition cache identity mismatch')
        if self.value is None:
            with torch.no_grad():
                self.value = encode_control(encoder, y, mode=mode, guard=guard)
            self.identity = key
        return self.value


def accumulate_update(loss_functions, optimizer, parameters, guard):
    """Three independent true-length graphs, same denominator even for null."""
    if len(loss_functions) != 3:
        raise ValueError('Exactly three exposures per update')
    guard.before_update()
    optimizer.zero_grad(set_to_none=True)
    values = []
    for compute in loss_functions:
        guard.check()
        loss = compute()
        guard.ensure_finite(loss, 'microbatch loss')
        values.append(float(loss.detach()))
        if loss.requires_grad:
            (loss/3).backward()
        guard.check()
        guard.ensure_finite([p.grad for p in parameters if p.grad is not None], 'accumulated gradients')
    # No gradient clipping: the frozen correction experiment explicitly forbids it.
    guard.before_update()
    guard.state['updates_attempted'] += 1
    guard.persist()
    optimizer.step()
    guard.ensure_finite(parameters, 'updated parameters')
    guard.ensure_finite(list(optimizer.state.values()), 'optimizer state')
    guard.state['updates_completed'] += 1
    guard.persist()
    guard.check()
    return values


def tensor_hashes(state):
    return {group: {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha(v)) for k, v in values.items()}
            for group, values in state.items() if group in ('imu_encoder', 'adapters')}


def environment():
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                default_dtype=str(torch.get_default_dtype()), threads=torch.get_num_threads(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def create_initial_state(package, output):
    """No backbone loading or forward; single fresh E and 8 zero W/b on CPU."""
    config = MDMIMUControlConfig(**package.common['control_config'])
    with isolated_rng(package.common['fresh_seed'], 'cpu'):
        encoder = make_imu_control_encoder(config)
        adapters = torch.nn.ModuleList([torch.nn.Linear(config.latent_dim, config.latent_dim) for _ in range(8)])
        for layer in adapters:
            torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)
    state = dict(imu_encoder=encoder.state_dict(), adapters=adapters.state_dict(), control_config=asdict(config),
                 initialization_seed=package.common['fresh_seed'], package_sha256=PACKAGE_SHA,
                 environment=environment(), model_forwards=0)
    state['tensor_hashes'] = tensor_hashes(state)
    atomic_artifact(output, lambda stream: torch.save(state, stream))
    return dict(artifact=artifact(output), tensor_hashes=state['tensor_hashes'], environment=state['environment'])
