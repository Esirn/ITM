"""Explicit adapter for owner-02's admitted, pre-canonicalized synthetic B0 pairs."""
import json
from pathlib import Path

import numpy as np

from itm.data.canonical_pair import SCHEMA, MAPPING_VERSION
from itm.data.observation import rotation_valid
from itm.models.mdm_caption_conditions import prepare_caption_input, _path, _sha

CANONICAL_PIPELINE_VERSION = 'itm-method-canonical-b0-v1'
CANONICAL_CONTRACT_VERSION = 'itm-rework-v0.3-draft'
CANONICAL_SCOPE = 'offline_synthetic_only_not_real_device_calibration'
CANONICAL_ALIGNMENT = 'admitted_precanonicalized_synthetic_B0_only'


def load_canonical_cache(record, *, workspace_root):
    if record.get('schema') != SCHEMA or record.get('mapping_version') != MAPPING_VERSION:
        raise ValueError('Explicit canonical schema and mapping required')
    with np.load(_path(record['imu_path'], workspace_root), allow_pickle=False) as data:
        for key, expected in [('schema', SCHEMA), ('mapping_version', MAPPING_VERSION)]:
            if key not in data or data[key].shape != () or data[key].item() != expected:
                raise ValueError(f'Canonical embedded {key} mismatch')
        arrays = {key: data[key].copy() for key in ('acceleration', 'orientation', 'sensor_mask', 'fps',
                                                   'timestamps', 'source_to_canonical_rotation', 'source_origin')}
    n = record['num_frames']
    if n < 3 or float(arrays['fps']) != 30. or record['fps'] != 30.:
        raise ValueError('Canonical cache requires declared 30 Hz support')
    if arrays['timestamps'].shape != (n,) or not np.array_equal(arrays['timestamps'], np.arange(n)/30.):
        raise ValueError('Canonical timestamps disagree with cache support')
    if arrays['acceleration'].shape != (n, 6, 3) or arrays['orientation'].shape != (n, 6, 3, 3):
        raise ValueError('Invalid canonical IMU shape')
    if arrays['sensor_mask'].shape != (6,) or arrays['sensor_mask'].dtype != np.bool_ or not arrays['sensor_mask'].all():
        raise ValueError('Canonical six-slot cache required')
    if not np.isfinite(arrays['acceleration']).all() or not rotation_valid(arrays['orientation']).all():
        raise ValueError('Invalid canonical acceleration/orientation')
    q, origin = arrays['source_to_canonical_rotation'], arrays['source_origin']
    if q.shape != (3, 3) or not rotation_valid(q) or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Invalid saved canonical transform')
    return arrays


class CanonicalB0Package:
    """Resolve the delivered package, not arbitrary manifests with admitted flags.

    Does not grant compute authorization or infer calibration from target GT.
    """
    def __init__(self, path, *, workspace_root):
        self.root = Path(workspace_root).resolve()
        self.path = _path(path, self.root).resolve()
        self.package = json.loads(self.path.read_text())
        if self.package['version'] != 'canonical-B0-delivery-v1':
            raise ValueError('Unsupported canonical delivery version')
        self.provenance_artifacts = {'canonical_package': self.path}
        for name in ('admission', 'build', 'verification', 'semantic_review', 'budget'):
            self.provenance_artifacts['canonical_'+name] = self.verify(self.package[name])
        self.budget = json.loads(self.provenance_artifacts['canonical_budget'].read_text())
        self.manifests = {}
        for entry in self.budget['data']['manifest_artifacts']:
            path = self.verify(entry)
            self.manifests[path.stem] = path
            self.provenance_artifacts['canonical_'+path.stem] = path
        self.sampling_plan_path = self.verify(self.budget['sampling_plan'])
        self.provenance_artifacts['canonical_sampling_plan'] = self.sampling_plan_path
        self.admissions = self.rows(self.provenance_artifacts['canonical_admission'], split_key=True)
        self.motions, self.imus, self.plans = {}, {}, {}
        for split, ids_key in [('train', 'training_ids'), ('val', 'validation_ids')]:
            expected = set(self.budget['data'][ids_key])
            for dest, suffix in [(self.motions, 'motion'), (self.imus, 'imu'), (self.plans, 'caption_plans')]:
                dest[split] = self.rows(self.manifests[split+'_'+suffix])
                if set(dest[split]) != expected or any(r['split'] != split for r in dest[split].values()):
                    raise ValueError('Canonical manifest IDs/split differ from the delivered budget')
            if len(expected) != self.package['admitted_'+split]:
                raise ValueError('Canonical admission count mismatch')
            for mid in expected:
                self._check_admission(split, mid)
        groups = [{r['source_group_id'] for r in self.motions[s].values()} for s in ('train', 'val')]
        if groups[0] & groups[1]:
            raise ValueError('Canonical train/validation source overlap')

    @staticmethod
    def rows(path, *, split_key=False):
        result = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['split'], str(row['motion_id'])) if split_key else str(row['motion_id'])
            if key in result:
                raise ValueError('Duplicate canonical record')
            result[key] = row
        return result

    def verify(self, entry):
        path = _path(entry['path'], self.root).resolve()
        if _sha(path) != entry['sha256']:
            raise ValueError(f'Canonical artifact hash mismatch: {path}')
        return path

    def require_manifests(self, split, motion, imu, captions):
        for actual, suffix in [(motion, 'motion'), (imu, 'imu'), (captions, 'caption_plans')]:
            if actual is None or _path(actual, self.root).resolve() != self.manifests[split+'_'+suffix]:
                raise ValueError('Use the delivered canonical motion/IMU/caption manifests together')

    def _check_admission(self, split, mid):
        row = self.admissions[(split, mid)]
        plan, imu, motion = self.plans[split][mid], self.imus[split][mid], self.motions[split][mid]
        for obj in (row, plan):
            for key, expected in [('schema', SCHEMA), ('mapping_version', MAPPING_VERSION),
                                  ('benchmark_admissible', True), ('world_alignment_certified', True),
                                  ('semantic_alignment_certified', True), ('world_alignment_scope', CANONICAL_SCOPE),
                                  ('status', 'admitted_synthetic_B0_only')]:
                if obj.get(key) != expected:
                    raise ValueError(f'Canonical admission mismatch: {mid} {key}')
            if obj.get('training_admissible') is not (split == 'train'):
                raise ValueError('Canonical train-only admission required')
        if plan['admission_artifact'] != self.package['admission'] or plan['artifacts'] != row['artifacts'] or imu['artifacts'] != row['artifacts']:
            raise ValueError('Canonical admission/artifact chain mismatch')
        if plan['caption_index'] != row['caption_index'] or plan['caption'] != row['caption']:
            raise ValueError('Canonical reviewed caption mismatch')
        for key in ('text_path', 'joint_vec_path', 'joints_path'):
            if _path(motion[key], self.root).resolve() != _path(row[key], self.root).resolve():
                raise ValueError('Canonical motion path differs from admission')
        if imu['imu_path'] != row['imu_path'] or motion['source_group_id'] != row['source_group_id'] or imu['source_group_id'] != row['source_group_id']:
            raise ValueError('Canonical cache/source group differs from admission')
        if plan['annotation_kind'] != 'full_clip_sentinel' or plan['frame_slice'] != [0, row['motion_frames']]:
            raise ValueError('Canonical B0 requires the reviewed full clip')

    def prepare(self, split, mid, normalization=None):
        mid = str(mid)
        if split not in self.plans or mid not in self.plans[split]:
            raise ValueError('Motion is not admitted to this canonical split')
        self._check_admission(split, mid)
        row = self.admissions[(split, mid)]
        for artifact in row['artifacts']:
            self.verify(artifact)
        item = prepare_caption_input(self.plans[split][mid], self.motions[split][mid], self.imus[split][mid],
                                     normalization, workspace_root=self.root, cache_loader=load_canonical_cache)
        cache = load_canonical_cache(self.imus[split][mid], workspace_root=self.root)
        for key in ('source_to_canonical_rotation', 'source_origin'):
            if not np.array_equal(cache[key], np.asarray(row[key])):
                raise ValueError('Saved canonical transform differs from admission')
        gt = np.load(_path(row['joints_path'], self.root), allow_pickle=False)
        if gt.shape != (len(item['motion']), 22, 3) or not np.isfinite(gt).all():
            raise ValueError('Canonical direct 20 Hz joint target shape/value mismatch')
        item['gt'] = gt.astype(np.float32)
        item['input_metadata'].update(
            input_pipeline_version=CANONICAL_PIPELINE_VERSION, input_contract_version=CANONICAL_CONTRACT_VERSION,
            schema=SCHEMA, mapping_version=MAPPING_VERSION, alignment_status=CANONICAL_ALIGNMENT,
            world_alignment_scope=CANONICAL_SCOPE, semantic_alignment_scope=row['semantic_alignment_scope'],
            training_admissible=(split == 'train'), benchmark_admissible=True,
            clock_reference='saved_canonical_pair_seconds_30Hz_IMU_to_20Hz_model',
            feature_recanonicalization='none_runtime; target_precanonicalized_offline',
            notebook_clock_correction='none_runtime; source_mapping_bound_in_admission',
            package_sha256=_sha(self.path), admission_artifact=self.package['admission'],
            gt_source='direct_20Hz_joints_npy', gt_path=row['joints_path'],
            source_group_id=row['source_group_id'], source_to_canonical_rotation=row['source_to_canonical_rotation'],
            source_origin=row['source_origin'], deployable_real_IMU_calibration=False)
        return item
