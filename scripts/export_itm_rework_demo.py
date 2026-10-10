#!/usr/bin/env python
"""Export existing B0 final motions to the condition lab; no model calls."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'itm-rework-20261009-v1'
ACTIONS = {'000506': '交替弓步', '000433': '踢腿后返回站立',
           '000382': '拍手后放下手臂', '000139': '站立／摇摆后行走',
           '000027': '出拳后转身', '004728': '右腿后举、保持、放下'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def export():
    frozen = ROOT / 'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json'
    slots = read(frozen)
    lookup = {(s['data_split'], s['motion_id'], s['sensor_config'], s['condition']): s for s in slots}
    old = ROOT / 'outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1'
    hashes = {a['path']: a['sha256'] for a in read(old / 'sources.json')}
    evaluated = ROOT / 'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2/sources.json'
    hashes.update({a['path']: a['sha256'] for a in read(evaluated)})
    f_root = ROOT / 'outputs/itm_rework/method/feature_balance_authorized_20261008_v1/samples'
    knockout_index = ROOT / 'outputs/itm_rework/protocol/imu_main_knockout_evaluation_20261009_v1/factorial_finals.json'
    knockouts = {r['sensor_config']: r for r in read(knockout_index)
                 if r['trajectory_id'].startswith('MK')}
    output = ROOT / 'outputs/itm_rework/protocol/user_motion_review_20261009_v1'
    hub = ROOT / 'src/itm/demo/static/rework-review.html'
    run_ids = [f'{PREFIX}-{m}-{c}' for m in ACTIONS for c in ('wrists', 'head')]
    if output.exists() or hub.exists() or any((ROOT / 'outputs/demo_runs' / r).exists() for r in run_ids):
        raise FileExistsError('Review export already exists; keep saved outputs intact.')
    output.mkdir(parents=True)
    sources, catalog = {}, []

    def load(path, key, expected=None):
        path = Path(path)
        actual = sha(path)
        bound = expected or hashes.get(str(path))
        if bound is None or actual != bound:
            raise ValueError(f'Unbound or changed source: {path}')
        sources[str(path)] = actual
        if path.suffix == '.npy':
            a = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as z:
                a = z[key].copy()
        if key in ('motion', 'joints') and a.ndim == 4:
            a = a[0]
        return a

    for motion_id, action in ACTIONS.items():
        for config in ('wrists', 'head'):
            paired = next(s for s in slots if s['motion_id'] == motion_id
                          and s['sensor_config'] == config and s['condition'] == 'paired')
            split, frames = paired['data_split'], paired['frames']
            gt = load(paired['gt_path'], 'joints')
            request = dict(mode='saved_review', experiment=action, split=split,
                           sample_a=motion_id, sample_b=None, sensor_config=config,
                           seed=paired['sampling_seed'], text_scale=1., imu_scale=1.,
                           saved_results_only=True)
            panels = []

            def panel(label, values, kind, text, imu, source):
                if values.shape != (frames, 22, 3) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid motion shape or values: {label}')
                panels.append(dict(label=label, kind=kind, text_source='A' if text else None,
                                   imu_source='A' if imu else None,
                                   caption=paired['text'] if text else '无文本条件',
                                   motion=values.tolist(), saved_source=source))

            panel('GT · 真实参考动作', gt, 'ground_truth', True, True, paired['gt_path'])
            for arm, condition, label in [
                    ('F', 'text_only', '仅文本 · 原 MDM 路径'),
                    ('F', 'paired', '文本＋IMU · F 损失配重候选'),
                    ('U', 'paired', '文本＋IMU · U 旧模型'),
                    ('F', 'physical_static', '文本＋静态 IMU · F 对照'),
                    ('F', 'imu_only', '仅 IMU · F')]:
                slot = lookup.get((split, motion_id, config, condition))
                if slot is None:
                    continue  # Training panel has no existing IMU-only samples.
                path = f_root / (slot['case_id'] + '.npz') if arm == 'F' else Path(slot['historical_output']['path'])
                mask = load(path, 'frame_mask')
                if mask.shape != (1, frames) or mask.dtype != np.bool_ or not mask.all():
                    raise ValueError(f'Invalid original frame mask: {path}')
                panel(label, load(path, 'motion'), 'generated', slot['text_present'],
                      slot['imu_present'], str(path))
            if motion_id == '000506':
                r = knockouts[config]
                panel('去掉 IMU 主引导项 · F（前100步仅文本）',
                      load(r['raw']['path'], 'joints', r['raw']['sha256']),
                      'generated', True, True, r['raw']['path'])
            observed_path = f_root / (paired['case_id'] + '.npz')
            acc = load(observed_path, 'acceleration')[0]
            imu = {f'A · {"head" if slot == 4 else "left wrist" if slot == 0 else "right wrist"}':
                   dict(acceleration=acc[:, slot].tolist(), sensor_config=config)
                   for slot in paired['sensor_slots']}
            run_id = f'{PREFIX}-{motion_id}-{config}'
            result = dict(run_id=run_id, request=request, fps=20, frame_count=frames,
                          samples={'A': dict(motion_id=motion_id, caption=paired['text'], frames=frames)},
                          panels=panels, imu=imu,
                          review_note='既有真实采样。训练/开发验证小面板；不是正式测试结果。')
            directory = ROOT / 'outputs/demo_runs' / run_id
            directory.mkdir(parents=True, exist_ok=False)
            write(directory / 'request.json', request)
            write(directory / 'result.json', result)
            # Check serialized motions retain every original float32 coordinate.
            for before, after in zip(panels, read(directory / 'result.json')['panels']):
                np.testing.assert_array_equal(np.asarray(before['motion'], dtype=np.float32),
                                              np.asarray(after['motion'], dtype=np.float32))
            catalog.append(dict(run_id=run_id, action=action, motion_id=motion_id,
                                split=split, sensor_config=config, frames=frames,
                                panels=len(panels), url='/?run=' + run_id,
                                result_sha256=sha(directory / 'result.json')))
    write(output / 'catalog.json', catalog)
    write(output / 'sources.json', [{'path': p, 'sha256': h} for p, h in sorted(sources.items())])
    rows = ''.join(f'<tr><td>{html.escape(r["action"])}</td><td>{r["split"]}</td>'
                   f'<td>{r["sensor_config"]}</td><td>{r["frames"] / 20:.2f} 秒</td>'
                   f'<td><a href="{r["url"]}">打开同步动作对比</a></td></tr>' for r in catalog)
    hub.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ITM 当前实验动作复盘</title><style>body{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;background:#f5f7fa;color:#17212e}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}a{color:#1269bd}</style>
<h1>ITM 当前实验动作复盘</h1><p>六个动作 × head／wrists，覆盖全部已有开发来源。打开后可播放、拖时间轴及旋转视角，所有面板同步。</p>
<p>GT 是参考动作；U 是旧模型；F 是损失配重候选。仅文本来自关闭 IMU 的原 MDM 路径。仅 IMU 结果只在已有验证样本中展示。弓步另含去主引导项实验，该实验前 100 步采用仅文本条件，不能视为同一起始条件的标准模块消融。</p>
<p>动作均为已保存的真实完整采样，没有重新生成、时间对齐或缩放。显示时各自根节点 XZ 居中，保留高度，根轨迹可用 Root trails 查看。此页用于讨论效果，开发小面板不代表正式测试集。点击播放即可查看骨架。</p>
<table><thead><tr><th>动作</th><th>来源</th><th>传感器</th><th>时长</th><th>查看</th></tr></thead><tbody>'''
                   + rows + '</tbody></table><p><a href="/">返回 Condition Lab</a></p></html>', encoding='utf-8')
    write(output / 'summary.json', dict(cases=len(catalog), panels=sum(r['panels'] for r in catalog),
                                      source_files=len(sources), new_model_calls=0, new_GPU_queries=0,
                                      new_sampling=0, all_frames=True, coordinate_fitting=False))
    print(read(output / 'summary.json'))


if __name__ == '__main__':
    export()
