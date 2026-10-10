#!/usr/bin/env python
"""Expose the completed expanded comparison in the existing uvicorn frontend."""
import argparse
import html
from pathlib import Path

import numpy as np

from itm.models.mdm_correction import read, verify, artifact
from run_expanded_canonical_training import ROOT, write


def export(training_plan, training_run, sampling_plan, sampling_run, evaluation, output):
    if output.exists(): raise FileExistsError(output)
    if read(evaluation / 'summary.json')['missing'] != 0: raise ValueError('Complete evaluation required')
    tp, sp = read(training_plan), read(sampling_plan)
    epochs = tp['epochs']
    guidance_tag = sp.get('extra_inference_tag', '')
    dataset = read(verify(tp['dataset'])); records = {r['motion_id']: r for r in dataset['records']}
    slots = read(ROOT / 'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')
    old_sources = {a['path']:a['sha256'] for a in read(ROOT / 'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2/sources.json')}
    grouped = {}
    for r in sp['rows'] + sp.get('reused_baselines', []): grouped.setdefault((r['split'], r['motion_id'], r['sensor_config']), []).append(r)
    output.mkdir(parents=True); catalog = []
    for (split, mid, sensor), group in grouped.items():
        r = records[mid]
        with np.load(verify(r['input']), allow_pickle=False) as z:
            gt, acceleration = z['gt'].copy(), z['acceleration'].copy()
        panels = []
        def panel(label, p, kind, text=True, imu=True, source=None):
            if p.shape != (r['frames'], 22, 3) or not np.isfinite(p).all(): raise ValueError('Motion shape/value mismatch')
            panels.append(dict(label=label, kind=kind, caption=r['text'], text_source='A' if text else None,
                               imu_source='A' if imu else None, motion=p.tolist(), saved_source=source))
        panel('GT · 参考动作', gt, 'ground_truth', source=str(verify(r['input'])))
        if epochs != 20:
            reference = ROOT / 'outputs/itm_rework/method/expanded_canonical_sampling_authorized_20261009_v1/artifacts' / f'{split}-{mid}-{sensor}-paired.npz'
            with np.load(reference, allow_pickle=False) as z: p = z['motion'][0]
            panel('扩展 U · 20epoch（同采样噪声）', p, 'generated', source=str(reference))
        if guidance_tag:
            reference = ROOT / 'outputs/itm_rework/method/expanded_canonical_sampling_authorized_20261009_v2/artifacts' / f'{split}-{mid}-{sensor}-paired.npz'
            with np.load(reference, allow_pickle=False) as z: p = z['motion'][0]
            panel('扩展 U · 100epoch text g=1（同权重／噪声）', p, 'generated', source=str(reference))
        for row in group:
            path = verify(row['reuse_final']) if row.get('reuse_final') else sampling_run / 'artifacts' / (row['case_id'] + '.npz')
            with np.load(path, allow_pickle=False) as z: p = z['motion'][0]
            panel(f'扩展 U · {epochs}epoch 文本＋IMU（text g={row["text_scale"]}）' if row['condition'] == 'paired' else 'MDM · 仅文本 g=2.5',
                  p, 'generated', imu=row['imu_present'], source=str(path))
        if split == 'val':
            for arm, label in [('simple_text_imu', '简单时序融合 · 文本＋IMU'), ('simple_imu_only', '简单时序网络 · 仅IMU')]:
                path = training_run / 'artifacts/validation' / arm / (mid + '-' + sensor + '.npz')
                with np.load(path, allow_pickle=False) as z: p = z['motion'][0]
                panel(label, p, 'generated', text=arm != 'simple_imu_only', source=str(path))
        else:
            s = next(s for s in slots if s['motion_id'] == mid and s['sensor_config'] == sensor and s['condition'] == 'paired')
            for arm, label in [('U', '原 U · 文本＋IMU（3训练来源）'), ('F', '原 F · 文本＋IMU（3训练来源）')]:
                path = Path(s['historical_output']['path']) if arm == 'U' else ROOT / 'outputs/itm_rework/method/feature_balance_authorized_20261008_v1/samples' / (s['case_id'] + '.npz')
                verify(dict(path=str(path), sha256=old_sources[str(path)]))
                with np.load(path, allow_pickle=False) as z: p = z['motion'][0]
                panel(label, p, 'generated', source=str(path))
        epoch_tag = '' if epochs == 20 else f'{epochs}ep-'
        extra_tag = '' if not guidance_tag else guidance_tag + '-'
        run_id = f'expanded-{epoch_tag}{extra_tag}20261009-v1-{split}-{mid}-{sensor}'
        directory = ROOT / 'outputs/demo_runs' / run_id
        directory.mkdir(parents=True, exist_ok=False)
        request = dict(mode='saved_review', experiment='扩展训练对比', split=split, sample_a=mid,
                       sensor_config=sensor, seed=sp['seed'], text_scale=1., imu_scale=1.)
        result = dict(run_id=run_id, request=request, samples={'A':dict(motion_id=mid, caption=r['text'])},
                      fps=20, frame_count=r['frames'], panels=panels,
                      imu={f'A slot{slot}': dict(acceleration=acceleration[:, slot].tolist()) for slot in ([4] if sensor == 'head' else [0, 1])},
                      comparison_scope='Development K=1, one seed. Baseline CFG differs as marked. Training anchors are not validation evidence.')
        write(directory / 'request.json', request); write(directory / 'result.json', result)
        catalog.append(dict(run_id=run_id, motion_id=mid, split=split, sensor_config=sensor, frames=r['frames'],
                            caption=r['text'], panels=len(panels), url='/?run=' + run_id, result=artifact(directory / 'result.json')))
    write(output / 'catalog.json', catalog)
    rows = ''.join(f'<tr><td>{r["split"]}</td><td>{r["motion_id"]}</td><td>{r["sensor_config"]}</td>'
                   f'<td>{html.escape(r["caption"])}</td><td><a href="{r["url"]}">播放对比</a></td></tr>' for r in catalog)
    hub = ROOT / 'src/itm/demo/static' / ('expanded-review.html' if epochs == 20 else f'expanded-{epochs}ep{("-"+guidance_tag) if guidance_tag else ""}-review.html')
    if hub.exists(): raise FileExistsError(hub)
    hub.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ITM 扩展训练对比</title><style>body{max-width:1200px;margin:35px auto;padding:0 20px;font:16px/1.6 system-ui;background:#f5f7fa;color:#17212e}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#1269bd}</style>
<h1>ITM 扩展训练对比</h1><p>128训练来源、32验证来源；本页完整生成比较固定8个验证来源及3个训练参考来源，均包含head／wrists。新U和简单基线均训练20个epoch。开发K=1，尚未进行正式测试或多种子评估。</p>
<p>验证组：GT、新U文本＋IMU、MDM仅文本（g=2.5）、简单文本＋IMU、简单仅IMU。train组：GT、新U、原U、原F；train仅用于观察熟悉例子，不能代表泛化性能。旧U/F与新U的训练规模及归一化不同，不是单变量模块消融。</p>
<p>骨架可同步播放、拖时间轴、旋转。显示各自根XZ居中，保留高度；Root trails显示位移。所有动作来自完整真实采样或实际基线预测，没有挑最佳噪声或对齐拟合。</p>
<table><thead><tr><th>来源</th><th>ID</th><th>传感器</th><th>原文本</th><th>查看</th></tr></thead><tbody>'''
        .replace('20个epoch', f'{epochs}个epoch')
        + rows + '</tbody></table><p><a href="/static/rework-review.html">原六动作复盘</a> · <a href="/static/expanded-review.html">20epoch对比</a> · <a href="/">Condition Lab</a></p></html>', encoding='utf-8')
    write(output / 'summary.json', dict(cases=len(catalog), panels=sum(r['panels'] for r in catalog), new_model_calls=0, new_sampling=0))
    print(read(output / 'summary.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-plan', 'training-run', 'sampling-plan', 'sampling-run', 'evaluation', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); export(a.training_plan.resolve(), a.training_run.resolve(), a.sampling_plan.resolve(), a.sampling_run.resolve(), a.evaluation.resolve(), a.output.resolve())
