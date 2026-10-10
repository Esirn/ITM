#!/usr/bin/env python
"""Saved-array F/U review sheets. No models, GPU, fitting or semantic scoring.

This complements the evaluator's all-frame interactive pages with fixed-order
image sheets and optional contact sheets of EVERY frame for detailed inspection.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from evaluate_b0_feature_balance import FROZEN, ROOT, read, sha, write
from itm.metrics.feature_balance import curves

CHAINS = [([0, 1, 4, 7, 10], '#1479c4'), ([0, 2, 5, 8, 11], '#d44a38'),
          ([0, 3, 6, 9, 12, 15], '#444444'), ([9, 13, 16, 18, 20], '#1479c4'),
          ([9, 14, 17, 19, 21], '#d44a38')]
ACTIONS = {'000433': 'kick, return to standing', '000382': 'clap, then lower arms',
           '000139': 'stand/sway, then walk', '000506': 'alternating forward lunges; clipped start/end',
           '004728': 'RIGHT leg backward lift, hold, lower', '000027': 'punch, then turn'}


def pose(ax, motion, frame, plane='oblique', gt=False):
    # Display-only root XZ subtraction, independently for each motion; all
    # metrics and global trajectory curves remain in their original coordinates.
    p = motion[frame].copy()
    p[:, [0, 2]] -= p[0, [0, 2]]
    x = p[:, 0] + .55*p[:, 2] if plane == 'oblique' else p[:, 2]
    y = p[:, 1] + .1*p[:, 2] if plane == 'oblique' else p[:, 1]
    for ids, color in CHAINS:
        ax.plot(x[ids], y[ids], color='#bbbbbb' if gt else color,
                lw=.8 if gt else 1.3, alpha=.65 if gt else 1.)
    ax.set(xlim=(-1.5, 1.5), ylim=(-.35, 2.3))
    ax.set_aspect('equal')
    ax.axis('off')


def signals(ax, p, g, kind):
    t = np.arange(len(g))/20
    for x, linestyle, alpha in [(g, '--', .45), (p, '-', 1.)]:
        if x is None:
            continue
        c = curves(x)
        if kind == 'height':
            series = [(x[:, 0, 1], 'pelvis', 'black'), (x[:, 7, 1], 'L ankle', '#1479c4'), (x[:, 8, 1], 'R ankle', '#d44a38')]
        elif kind == 'legZ':
            series = [(x[:, 7, 2]-x[:, 0, 2], 'L ankle', '#1479c4'), (x[:, 8, 2]-x[:, 0, 2], 'R ankle', '#d44a38')]
        elif kind == 'knees':
            series = [(c['left_knee_degrees'], 'L knee', '#1479c4'), (c['right_knee_degrees'], 'R knee', '#d44a38')]
        else:
            series = [(np.linalg.norm(x[:, 20]-x[:, 21], axis=-1), 'wrist sep', 'black'),
                      (x[:, 20, 1]-x[:, 0, 1], 'L wrist-Y', '#1479c4'),
                      (x[:, 21, 1]-x[:, 0, 1], 'R wrist-Y', '#d44a38')]
        for y, label, color in series:
            ax.plot(t, y, linestyle=linestyle, color=color, alpha=alpha,
                    label=label if x is p else None, lw=.8)
    ax.set_title(dict(height='Height (m)', legZ='Ankle minus pelvis Z (m)',
                      knees='Knee angle (deg)', hands='Hands (m)')[kind], fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=.2)


def render(run, output, detail_ids=()):
    output.mkdir(parents=True, exist_ok=False)
    order = read(FROZEN / 'review_order.json')['rows']
    slots = read(FROZEN / 'sampling_slots.json')
    specs = {cid: s for s in slots for cid in (s['case_id'], s['historical_case_id'])}
    sources, loaded = {}, {}
    for row in order:
        s = specs[row['case_id']]
        path = run / 'samples' / (s['case_id']+'.npz') if row['arm'] == 'F' else Path(s['historical_output']['path'])
        gtpath = Path(s['gt_path'])
        g = np.load(gtpath, allow_pickle=False)
        sources[str(gtpath)] = sha(gtpath)
        p = None
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                p = z['motion'][0].copy()
            if p.shape != g.shape or not np.isfinite(p).all():
                raise ValueError('Invalid saved full sequence: '+str(path))
            sources[str(path)] = sha(path)
        loaded[row['review_id']] = (p, g, ACTIONS[row['motion_id']])
    artifacts = []
    for page in range(20):
        fig = plt.figure(figsize=(26, 15))
        gs = GridSpec(6, 14, figure=fig, hspace=.7, wspace=.65)
        for y, row in enumerate(order[page*6:page*6+6]):
            p, g, target = loaded[row['review_id']]
            for x, frame in enumerate(np.linspace(0, len(g)-1, 10).round().astype(int)):
                ax = fig.add_subplot(gs[y, x])
                pose(ax, g, frame, gt=True)
                if p is not None: pose(ax, p, frame)
                ax.set_title(f'{frame/20:.2f}s', fontsize=7)
                if x == 0:
                    ax.text(0, 1.28, row['review_id']+' | '+target+(' | MISSING' if p is None else ''),
                            transform=ax.transAxes, fontsize=10, weight='bold')
            for x, kind in enumerate(('height', 'legZ', 'knees', 'hands'), 10):
                signals(fig.add_subplot(gs[y, x]), p, g, kind)
        fig.suptitle(f'Fixed review order {page+1}/20 | left blue / right red | gray poses + dashed curves: GT | '
                     'root XZ centered for display only; no metric fitting', fontsize=10)
        path = output / f'page_{page+1:02}.png'
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        artifacts.append(dict(path=str(path), sha256=sha(path)))
        print('rendered', path.name, flush=True)
    for rid in detail_ids:
        p, g, target = loaded[rid]
        if p is None: continue
        for plane in ('oblique', 'ZY'):
            nrows = (len(p)+11)//12
            fig, axes = plt.subplots(nrows, 12, figsize=(24, 2*nrows))
            for frame, ax in enumerate(axes.flat):
                if frame >= len(p): ax.axis('off'); continue
                pose(ax, g, frame, plane, gt=True)
                pose(ax, p, frame, plane)
                ax.set_title(str(frame), fontsize=7)
            fig.suptitle(rid+' | '+target+' | EVERY frame | '+plane+' | L blue / R red; GT gray', fontsize=12)
            path = output / f'{rid}_all_frames_{plane}.png'
            fig.savefig(path, dpi=95, bbox_inches='tight')
            plt.close(fig)
            artifacts.append(dict(path=str(path), sha256=sha(path)))
    write(output / 'rendering.json', dict(order_sha256=sha(FROZEN/'review_order.json'),
        renderer=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        sources=[dict(path=p, sha256=h) for p, h in sorted(sources.items())], images=artifacts,
        scope='Fixed hidden arm/config/condition order; snapshots and all-frame curves. Historical U known; '
              'not fully blind or independent human review. No semantic labels assigned by renderer.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detail-ids', nargs='*', default=[])
    args = parser.parse_args()
    render(args.run.resolve(), args.output.resolve(), args.detail_ids)
