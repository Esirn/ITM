#!/usr/bin/env python
"""Match the fixed MDM text CFG2.5; reuse already complete reference outputs."""
import argparse
from pathlib import Path

from itm.models.mdm_correction import read, artifact, verify
from run_expanded_canonical_training import write


def freeze(base, prior_run, prior_evaluation, output):
    if output.exists(): raise FileExistsError(output)
    plan = read(base)
    reused, rows = [], []
    for r in plan['rows']:
        if r['condition'] == 'paired':
            rows.append(dict(r, case_id=r['case_id'] + '-text-g2_5', text_scale=2.5))
        else:
            final = prior_run / 'artifacts' / (r['case_id'] + '.npz')
            reused.append(dict(r, reuse_final=artifact(final)))
    if len(rows) != 22 or len(reused) != 16: raise ValueError('Fixed22 new/16 reused population required')
    for r in reused: verify(r['reuse_final'])
    plan.update(rows=rows, reused_baselines=reused, counts=dict(samples=22, validation_samples=16,
        training_anchor_samples=6, mdm_calls=22000, encoder_calls=44, caption_calls=11, decodes=22),
        method='expanded U epoch100, matched text CFG2.5/IMU1/joint1',
        extra_inference_tag='g2_5', base_unit_guidance_plan=artifact(base),
        cached_evaluation=artifact(prior_evaluation / 'rows.json'),
        cached_evaluation_summary=artifact(prior_evaluation / 'summary.json'),
        scope='Development same-weight matched-text-guidance comparison; 22 new full samples,16 prior MDMSamples and128 prior simple outputs reused; no training or new noise groups')
    plan['expected_artifacts'] += [artifact(base), artifact(Path(__file__).resolve()),
        plan['cached_evaluation'], plan['cached_evaluation_summary'],
        *[r['reuse_final'] for r in reused],
        *[artifact(Path(r['reuse_final']['path']).with_suffix('.json')) for r in reused]]
    output.mkdir(parents=True); write(output / 'plan.json', plan); print(plan['counts'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','prior-run','prior-evaluation','output'): p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args(); freeze(a.base.resolve(),a.prior_run.resolve(),a.prior_evaluation.resolve(),a.output.resolve())
