#!/usr/bin/env python
"""Freeze then CPU-decompose saved weight/control and state prediction differences."""
import argparse
from pathlib import Path

from evaluate_b0_native_training_sufficiency import read,artifact,verify,arrays,matrix,write,aggregate
from itm.metrics.prefix_attribution import two_component_delta

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'docs/itm_rework/protocol/b0_prefix_mechanism_20261008_v1/package.json'
PACKAGE_SHA='8b32b863da7a530cc71350384663678adfd072f8db53bbb01add176b3040853c'


def freeze(output,run_dir,manifest):
    if output.exists():raise FileExistsError('Fresh CPU decomposition plan required')
    if artifact(PACKAGE)['sha256']!=PACKAGE_SHA:raise ValueError('Original prefix package changed')
    files={Path(a['path']).stem:verify(a) for a in read(PACKAGE)['artifacts']}
    paths=read(files['trajectories']);states=read(files['states']);probes=read(files['probes'])
    tmap={t['trajectory_id']:t for t in paths}
    smap={(tmap[s['trajectory_id']]['panel_case_id'],tmap[s['trajectory_id']]['path_arm'],s['timestep']):s['state_id'] for s in states}
    pmap={(p['state_id'],p['checkpoint']):p['probe_id'] for p in probes}
    jobs=[]
    for case in [t for t in paths if t['path_arm']=='F']:
        for t in [999,975,950,925,900]:
            def ref(path,weight,branch):return dict(probe_id=pmap[(smap[(case['panel_case_id'],path,t)],weight)],branch=branch)
            for comparison,ordering,p,i,r,first in [
                ('F_minus_U','direct_at_F_state',ref('F','F200','f11'),ref('F','U200','f11'),ref('U','U200','f11'),True),
                ('F_minus_U','direct_at_reference_state',ref('F','F200','f11'),ref('U','F200','f11'),ref('U','U200','f11'),False),
                ('F_minus_T','direct_at_F_state',ref('F','F200','f11'),ref('F','F200','f10'),ref('T','F200','f10'),True),
                ('F_minus_T','direct_at_reference_state',ref('F','F200','f11'),ref('T','F200','f11'),ref('T','F200','f10'),False)]:
                jobs.append(dict(row_id=f'PA{len(jobs)+1:03d}',**{k:case[k] for k in ('panel_case_id','data_split','motion_id','sensor_config')},
                    timestep=t,comparison=comparison,ordering=ordering,prediction=p,intermediate=i,reference=r,direct_is_first=first,status='not_run'))
    sources=[PACKAGE,files['trajectories'],files['states'],files['probes'],manifest,Path(__file__).resolve(),
             ROOT/'src/itm/metrics/prefix_attribution.py',ROOT/'src/itm/metrics/training_sufficiency.py',
             ROOT/'src/itm/metrics/feature_balance.py',ROOT/'scripts/evaluate_b0_native_training_sufficiency.py',
             ROOT/'tests/test_prefix_attribution.py']
    output.mkdir(parents=True)
    write(output/'plan.json',dict(version='b0-prefix-saved-prediction-algebra-v1',date='2026-10-08',
        scope='Main-session exploratory CPU follow-up designed after prior partialv2 exposure, before reading completev3 effects; not blind preregistration or a new primary criterion.',
        run_directory=str(run_dir.resolve()),implementation=artifact(manifest),scientific_package=artifact(PACKAGE),
        jobs=jobs,planned_rows=240,source_count=12,timesteps=[999,975,950,925,900],
        formula='prediction-reference=(prediction-intermediate)+(intermediate-reference). Both reference orderings required. '
            'Report raw/centered component energies, signed cross terms and fixed-time GT MSE change; covariance prevents interpreting energies as additive percentages.',
        direct='Checkpoint F-U difference for F_minus_U; paired IMU vs strict-null text difference for F_minus_T, at a fixed saved state.',
        state='Model output change between saved path states with checkpoint/condition fixed. States came from different prior trajectories; not an isolated causal distribution-shift intervention.',
        gate='Complete original queue/all mandatory software anchors required for primary values. Never complete by joining previous partial data.',
        aggregation='Equal source means/min/max/n per split/config/t/comparison/ordering; no pooling times or treating frames as sources.',
        tolerances='Original float64 64eps identity bounds; no new behavioral threshold, checkpoint selection or statistical significance gate.',
        prohibited=['GPU query','model forward','training','noise generation','sampling','decode','alignment/fitting','paper claim change'],
        sources=[artifact(p) for p in sources]))
    print(dict(plan=artifact(output/'plan.json'),planned_rows=len(jobs),new_model_calls=0))


def analyze(plan_path,output):
    if output.exists():raise FileExistsError('Fresh attribution evidence required')
    plan=read(plan_path)
    for a in plan['sources']:verify(a)
    run=Path(plan['run_directory']);status=read(run/'runtime/status.json')
    if status['implementation']!=plan['implementation']:raise ValueError('Planned implementation changed')
    if status['status']!='completed':raise ValueError('Primary decomposition requires complete queue')
    receipt=read(verify(status['receipt']))
    if receipt['status']!='completed' or any(c['status']!='completed' for c in receipt['checks']):raise ValueError('Software gate incomplete')
    for a in receipt['artifacts']:verify(a)
    table={p['id']:p for p in status['probes']};data={};targets={};rows=[]
    for job in plan['jobs']:
        values=[]
        for ref in [job['prediction'],job['intermediate'],job['reference']]:
            key=ref['probe_id'];record=table[key]
            if record['status']!='completed' or record.get('diagnostic_only'):raise ValueError('Uncertified prediction slot')
            if key not in data:data[key]=arrays(verify(record['raw_artifact']))
            values.append(matrix(data[key][ref['branch']]))
        cid=job['panel_case_id']
        if cid not in targets:targets[cid]=matrix(arrays(run/'prefix/inputs'/(cid+'.npz'))['target'])
        metrics=two_component_delta(*values,targets[cid],direct_is_first=job['direct_is_first'])
        rows.append(dict(job,status='valid',metrics=metrics))
    output.mkdir(parents=True)
    write(output/'rows.json',rows)
    write(output/'aggregates.json',aggregate(rows,['data_split','sensor_config','timestep','comparison','ordering'],'metrics'))
    write(output/'summary.json',dict(status='completed',planned=240,valid=len(rows),new_model_calls=0,new_GPU_queries=0,
        interpretation='Algebra of saved conditional predictions; ordering-sensitive components and signed covariance. No unique mechanism, behavior repair or training sufficiency certification.'))
    write(output/'sources.json',[artifact(plan_path),artifact(run/'runtime/status.json'),status['receipt'],status['implementation'],*plan['sources'],*receipt['artifacts']])
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.glob('*.json'))]);print(dict(planned=240,valid=len(rows),new_model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path)
    for n in ('run','implementation','plan','output'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.freeze:freeze(a.freeze.resolve(),a.run.resolve(),a.implementation.resolve())
    else:analyze(a.plan.resolve(),a.output.resolve())
