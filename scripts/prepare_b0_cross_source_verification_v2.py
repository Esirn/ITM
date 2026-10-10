#!/usr/bin/env python
"""Bind a Python3.10-compatible verifier; reuse untouched complete v1 CPU mathematics."""
from pathlib import Path
from evaluate_b0_native_training_sufficiency import read,verify,artifact,write

ROOT=Path(__file__).resolve().parents[1]

def run():
    old=ROOT/'outputs/itm_rework/method/cross_source_component_implementation_20261009_v1'
    results=ROOT/'outputs/itm_rework/protocol/cross_source_component_evaluation_20261009_v1'
    output=ROOT/'outputs/itm_rework/method/cross_source_component_implementation_20261009_v2'
    if output.exists():raise FileExistsError('Fresh verification-only manifest required')
    manifest=read(old/'implementation.json');sources={a['path']:a for a in manifest['expected_artifacts']}
    for a in sources.values():verify(a)
    for a in read(results/'artifact_index.json'):verify(a)
    summary=read(results/'summary.json')
    if summary['status']!='completed' or summary['row_counts']!={'valid':480} or summary['pair_counts']!={'valid':240}:raise ValueError('Complete unchanged CPU mathematics required')
    paths=[old/'implementation.json',old/'first_verification_failure.json',results/'artifact_index.json',Path(__file__).resolve(),ROOT/'scripts/verify_b0_cross_source_components_v2.py']
    paths.extend(results/p for p in ['rows.json','pairs.json','aggregates.json','sources.json','summary.json','old_final_references.json'])
    for p in paths:a=artifact(p);sources[a['path']]=a
    output.mkdir(parents=True)
    write(output/'implementation.json',dict(version='itm-cross-source-component-verification-v2',plan=manifest['plan'],inherited_implementation=artifact(old/'implementation.json'),inherited_results_index=artifact(results/'artifact_index.json'),expected_artifacts=[sources[k] for k in sorted(sources)],rows=480,F_U_pairs=240,reason='Verification-only Python3.10 syntax compatibility; original adapter/results/code and tolerances preserved. No repeated old CPU evaluation.',new_GPU_queries=0,new_model_calls=0,new_noise_draws=0,new_decodes=0))
    print(dict(implementation=artifact(output/'implementation.json'),sources=len(sources)))

if __name__=='__main__':run()
