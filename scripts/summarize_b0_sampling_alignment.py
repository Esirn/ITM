#!/usr/bin/env python
"""Render saved owner02 metrics and decode completed review notes; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1'
def read(p):return json.loads(p.read_text())
def mean(r,k):return r['metrics'][k]['mean']
def fmt(x):return 'N/A' if x is None else f'{x:.6f}'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();out=a.output
    samples=read(out/'samples.json');c=read(out/'contrasts.json');q=read(out/'probes.json');t=read(out/'traces.json')
    text=['# 固定R guidance完整数值表','', '旧R=2.5/1/1，S1=1/1/1。每格3来源等权；contact仅val000433/000382，n=2，train和val000139为N/A。所有来源min/max与逐条轴向extent/root path在JSON中保留。误差差值为前项减后项；jerk是绝对量，须同GT解释。','']
    keys=['global_MPJPE_m','root_relative_MPJPE_m','root_path_RMSE_m','head_or_wrists_position_joint_proxy_m','velocity_joint_proxy_RMSE_m_s','jerk_absolute_m_s3','GT_jerk_absolute_m_s3','foot_contact_speed_proxy_m_s','GT_foot_contact_speed_proxy_m_s']
    text+=['## 全部40格','', '| arm | split | config | condition | global m | relative m | root m | joint proxy m | velocity m/s | jerk | GT jerk | contact m/s | GT contact |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in samples['aggregates']:
        text.append('| '+' | '.join([r[k] for k in ('arm','data_split','sensor_config','condition')]+[fmt(mean(r,k)) for k in keys])+' |')
    for name,label,group in [('between_aggregates','全部S1−旧R匹配条件','condition'),('within_aggregates','各guidance内paired−comparator','comparator'),('difference_aggregates','(S1 paired−comparator)−(旧R paired−comparator)','comparator')]:
        text+=['',f'## {label}','','| arm | split | config | comparison | Δglobal | Δrelative | Δroot | Δproxy | Δvelocity |','|---|---|---|---|---:|---:|---:|---:|---:|']
        for r in c[name]:text.append('| '+' | '.join([r.get('arm','S1−oldR'),r['data_split'],r['sensor_config'],r[group]]+[fmt(mean(r,k)) for k in keys[:5]])+' |')
    text+=['','## Q：全部条件的原量与分解','','同来源先平均2次noise replicate，随后3来源等权；按t分别报告。legacy为float64概念重组，Q无f01，不是四分支float32轨迹重放。','','| split | config | t | condition | unit MSE | legacy MSE | legacy−unit | cross | energy |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in q['derived_aggregates']:
        text.append('| '+' | '.join([r['data_split'],r['sensor_config'],str(r['timestep']),r['condition']]+[fmt(r['values'][k]['mean']) for k in ['unit_mse','legacy_reduced_mse','legacy_minus_unit','twice_error_text_inner_product','text_energy']])+' |')
    text+=['','## Trace：全部实际S1条件/t的局部文本增量','','不是旧guidance轨迹，没有自由输出因果归因或新target MSE。','','| split | config | condition | t | text increment RMS | min(source RMS) | max(source RMS) |','|---|---|---|---:|---:|---:|---:|']
    for r in t['aggregates']:
        s=r['values']['text_increment_rms'];text.append('| '+' | '.join([r['data_split'],r['sensor_config'],r['condition'],str(r['timestep'])]+[fmt(s[k]) for k in ('mean','min','max')])+' |')
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    note_path=out/'review_notes_hidden.txt'
    notes={l[:4]:l[5:] for l in note_path.read_text().splitlines() if re.match(r'^S\d{3} ',l)}
    rows=read(ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1/review_order.json')['rows']
    assert len(notes)==120 and set(notes)=={r['review_id'] for r in rows}
    payload=dict(notes_sha256_before_key=hashlib.sha256(note_path.read_bytes()).hexdigest(),rows=[dict(r,note=notes[r['review_id']]) for r in rows],limitation='All notes completed before this bulk decoding; prior results and S001 mapping known; single AI coarse assessment, not blind human validation or semantic success rate.')
    (out/'review_unmasked.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('40 sample cells, all contrast/Q/trace tables, 120 review notes')

if __name__=='__main__':main()
