#!/usr/bin/env python
"""Deterministic tables for already evaluated V/Q arrays; no inference."""
import argparse
from pathlib import Path
from freeze_b0_state_diagnostic import read, write
from evaluate_b0_state_diagnostic import OUT


def build(directory):
    directory=Path(directory)
    comparisons=read(directory/'V_Q_comparison.json')
    source=read(directory/'V_source_means.json')
    aggregate=read(directory/'V_aggregates.json')
    qa=read(directory/'Q_aggregates.json')
    blocks=['all','root','ric','rotation6d','local_velocity','contact','ric_legs','ric_trunk_head','ric_arms']
    text=['# V/Q完整来源与固定分区表','',
        '全部为预训练归一化HumanML263特征MSE，负差值表示前项误差更低；不是物理单位或成功率。Q先平均两次噪声，V只有一条paired轨迹。四个t相关，不跨t合并。','',
        '## 主要比较：逐split/config/t的3来源等权均值与范围','',
        '| split | config | t | 比较 | V均值 | V最小 | V最大 | Q均值 | n/计划 |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in aggregate:
        if r['contrast'] not in ('P-S','P-R'):continue
        q=next(s for s in qa if all(s[k]==r[k] for k in ('data_split','sensor_config','timestep','contrast')))
        a=r['blocks']['all']['delta_mse'];b=q['blocks']['all']['delta_mse']['mean']
        text.append(f"| {r['data_split']} | {r['sensor_config']} | {r['timestep']} | {r['contrast']} | {a['mean']:+.8f} | {a['min']:+.8f} | {a['max']:+.8f} | {b:+.8f} | 3/3 |")
    text+=['','## 全部48个状态的整体误差比较','',
        '| split | source | config | t | V P−S | V P−R | V R−S | Q P−S | Q P−R | Q R−S |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in ('train','val'):
        for mid in sorted({r['motion_id'] for r in source if r['data_split']==split}):
            for config in ('head','wrists'):
                for t in (900,500,250,50):
                    rows=[next(r for r in comparisons if (r['data_split'],r['motion_id'],r['sensor_config'],r['timestep'],r['contrast'])==(split,mid,config,t,c)) for c in ('P-S','P-R','R-S')]
                    values=[r['blocks']['all'][family]['delta_mse'] for family in ('V','Q') for r in rows]
                    text.append('| '+' | '.join([split,mid,config,str(t),*(f'{x:+.8f}' for x in values)])+' |')
    text+=['','P−T与P−S在本次全部48状态的预测及指标一致（S=T）；完整192项差值仍单独保存。','',
        '## 全部来源t50分区差值与全263维加权贡献','',
        '主五块权重加和还原整体；RIC三个子块不能再与主RIC重复累加。全部其他t的9块结果见JSON。','',
        '| split | source | config | 对比 | block | V块MSE差 | V全263维贡献 | Q块MSE差 | V−Q块差 |',
        '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in comparisons:
        if r['timestep']!=50 or r['contrast']=='P-T':continue
        for b in blocks:
            a=r['blocks'][b]
            text.append('| '+' | '.join([r['data_split'],r['motion_id'],r['sensor_config'],r['contrast'],b,
                f"{a['V']['delta_mse']:+.8f}",f"{a['V']['weighted_delta']:+.8f}",f"{a['Q']['delta_mse']:+.8f}",f"{a['V_minus_Q_delta_mse']:+.8f}"])+' |')
    text+=['','完整预测量级/残差RMS/max/count见branches.json与V_contrasts.json；Q/V差只描述所选状态，不识别因果。所有行完成，无缺失。']
    (directory/'tables.md').write_text('\n'.join(text)+'\n')
    def select(mid,config,c):
        return sorted([r for r in source if (r['motion_id'],r['sensor_config'],r['contrast'])==(mid,config,c)],key=lambda r:-r['timestep'])
    summary=dict(
        train_all_sources_both_configs_each_t_P_better_than_S_and_R=all(r['blocks']['all']['delta_mse']<0 for r in source if r['data_split']=='train' and r['contrast'] in ('P-S','P-R')),
        val_head_each_source_each_t_P_worse_than_R=all(r['blocks']['all']['delta_mse']>0 for r in source if r['data_split']=='val' and r['sensor_config']=='head' and r['contrast']=='P-R'),
        clap_wrists_each_t_P_and_R_worse_than_S=all(r['blocks']['all']['delta_mse']>0 for c in ('P-S','R-S') for r in select('000382','wrists',c)),
        emphasis='Descriptive predicates over complete correlated states, not success rates or equivalence/causal/generalization claims.',
        next_owner='01: propose one evidence-based minimal correction and paired full-action validation budget;02 freezes any new experiment. No new authorization implied.',
        model_effectiveness_certified=False,new_motion_results=0)
    assert summary['train_all_sources_both_configs_each_t_P_better_than_S_and_R']
    assert summary['val_head_each_source_each_t_P_worse_than_R']
    assert summary['clap_wrists_each_t_P_and_R_worse_than_S']
    write(directory/'findings.json',summary)
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=OUT);args=p.parse_args();build(args.directory)
