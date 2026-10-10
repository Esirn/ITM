#!/usr/bin/env python
"""Standalone development comparison figure, no animation or model calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(evaluation,output):
    if output.exists():raise FileExistsError(output)
    rows=json.loads((evaluation/'aggregates.json').read_text())
    lookup={r['arm']:r['metrics'] for r in rows if r['population']=='matched_val8'}
    arms=['MDM_text_g2.5','original_U100','head_supervised_U','simple_text_imu','simple_imu_only']
    labels=['Text MDM','Original U','Head loss U','Simple text+IMU','Simple IMU']
    colors=['#8a8a8a','#4477aa','#228833','#ccbb44','#aa3377']
    panels=[('observed_relative_position_mean_m',100,'Head relative position error (cm)'),
            ('observed_relative_velocity_RMSE_m_s',1,'Head relative velocity error (m/s)'),
            ('root_relative_MPJPE_m',100,'Full-body relative position error (cm)'),
            ('observed_relative_jerk_mean_m_s3',1,'Head relative jerk (m/s³)')]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(key,factor,title) in zip(axes.flat,panels):
        values=[lookup[a][key]*factor for a in arms]
        ax.bar(np.arange(5),values,color=colors,width=.65)
        ax.set_xticks(np.arange(5),labels,rotation=15,ha='right')
        ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        if key=='observed_relative_jerk_mean_m_s3':
            ax.axhline(lookup[arms[0]]['GT_observed_relative_jerk_mean_m_s3'],color='#333',linestyle='--',label='GT mean')
            ax.legend(frameon=False)
    fig.suptitle('Head IMU: matched 8 development sequences, one seed / K=1\nLower errors are better; low jerk alone does not establish naturalness',fontsize=12)
    output.mkdir(parents=True)
    fig.savefig(output/'head_comparison.png',dpi=180)
    fig.savefig(output/'head_comparison.pdf')
    plt.close(fig)
    print(output/'head_comparison.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.evaluation.resolve(),a.output.resolve())
