#!/usr/bin/env python
"""Render fixed-source GT contact sheets and continuous kinematic traces for review."""
import argparse
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itm.data.canonical_pair import CHAINS
from itm.data.manifest import read_jsonl
from audit_caption_intervals import artifact, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for r in read_jsonl(a.pairs):
        joints=np.load(r['joints_path']).astype(float)
        fig=plt.figure(figsize=(16,8))
        ids=np.unique(np.linspace(0,len(joints)-1,16).round().astype(int))
        span=max(2.,float(np.ptp(joints,axis=(0,1)).max()))/2
        centre=(joints.max((0,1))+joints.min((0,1)))/2
        for n,i in enumerate(ids):
            ax=fig.add_subplot(3,8,n+1,projection='3d')
            for chain,color in zip(CHAINS,['crimson','royalblue','black','crimson','royalblue']):
                v=joints[i,chain]
                ax.plot(v[:,0],v[:,2],v[:,1],color=color,lw=2)
            ax.set(xlim=(centre[0]-span,centre[0]+span),ylim=(centre[2]-span,centre[2]+span),zlim=(0,2*span))
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=15,azim=-60)
            ax.set_title(f'{i/20:.2f}s',fontsize=9); ax.set_axis_off()
        t=np.arange(len(joints))/20
        ax=fig.add_subplot(3,3,7)
        for i,c,label in [(0,'black','pelvis'),(7,'royalblue','L ankle'),(8,'crimson','R ankle')]:
            ax.plot(t,joints[:,i,1],color=c,label=label)
        ax.set(xlabel='seconds',ylabel='height (m)'); ax.legend(fontsize=8)
        ax=fig.add_subplot(3,3,8)
        for i,c,label in [(20,'royalblue','L wrist'),(21,'crimson','R wrist')]:
            ax.plot(t,joints[:,i,1]-joints[:,0,1],color=c,label=label)
        distance=np.linalg.norm(joints[:,20]-joints[:,21],axis=-1)
        ax.plot(t,distance,color='gray',label='hand separation');ax.legend(fontsize=8);ax.set(xlabel='seconds',ylabel='metres')
        ax=fig.add_subplot(3,3,9);ax.plot(joints[:,0,0],joints[:,0,2]);ax.scatter(joints[0,0,0],joints[0,0,2],color='green')
        ax.set(xlabel='root X (m)',ylabel='root Z (m)');ax.axis('equal')
        fig.suptitle(r['motion_id']+' '+r['split']+' | LEFT blue / RIGHT red\n'+textwrap.fill(r['caption'],140),fontsize=11)
        fig.tight_layout(rect=(0,0,1,.9))
        path=a.output/(r['motion_id']+'.png');fig.savefig(path,dpi=110);plt.close(fig)
        reports.append({'motion_id':r['motion_id'],'sheet':artifact(path),'frame_indices':ids.tolist(),
                        'caption_index':r['caption_index'],'caption':r['caption'], 'review_status':'pending',
                        'hand_distance_min_max_m':[float(distance.min()),float(distance.max())],
                        'root_height_range_m':float(np.ptp(joints[:,0,1]))})
    write_json(a.output/'index.json',{'pairs':artifact(a.pairs),'cases':reports,'code':artifact(Path(__file__)),
        'limitation':'GT contact sheets and all-frame traces; objects/fingers/fine timing not certified by a 22-joint skeleton.'})

if __name__=='__main__':main()
