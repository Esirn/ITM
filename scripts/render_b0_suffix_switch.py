#!/usr/bin/env python
"""All-frame CPU review of saved suffix motions; exposed labels, no model calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_b0_native_training_sufficiency import read,verify,artifact,arrays,write
from render_b0_feature_balance_review import pose,signals,CHAINS


def render(results,output):
    if output.exists():raise FileExistsError('Fresh suffix review required')
    for a in read(results/'artifact_index.json'):verify(a)
    finals=read(results/'finals.json');old=read(results/'old_final_references.json')
    root=Path(__file__).resolve().parents[1]
    slots=read(root/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')
    output.mkdir(parents=True);payload=[];origins=[]
    for config in ('head','wrists'):
        slot=next(s for s in slots if s['motion_id']=='000506' and s['data_split']=='train' and s['sensor_config']==config and s['condition']=='paired')
        g=np.load(slot['gt_path'],allow_pickle=False);origins.append(artifact(slot['gt_path']))
        pop=[]
        for r in finals:
            if r['sensor_config']!=config:continue
            if 'raw' not in r:continue
            p=arrays(verify(r['raw']))['joints'][0];pop.append((r['trajectory_id']+' '+r['condition']+' '+r['status'],p))
        for arm,condition in [('U','paired'),('F','text_only')]:
            case=next(s for s in slots if s['motion_id']=='000506' and s['data_split']=='train' and s['sensor_config']==config and s['condition']==condition)
            path=root/'outputs/itm_rework/method'/('sampling_alignment_authorized_20260927_v1' if arm=='U' else 'feature_balance_authorized_20261008_v1')/'samples'/((case['historical_case_id'] if arm=='U' else case['case_id'])+'.npz')
            p=arrays(path)['motion'][0];origins.append(artifact(path));pop.append(('Original '+arm+' '+condition,p))
        payload.append(dict(sensor_config=config,gt=g.tolist(),motions=[dict(label=label,joints=p.tolist()) for label,p in pop]))
        fig,axes=plt.subplots(len(pop),12,figsize=(25,3*len(pop)))
        for row,(label,p) in enumerate(pop):
            for col,frame in enumerate(np.linspace(0,len(g)-1,8).round().astype(int)):
                ax=axes[row,col];pose(ax,g,frame,gt=True);pose(ax,p,frame);ax.set_title(f'{frame/20:.2f}s',fontsize=8)
                if col==0:ax.text(0,1.25,label,transform=ax.transAxes,fontsize=8)
            for col,kind in enumerate(('height','legZ','knees','hands'),8):signals(axes[row,col],p,g,kind)
        fig.suptitle(config+' / same-history suffix and OLD different-history references; GT gray/dashed; no metric alignment',fontsize=11)
        fig.tight_layout();fig.savefig(output/(config+'_overview.png'),dpi=100);plt.close(fig)
        for r in finals:
            if r['sensor_config']!=config or 'raw' not in r:continue
            p=arrays(verify(r['raw']))['joints'][0]
            for start in range(0,len(p),48):
                fig,axes=plt.subplots(6,8,figsize=(20,15))
                for offset,ax in enumerate(axes.flat):
                    frame=start+offset
                    if frame>=len(p):ax.axis('off');continue
                    pose(ax,g,frame,gt=True);pose(ax,p,frame);ax.set_title(str(frame),fontsize=9)
                fig.suptitle(r['trajectory_id']+' '+r['condition']+' / '+r['status']+' / EVERY frame / GT gray, L blue R red',fontsize=12)
                fig.tight_layout();fig.savefig(output/(r['trajectory_id']+f'_frames_{start:03d}.png'),dpi=100);plt.close(fig)
    data=json.dumps(payload,separators=(',',':'))
    html='''<!doctype html><meta charset="utf-8"><title>Same-history suffix review</title><style>body{font-family:sans-serif;margin:24px}canvas{background:#fafafa;margin:8px}label{display:block}input{width:90%}</style><h1>Same-history suffix all-frame review</h1><p>Labels exposed; main-session rough review. GT gray. Display root XZ centered per motion; metrics use original coordinates.</p><label>Frame <input id="frame" type="range" min="0" max="135" value="0"><span id="value"></span></label><label>Projection <select id="plane"><option>oblique</option><option>ZY</option></select></label><div id="panels"></div><script>const data=DATA;const chains=CHAINS;const panels=document.getElementById('panels');const items=[];for(const row of data){const h=document.createElement('h2');h.textContent=row.sensor_config;panels.append(h);for(const m of row.motions){const d=document.createElement('div'),label=document.createElement('div'),c=document.createElement('canvas');label.textContent=m.label;c.width=320;c.height=300;d.style.display='inline-block';d.append(label,c);panels.append(d);items.push({c,m,g:row.gt});}}function draw(){const f=+document.getElementById('frame').value;document.getElementById('value').textContent=f+' / '+(f/20).toFixed(2)+' s';for(const item of items){const ctx=item.c.getContext('2d');ctx.clearRect(0,0,320,300);for(const [p,gt]of[[item.g,true],[item.m.joints,false]]){const q=p[f],root=q[0];for(const [ids,color]of chains){ctx.strokeStyle=gt?'#bbb':color;ctx.lineWidth=gt?1:2;ctx.beginPath();for(let j=0;j<ids.length;j++){const v=q[ids[j]],z=v[2]-root[2],x=document.getElementById('plane').value==='ZY'?z:v[0]-root[0]+.55*z,y=v[1]+.1*z;const a=160+100*x,b=270-100*y;j?ctx.lineTo(a,b):ctx.moveTo(a,b);}ctx.stroke();}}}}document.getElementById('frame').oninput=draw;document.getElementById('plane').onchange=draw;draw();</script>'''
    html=html.replace('DATA',data).replace('CHAINS',json.dumps(CHAINS));(output/'review.html').write_text(html)
    write(output/'rendering.json',dict(results=artifact(results/'artifact_index.json'),original_references=origins,renderer=artifact(Path(__file__).resolve()),
        all_frames=True,independent_review=False,blind=False,new_model_calls=0,new_GPU_queries=0,new_decodes=0))
    write(output/'artifact_index.json',[artifact(p) for p in sorted(output.iterdir())]);print(dict(files=len(read(output/'artifact_index.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.results.resolve(),a.output.resolve())
