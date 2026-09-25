#!/usr/bin/env python
"""Supplementary continuous bone/semantic diagnostics; no admission threshold fitting."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_caption_intervals import artifact, write_json
from itm.data.canonical_pair import CHAINS
from itm.data.manifest import read_jsonl


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--pairs',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=q.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    raw=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,-1,0],[0,1,0],[0,-1,0],[0,-1,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[1,0,0],[-1,0,0],[0,0,1],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0]])
    results=[]
    for r in read_jsonl(a.pairs):
        f=np.load(r['joint_vec_path']);p=np.load(r['joints_path']);t=np.arange(len(f))/20.
        six=f[:,67:193].reshape(len(f),21,6).astype(float)
        x=six[...,:3];x/=np.linalg.norm(x,axis=-1,keepdims=True)
        z=np.cross(x,six[...,3:]);z/=np.linalg.norm(z,axis=-1,keepdims=True);y=np.cross(z,x)
        rotations=np.stack([x,y,z],axis=-1)
        yaw=np.r_[0,np.cumsum(f[:-1,0].astype(float))]*2
        root=Rotation.from_rotvec(np.column_stack([yaw*0,yaw,yaw*0])).as_matrix()
        errors=[]
        for chain in CHAINS:
            current=root
            for first,second in zip(chain[:-1],chain[1:]):
                current=current@rotations[:,second-1]
                direction=current@raw[second]
                bone=p[:,second]-p[:,first];bone/=np.linalg.norm(bone,axis=-1,keepdims=True)
                errors.append(float(np.max(np.abs(direction-bone))))
        left=p[:,1]-p[:,2]+p[:,16]-p[:,17];left/=np.linalg.norm(left,axis=-1,keepdims=True)
        projection=np.sum((p[:,21]-p[:,0])*left,axis=-1)
        speed=np.linalg.norm(np.diff(p[:,0][:,[0,2]],axis=0),axis=-1)*20
        result={'motion_id':r['motion_id'],'max_bone_direction_component_error':max(errors),
            'root_height_change_m':float(p[-1,0,1]-p[0,0,1]),
            'right_wrist_anatomical_left_projection_m':projection.tolist(),
            'root_horizontal_speed_m_s':speed.tolist(),
            'half_second_mean_horizontal_speed_m_s':[float(np.mean(speed[i:i+10])) for i in range(0,len(speed),10)],
            'wrist_separation_m':np.linalg.norm(p[:,20]-p[:,21],axis=-1).tolist()}
        if r['motion_id']=='000510':result['hold_left_projection_min_median_max_m']=np.quantile(projection[(t>=.8)&(t<=3.5)],[0,.5,1]).tolist()
        results.append(result)
    write_json(a.output,{'pairs':artifact(a.pairs),'code':artifact(Path(__file__)),'cases':results,
        'purpose':'Continuous supplemental review evidence, not a predeclared new acceptance threshold or generated IMU certification.'})

if __name__=='__main__':main()
