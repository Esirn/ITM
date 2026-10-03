"""GT-only full-frame curves and fixed poses; no model-generated outputs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
p=np.load(ROOT/'outputs/itm_rework/canonical_b0_20260925_v3/train/000506/joints.npy').astype(float)
chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
fig=plt.figure(figsize=(15,8));grid=fig.add_gridspec(3,6,height_ratios=[1.2,1,1])
ax=fig.add_subplot(grid[0,:3]);ax.plot(np.arange(len(p))/20,p[:,0,1]);ax.set(title='GT000506: pelvis height, all136 frames',xlabel='seconds',ylabel='metres (+Y)');ax.grid(alpha=.2)
ax=fig.add_subplot(grid[0,3:]);ax.plot(np.arange(len(p))/20,p[:,7,2]-p[:,0,2],label='left ankle');ax.plot(np.arange(len(p))/20,p[:,8,2]-p[:,0,2],label='right ankle');ax.legend();ax.set(title='Ankle forward displacement relative to pelvis',xlabel='seconds',ylabel='metres (+Z)');ax.grid(alpha=.2)
for i,t in enumerate([0,10,22,35,43,55,67,80,95,110,123,135]):
 ax=fig.add_subplot(grid[1+i//6,i%6]);q=p[t].copy();q[:,2]-=q[0,2]
 for j,c in enumerate(chains):ax.plot(q[c,2],q[c,1],'.-',color=['#e85a4f','#3572bd','#444444','#e85a4f','#3572bd'][j],lw=1.4,markersize=3)
 ax.set(xlim=(-1,1),ylim=(0,2),title=f'frame {t}');ax.set_aspect('equal');ax.grid(alpha=.15)
fig.suptitle('GT only: blue left / red right; side view +Z forward. No fitted alignment.',fontsize=14)
fig.tight_layout();fig.savefig(Path(__file__).with_name('gt000506.png'),dpi=130);plt.close(fig)
