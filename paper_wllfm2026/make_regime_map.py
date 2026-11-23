"""Fig. 1: regime map summarizing Tables III and IV.

Usage: python paper_wllfm2026/make_regime_map.py [output.pdf]

Box placement follows the measured winners per dataset (Tables III and IV);
positions along the axes are qualitative, and the grey regions were not evaluated.
"""
import os, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(6.4,3.0)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
def box(x,y,w,h,c,t1,t2,fs=7.5,tc='black',ec='none'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.25',fc=c,ec=ec,lw=0.8))
    ax.text(x+w/2,y+h*0.62,t1,ha='center',va='center',fontsize=fs,fontweight='bold',color=tc); ax.text(x+w/2,y+h*0.28,t2,ha='center',va='center',fontsize=fs-0.5,style='italic',color=tc)
box(0.9,3.2,2.5,2.4,'#A7BFEF','Specialist optimal','ETTh1 (energy)')
box(3.6,3.2,2.5,2.4,'#FBE5A0','Competitive regime','Exchange (finance)\nM4 Daily (demand)')
box(6.3,3.2,3.4,2.4,'#F2F2F2','Classical baselines optimal','not evaluated',tc='#666666')
box(0.9,0.5,4.4,2.4,'#F2F2F2','Low predictability','not evaluated',tc='#666666')
box(5.5,0.5,4.2,2.4,'#B7D8AC','Generalist (FM) optimal','Traffic (transport)')
ax.plot([6.2,6.2],[0.4,5.7],ls=':',color='#999999',lw=0.8,zorder=0); ax.plot([1.0,9.7],[3.05,3.05],ls=':',color='#999999',lw=0.8)
ax.annotate('',xy=(0.6,5.9),xytext=(0.6,0.2),arrowprops=dict(arrowstyle='->',lw=1)); ax.annotate('',xy=(9.9,0.2),xytext=(0.6,0.2),arrowprops=dict(arrowstyle='->',lw=1))
ax.text(0.3,3.0,'Entropy / physical constraints',rotation=90,ha='center',va='center',fontsize=7.5); ax.text(5.2,-0.15,'Temporal structure (periodicity, human cycles)',ha='center',va='center',fontsize=7.5)
ax.text(0.6,-0.1,'Weak',ha='center',fontsize=7); ax.text(9.8,-0.1,'Strong',ha='center',fontsize=7)
out=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'outputs','matrix_v2.pdf')
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,bbox_inches='tight'); print('Wrote',out)
