"""Publication plotting from summary CSVs; no LaTeX or external project paths."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from plot_style import key,LABELS,COLORS,MARKERS
ROOT=Path(__file__).resolve().parent

def styled_axis(ax,label,lim=None):
 ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_axisbelow(True)
 if lim is not None:ax.set_ylim(*lim)

def style_lines(fig):
 for ax in fig.axes:
  for line in ax.lines:
   if line.get_transform()==ax.get_yaxis_transform():
    line.set_color('#B43C4A');line.set_linewidth(1.05);line.set_linestyle('--')
   elif line.get_transform()==ax.get_xaxis_transform():
    line.set_color('#555555');line.set_linewidth(.8);line.set_linestyle(':')
   else:
    c=to_hex(line.get_color()).lower()
    if c==COLORS['cglcp'].lower() and line.get_linestyle() not in ['None','none','']:line.set_linewidth(line.get_linewidth()*1.23)
    if c in [COLORS[k].lower() for k in ['rho','alpha','safe']] and line.get_linestyle() not in ['None','none','']:line.set_linestyle('--')

def emit(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(path);fig.savefig(path.with_suffix('.png'),dpi=150);plt.close(fig)

def plot_local(summary,output):
 d=pd.read_csv(summary)
 m7=['pooled_cp','global_rate_plugin','graph_lcp','pcs_sc_corrected','localized_pcs_sc_corrected','cglcp','oracle_local_rho']
 fig,axs=plt.subplots(2,2,figsize=(6.4,5.5),sharex=True)
 for ax,(metric,label,lim) in zip(axs.flat,[('coverage','Total coverage',(.76,1.02)),('finite_conditional_coverage','Coverage | finite',(.76,1.02)),('median_finite_width','Width | finite',None),('finite_rate','Finite rate',(-.02,1.03))]):
  for m in m7:
   g=d[d.method==m].sort_values('local_contamination');k=key(m);x=g.local_contamination.to_numpy();y=g[metric].to_numpy()
   ax.plot(x,y,color=COLORS[k],marker=MARKERS[k],ms=4,lw=1.3,label=LABELS[k])
   if metric!='median_finite_width':ax.errorbar(x,y,yerr=np.maximum(0,[y-g[metric+'_ci95_lower'],g[metric+'_ci95_upper']-y]),fmt='none',ecolor=COLORS[k],capsize=1.5,lw=.7)
  styled_axis(ax,label,lim);ax.axvline(.1,ls=':',color='black',lw=.8);ax.set_xticks([.1,.2,.3,.4]);ax.tick_params(labelsize=10)
  if 'coverage' in metric:ax.axhline(.9,ls='--',color='black',lw=.8)
 for ax in axs[1]:ax.set_xlabel(r'Local contamination $B/n$')
 fig.subplots_adjust(left=.125,right=.98,bottom=.12,top=.96,wspace=.45,hspace=.22)
 style_lines(fig)
 h,l=axs[1,1].get_legend_handles_labels();order=[0,1,2,3,4,6,5]
 axs[1,1].legend([h[i] for i in order],[l[i] for i in order],loc='center',bbox_to_anchor=(.63,.47),fontsize=8.1,frameon=True,facecolor='white',edgecolor='none',framealpha=.95,labelspacing=.35,handlelength=1.3,handletextpad=.45,borderpad=.3)
 emit(fig,output/'local_necessity.pdf')

def plot_robustness(summary,output):
 s3=pd.read_csv(summary)
 methods=['local_uncorrected','global_rate_plugin','cglcp_certified','oracle_local_alpha','oracle_certificate_budget','oracle_safe'];order=['benign','moderate','severe','adversarial']
 fig,axs=plt.subplots(2,1,figsize=(5.6,5.5),sharex=True)
 for ax,metric,label in zip(axs,['coverage','median_finite_width'],['(a) Safe-query coverage','(b) Median finite width']):
  for m in methods:
   g=s3[s3.method==m].set_index('mechanism').loc[order];k=key(m)
   ax.plot(range(4),g[metric],color=COLORS[k],marker=MARKERS[k],lw=1.3,ms=4,label=LABELS[k])
  styled_axis(ax,label);ax.tick_params(labelsize=10)
 axs[0].set_ylim(.84,1.09);axs[0].set_yticks([.85,.9,.95,1.0]);axs[0].axhline(.9,color='black',ls='--',lw=.8)
 style_lines(fig)
 h,l=axs[0].get_legend_handles_labels()
 # Column-major input produces the accepted row-wise order, with CGLCP last.
 order=[0,1,5,3,4,2]
 axs[0].legend([h[i] for i in order],[l[i] for i in order],loc='upper right',ncol=2,fontsize=9.2,frameon=True,facecolor='white',edgecolor='none',columnspacing=.8,handlelength=1.3,labelspacing=.25)
 axs[1].set_xlim(-.25,3.4);axs[1].set_xticks(range(4),['Benign','Moderate','Severe','Adversarial']);axs[1].set_xlabel('Unsafe-score mechanism')
 fig.subplots_adjust(left=.15,right=.98,bottom=.12,top=.96,hspace=.22);emit(fig,output/'score_robustness.pdf')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--local-summary',type=Path,default=ROOT/'frozen/local_necessity/summary.csv')
 p.add_argument('--robustness-summary',type=Path,default=ROOT/'frozen/score_robustness/summary.csv')
 p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures')
 a=p.parse_args()
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 plot_local(a.local_summary,a.output_dir);plot_robustness(a.robustness_summary,a.output_dir)
 print(f'Figures saved to {a.output_dir.resolve()}')
if __name__=='__main__':main()
