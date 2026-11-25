"""Optional publication figure, derived only from the archived E16 summary."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
s=json.loads((root/'results/e16/summary.json').read_text())
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1,1.1]})
colors=['#187E8A','#C77923']
for i,b in enumerate(['8192','40960']):
 r=s['best_minus_base'][b]['RR'];d=r['delta']*100;lo,hi=np.array(r['paired_domain_stratified_bootstrap_CI95'])*100
 ax.errorbar(d,1-i,xerr=[[d-lo],[hi-d]],fmt='o',color=colors[i],capsize=5,label=f'{int(b):,} tokens')
 ax.text(6.4,1-i,f"Holm p={r['Holm_adjusted_p_over_two_budgets']:.3f}",ha='right',va='bottom',fontsize=9)
ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks([1,0],['8,192 cap','40,960 cap']);ax.set_ylim(-.6,1.6);ax.set_xlim(-2.2,6.6);ax.set_xlabel('RR difference (percentage points)');ax.set_title('Coverage: positive, not confirmed');ax.grid(axis='x',alpha=.2)
x=np.arange(2)
for i,arm in enumerate(['base','best']):
 v=[s['cells'][f'{b}_{arm}']['domain_macro']['tokens_mean'] for b in ['8192','40960']]
 bars=bx.bar(x+(i-.5)*.34,v,width=.34,color=['#52647A','#187E8A'][i],label=['Base','Process distillation'][i])
 for bar,value in zip(bars,v):bx.text(bar.get_x()+bar.get_width()/2,value+75,f'{value:,.0f}',ha='center',fontsize=9)
bx.set_xticks(x,['8,192 cap','40,960 cap']);bx.set_ylim(0,7800);bx.set_ylabel('Mean generated tokens');bx.set_title('Output length: −27.5% / −23.9%');bx.legend(frameon=False,loc='upper right');bx.grid(axis='y',alpha=.2);bx.set_axisbelow(True)
fig.suptitle('E16: 36 held-out tasks, 4 samples/task, one selected checkpoint',fontsize=13)
fig.text(.5,.01,'RR bars: unadjusted paired 95% intervals; RR p-values: Holm over 2 budgets. Token intervals reported in README.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.07,1,.93]);(root/'figures').mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(root/f'figures/e16_primary.{ext}',dpi=180,bbox_inches='tight')
