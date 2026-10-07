#!/usr/bin/env python3
"""Reader-facing figures from frozen summary tables; no scoring or statistical refit."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,sys,platform,datetime
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'semantic_novelty_release_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import numpy as np

DEFAULT_ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=DEFAULT_ROOT)
parser.add_argument('--out',type=Path,default=DEFAULT_ROOT/'outputs/figures')
args=parser.parse_args(); root=args.project; out=args.out; out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.5,
                    'xtick.labelsize':8,'ytick.labelsize':8.5,'legend.fontsize':8,'pdf.fonttype':42,
                    'ps.fonttype':42,'axes.linewidth':.6,'savefig.facecolor':'white','figure.facecolor':'white',
                    'text.color':'#202020','axes.labelcolor':'#202020','xtick.color':'#303030','ytick.color':'#303030'})
DARK='#202020'; MID='#666666'; LIGHT='#e0e0e0'; MARK={'random':'o','source':'s','topic':'^'}
NAME={'random':'Random','source':'Source','topic':'Lexical proxy'}
metadata={'ACL':{'folder':'data/derived/original_acl','k':156,'n':1557,'h':3799,'family':45},
          'Biomedical':{'folder':'data/derived/original_biomedical','k':148,'n':1471,'h':3391,'family':18}}
sources={}; saved={}; selected=[]; qa=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def check(condition,message):
    if not condition:raise ValueError(message)
    qa.append(message)

def table(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

for collection,meta in metadata.items():
    d=root/meta['folder']; manifest=json.loads((d/'analysis_manifest.json').read_text())
    check(manifest['family_size']==meta['family'],f'{collection}: original family size verified')
    sources[str((d/'analysis_manifest.json').relative_to(root))]=sha(d/'analysis_manifest.json')
    for fn in ['table1_cohort_construction.csv','table2_aggregate.csv','table3_rank_selection.csv','table4_paired_primary.csv','baseline_cutoff.json']:
        p=d/fn;digest=sha(p)
        check(digest==manifest['output_sha256'][fn],f'{collection}: {fn} matches frozen analysis manifest')
        sources[str(p.relative_to(root))]=digest
        saved[collection,fn]=json.loads(p.read_text()) if fn.endswith('.json') else table(p)
    cohort=saved[collection,'table1_cohort_construction.csv']
    check(sum(int(x['n']) for x in cohort if x['cohort']=='history')==meta['h'],f'{collection}: historical count agrees with frozen cohort CSV')
    check(sum(int(x['n']) for x in cohort if x['cohort']=='focal')==meta['n'],f'{collection}: focal count agrees with frozen cohort CSV')
    cutoff=saved[collection,'baseline_cutoff.json']
    check(cutoff['k']==meta['k'] and cutoff['n']==meta['n'] and cutoff['baseline_selected_in_band']==16,
          f'{collection}: cohort denominator and 16 selected papers in cutoff band verified')

conditions=[]
for c in metadata:
    rates=[.1,.25,.5] if c=='ACL' else [.25,.5]
    mechanisms=['random','source','topic'] if c=='ACL' else ['random','source']
    conditions += [(c,r,m) for r in rates for m in mechanisms]
paired=[x for x in conditions if x[2]!='random']
check(len(conditions)==13,'All 13 main conditions selected')
check(len(paired)==8,'All 8 paired structured-control conditions selected')

def get(c,r,m,metric,paired=False):
    fn='table4_paired_primary.csv' if paired else ('table2_aggregate.csv' if metric=='G' else 'table3_rank_selection.csv')
    matches=[v for v in saved[c,fn] if float(v['rate'])==r and v['mechanism' if paired else 'condition']==m and v['metric']==metric and (paired or v['policy']=='mixed')]
    check(len(matches)==1,f'{c} {m} {r:g}: unique {metric} '+('paired' if paired else 'mixed')+' row')
    v=matches[0]
    if paired:
        check(v['matching']==('year;source' if m=='topic' else 'year'),f'{c} {m} {r:g}: control matching metadata verified')
    check(v['n_masks']=='200' and v['n_undefined']=='0',f'{c} {m} {r:g}: {metric} uses 200 complete masks')
    record={'collection':c,'source_file':metadata[c]['folder']+'/'+fn,'rate':r,'mechanism':m,'metric':metric,
            'paired':paired,'mean':float(v['mean']),'ci_low_raw':v['ci_low_raw'],'ci_high_raw':v['ci_high_raw'],
            'mcse':float(v['mcse']),'interval':v['interval']}
    selected.append(record)
    return v

def point(c,r,m,metric,pair=False):
    v=get(c,r,m,metric,pair); f=metadata[c]['family']
    check(v['interval']==f'approximate_95pct_FWER_{f}_Bonferroni_Student_mean',f'{c} {m} {r:g}: {metric} retains original {f}-mean interval')
    z=[float(v[s]) for s in ['mean','ci_low_raw','ci_high_raw']]
    check(all(math.isfinite(x) for x in z) and z[1]<=z[0]<=z[2],f'{c} {m} {r:g}: {metric} finite ordered interval')
    return z

main={(*cond,metric):point(*cond,metric) for cond in conditions for metric in ['G','T','Q']}
pair_values={(*cond,metric):point(*cond,metric,True) for cond in paired for metric in ['G','T','Q']}
cutoff_values={}
for c,r,m in conditions:
    vals={name:float(get(c,r,m,name)['mean']) for name in ['selected_lost_count','loss_in_cutoff_band_per_k','loss_outside_cutoff_band_per_k','far_loss_count_0.1']}
    k=metadata[c]['k']; inside=vals['loss_in_cutoff_band_per_k']*k; outside=vals['loss_outside_cutoff_band_per_k']*k
    total=vals['selected_lost_count'];far=vals['far_loss_count_0.1']
    check(abs(inside+outside-total)<1e-9,f'{c} {m} {r:g}: rank-band stacks reconcile to selected loss count')
    check(abs(total-main[c,r,m,'T'][0]*k)<1e-9,f'{c} {m} {r:g}: count equals T times k')
    check(0<=inside<=16+1e-9 and 0<=far<=total,f'{c} {m} {r:g}: loss component bounds verified')
    cutoff_values[c,r,m]=[inside,outside,total,far]

outputs={}
def export(fig,stem):
    fig.canvas.draw()
    # No tight bounding box: every exported figure remains exactly 5.2 inches wide.
    fig.savefig(out/(stem+'.pdf'),metadata={'Title':stem,'Creator':'rebuild_figures.py','CreationDate':None,'ModDate':None})
    fig.savefig(out/(stem+'.png'),dpi=300)
    outputs[stem]={'width_in':float(fig.get_size_inches()[0]),'height_in':float(fig.get_size_inches()[1]),
                  'pdf_sha256':sha(out/(stem+'.pdf')),'png_sha256':sha(out/(stem+'.png'))}
    plt.close(fig)

def axis_style(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#dddddd',linewidth=.45)
    for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=7)
    ax.tick_params(axis='x',length=3,width=.6)

def yvalues(rows):
    # Leave an explicitly labeled gap between the two independently analyzed collections.
    ys=[];cursor=0
    for i,row in enumerate(rows):
        if i and row[0]!=rows[i-1][0]:cursor+=1.5
        ys.append(cursor);cursor+=1
    return ys

def forest_frame(rows,height,title,sub):
    fig,axes=plt.subplots(1,3,figsize=(6.4,height),sharey=True)
    fig.subplots_adjust(left=.257,right=.977,bottom=.17,top=.80,wspace=.22)
    ys=yvalues(rows)
    for ax in axes:
        axis_style(ax);ax.set_ylim(ys[-1]+.8,-1.65)
        ax.set_yticks(ys)
    axes[0].set_yticklabels([f'{NAME[m]}  {r:.0%}' for c,r,m in rows])
    for ax in axes[1:]:ax.tick_params(labelleft=False)
    for c in metadata:
        idx=next(i for i,row in enumerate(rows) if row[0]==c)
        axes[0].text(-.22,ys[idx]-.97,c if c=='Biomedical' else 'ACL/EMNLP',transform=axes[0].get_yaxis_transform(),
                     fontweight='bold',ha='right',va='center',fontsize=9,clip_on=False)
    split=next(i for i,row in enumerate(rows) if row[0]=='Biomedical')
    for ax in axes:ax.axhline((ys[split-1]+ys[split])/2,color='#bdbdbd',lw=.6)
    fig.text(.025,.973,title,weight='bold',size=11,va='top')
    fig.text(.025,.922,sub,size=8.3,va='top',linespacing=1.35)
    return fig,axes,ys


# New presentation only: the source rows, original intervals and checks above are unchanged.
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':8})
def compact_frame(rows,height):
    fig,axes=plt.subplots(1,3,figsize=(5.2,height),sharey=True)
    fig.subplots_adjust(left=.295,right=.968,bottom=.145,top=.90,wspace=.29)
    ys=yvalues(rows)
    for ax in axes:
        axis_style(ax);ax.set_ylim(ys[-1]+.7,-1.4);ax.set_yticks(ys)
    axes[0].set_yticklabels([f'{NAME[m]} {r:.0%}' for c,r,m in rows])
    for ax in axes[1:]:ax.tick_params(labelleft=False)
    for c in metadata:
        idx=next(i for i,row in enumerate(rows) if row[0]==c)
        axes[0].text(-.04,ys[idx]-.90,'ACL/EMNLP' if c=='ACL' else c,transform=axes[0].get_yaxis_transform(),fontweight='bold',ha='right',va='center',fontsize=8,clip_on=False)
    split=next(i for i,row in enumerate(rows) if row[0]=='Biomedical')
    for ax in axes:ax.axhline((ys[split-1]+ys[split])/2,color='#bdbdbd',lw=.6)
    return fig,axes,ys

fig=plt.figure(figsize=(5.2,2.45));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,heading,body):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.006',facecolor='white',edgecolor=MID,lw=.8))
    ax.text(x+w/2,y+h-.05,heading,ha='center',va='top',weight='bold',size=8)
    ax.text(x+w/2,y+h-.15,body,ha='center',va='top',size=8,linespacing=1.35)
def arrow(p,q):ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,lw=.9,color=DARK))
box(.015,.60,.285,.335,'Historical papers','2018-2020\nTitle + abstract')
box(.357,.60,.285,.335,'Historical masks','Random / structured\nExact missing count')
box(.700,.60,.285,.335,'Retained history','Masked: authentic title\nOthers: title + abstract')
arrow((.307,.765),(.35,.765));arrow((.65,.765),(.692,.765))
box(.015,.055,.285,.34,'Fixed focal papers','2021 title + abstract\nUnchanged across masks')
box(.357,.055,.285,.34,'Nearest-prior score','Pinned SPECTER\n1 - max prior cosine')
box(.700,.055,.285,.34,'Measurement uses','G: contrast distortion\nT: selection; Q: ranks')
arrow((.307,.225),(.35,.225));arrow((.65,.225),(.692,.225))
ax.plot([.842,.842,.5],[.59,.48,.48],color=DARK,lw=.9);arrow((.5,.48),(.5,.40))
export(fig,'figure1_study_design')

fig,axes,ys=compact_frame(conditions,4.7)
spec=[('G',1,(-.018,.41),[0,.2,.4],.1,'a  Contrast','G (baseline SD)'),('T',100,(0,35),[0,10,20,30],5,'b  Selection','T (% selected)'),('Q',100,(0,12),[0,4,8,12],1,'c  Ranks','Q (% rank range)')]
for ax,(metric,scale,lim,ticks,tol,title,label) in zip(axes,spec):
    ax.set_title(title,loc='left',pad=8);ax.set(xlim=lim,xticks=ticks,xlabel=label);ax.axvline(tol,lw=.85,color=MID,ls=(0,(4,3)))
    for y,(c,r,m) in zip(ys,conditions):
        mean,lo,hi=[x*scale for x in main[c,r,m,metric]]
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=MARK[m],markersize=3.5,markerfacecolor='white' if m=='source' else DARK,markeredgecolor=DARK,markeredgewidth=.8,color=DARK,lw=.85,capsize=2,capthick=.8,zorder=3)
export(fig,'figure2_primary_endpoints')

fig,axes,ys=compact_frame(paired,3.5)
spec=[('G',1,(-.04,.38),[0,.1,.2,.3],'a  Contrast',r'$\Delta G$ (SD)'),('T',100,(-4.5,7.5),[-4,0,4],'b  Selection',r'$\Delta T$ (pp)'),('Q',100,(-.7,1.75),[-.5,.5,1.5],'c  Ranks',r'$\Delta Q$ (pp)')]
for ax,(metric,scale,lim,ticks,title,label) in zip(axes,spec):
    ax.set_title(title,loc='left',pad=8);ax.set(xlim=lim,xticks=ticks,xlabel=label);ax.axvline(0,lw=.85,color=MID,ls=(0,(4,3)))
    for y,(c,r,m) in zip(ys,paired):
        mean,lo,hi=[x*scale for x in pair_values[c,r,m,metric]]
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=MARK[m],markersize=3.5,markerfacecolor='white' if m=='source' else DARK,markeredgecolor=DARK,markeredgewidth=.8,color=DARK,lw=.85,capsize=2,capthick=.8,zorder=3)
export(fig,'figure3_paired_effects')

fig,axes=plt.subplots(1,2,figsize=(5.2,4.6),sharey=True,gridspec_kw={'width_ratios':[1.4,1]})
fig.subplots_adjust(left=.29,right=.985,top=.88,bottom=.155,wspace=.20)
ys=yvalues(conditions)
for ax in axes:
    axis_style(ax);ax.set_ylim(ys[-1]+.7,-1.4);ax.set_yticks(ys);ax.set(xlim=(0,52),xticks=[0,20,40]);ax.set_xlabel('Expected papers lost')
axes[0].set_yticklabels([f'{NAME[m]} {r:.0%}' for c,r,m in conditions]);axes[1].tick_params(labelleft=False)
axes[0].set_title('a  Rank-band origins',loc='left',pad=28);axes[1].set_title('b  Score margin\n> 0.10 baseline SD',loc='left',pad=8)
for c in metadata:
    idx=next(i for i,row in enumerate(conditions) if row[0]==c)
    axes[0].text(-.04,ys[idx]-.9,'ACL/EMNLP' if c=='ACL' else c,transform=axes[0].get_yaxis_transform(),weight='bold',ha='right',va='center',size=8,clip_on=False)
split=next(i for i,row in enumerate(conditions) if row[0]=='Biomedical')
for ax in axes:ax.axhline((ys[split-1]+ys[split])/2,color='#bdbdbd',lw=.6)
for y,(c,r,m) in zip(ys,conditions):
    inside,outside,total,far=cutoff_values[c,r,m]
    axes[0].barh(y,inside,height=.54,color=LIGHT,edgecolor=DARK,lw=.6,zorder=2)
    axes[0].barh(y,outside,left=inside,height=.54,color=MID,edgecolor=DARK,lw=.6,hatch='///',zorder=2)
    lo,hi=[x*metadata[c]['k'] for x in main[c,r,m,'T'][1:]]
    axes[0].errorbar(total,y,xerr=[[total-lo],[hi-total]],fmt='none',color=DARK,capsize=2,lw=.85,zorder=4)
    axes[1].plot(far,y,marker='D',ms=3.8,mfc='white',mec=DARK,mew=.9,ls='none')
handles=[Patch(facecolor=LIGHT,edgecolor=DARK,label='Inside'),Patch(facecolor=MID,edgecolor=DARK,hatch='///',label='Outside')]
axes[0].legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.005),frameon=False,ncol=2,columnspacing=.8,handlelength=1.25,fontsize=8,borderaxespad=0)
export(fig,'figure4_cutoff_losses')

# Machine-readable bindings of the unchanged measurements to the new displays.
(out/'plotted_rows.json').write_text(json.dumps(selected,indent=2)+'\n')
(out/'cutoff_plot_counts.json').write_text(json.dumps([{'collection':c,'rate':r,'mechanism':m,'k':metadata[c]['k'],'inside_count':v[0],'outside_count':v[1],'total_count':v[2],'far_margin_count':v[3]} for (c,r,m),v in cutoff_values.items()],indent=2)+'\n')
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':'release-relative','script_sha256':sha(Path(__file__)),'source_sha256':sources,'outputs':outputs,'plotted_rows_sha256':sha(out/'plotted_rows.json'),'cutoff_counts_sha256':sha(out/'cutoff_plot_counts.json'),'new_scoring':False,'new_statistical_fit':False,'original_outputs_modified':False,'checks_passed':qa,'visual_qa':'pending latest PNG inspection','changes':'Same original data/intervals; compact labels and moved explanatory notes to captions. Figure4 separately displays overlapping rank-band and score-margin counts into two panels.'}
(out/'figures_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'outputs':outputs,'checks_passed':len(qa),'main_conditions':len(conditions),'paired_conditions':len(paired)},indent=2))
