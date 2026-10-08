"""Format frozen aggregate summaries as current tables; no scoring or estimation."""
import argparse,csv,json,hashlib
from pathlib import Path

def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path,default=Path('outputs/revision_mechanism_tables'));a=ap.parse_args()
    d=a.root/'data/derived/revision_descriptive/mechanism';a.out.mkdir(parents=True,exist_ok=True)
    intensity=rows(d/'intensity_summary.csv');neighbors=rows(d/'neighbor_summary.csv')
    short={'random':'Random','source':'Source','source_matched_control':'Source control','topic':'Lexical','topic_matched_control':'Lexical control'}
    def fmt(r):return f"{100*float(r['within_stratum_fraction_mean']):.1f} ({100*float(r['within_stratum_fraction_assignment_sd']):.1f})"
    source=[];lexical=[];nb=[]
    for study in ('ACL','Biomedical'):
        cells=sorted({(r['condition'],float(r['rate'])) for r in intensity if r['study']==study and r['stratum_type']=='source'})
        for cond,rate in cells:
            pair=[r for r in intensity if r['study']==study and r['condition']==cond and float(r['rate'])==rate and r['stratum_type']=='source'];pair.sort(key=lambda r:r['stratum']!=('ACL' if study=='ACL' else 'PLOSCB'))
            assert len(pair)==2 and all(int(r['n_masks'])==200 for r in pair)
            source.append(f"{study} & {short[cond]} & {100*rate:.0f} & {fmt(pair[0])} & {fmt(pair[1])} & {pair[0]['N_s']} & {pair[1]['N_s']} "+r'\\')
    for cond,rate in sorted({(r['condition'],float(r['rate'])) for r in intensity if r['stratum_type']=='lexical_proxy'}):
        pair=[r for r in intensity if r['condition']==cond and float(r['rate'])==rate and r['stratum_type']=='lexical_proxy'];pair.sort(key=lambda r:r['stratum']!='translat');assert len(pair)==2
        lexical.append(f"{short[cond]} & {100*rate:.0f} & {fmt(pair[0])} & {fmt(pair[1])} "+r'\\')
    for r in neighbors:
        if r['condition']!='source' or r['focal_source']=='ALL':continue
        nb.append(f"{r['study']}/{r['focal_source']} & {100*float(r['rate']):.0f} & {100*float(r['exposed_fraction_mean']):.1f} & {100*float(r['identity_switch_fraction_mean']):.1f} & {100*float(r['winner_title_fraction_mean']):.1f} & {float(r['mean_shift_sd_mean']):+.3f} & {100*float(r['baseline_selected_loss_fraction_mean']):.1f} "+r'\\')
    assert (len(source),len(lexical),len(nb))==(21,6,8)
    for name,body,cols,header in [
        ('table_S16_source_intensity',source,'llrrrrr',r'Collection & Condition & Masking (\%) & Source A & Source B & $N_A$ & $N_B$ \\'),
        ('table_S17_lexical_intensity',lexical,'lrrr',r'Condition & Masking (\%) & Proxy & Nonproxy \\'),
        ('table_S18_neighbor_summary',nb,'llrrrrr',r'Collection/source & Masking (\%) & Exposed & Switch & Title & Shift & Selected loss (\%) \\')]:
        (a.out/(name+'.tex')).write_text('\n'.join([r'\begin{tabular}{'+cols+'}',r'\toprule',header,r'\midrule',*body,r'\bottomrule',r'\end{tabular}'])+'\n')
        (a.out/(name+'_rows.txt')).write_text('\n'.join(body)+'\n')
    receipt={'scope':'Frozen aggregate reporting only. Tables S16/S17: percentages (assignment SD); source A/B historical counts. Table S18 selected loss denominator is baseline-selected count within each source; exposure/switch/title use fixed focal-source N. No new inference or neighbor identification.','rows':{'S16':len(source),'S17':len(lexical),'S18':len(nb)},'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [d/'intensity_summary.csv',d/'neighbor_summary.csv']}}
    (a.out/'reporting_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
