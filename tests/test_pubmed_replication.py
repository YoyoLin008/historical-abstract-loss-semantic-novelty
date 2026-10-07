"""Independent-collection contracts on fabricated records; no biomedical scores."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import replication.score as score_module
from replication.score import (create_seed_manifest, validate_seed_manifest,
    replication_provenance, validate_cohort_manifest, select_all_cohorts,
    matched_control, validate_config, run as score_run)
from replication.analysis import (baseline_context, mask_metrics, selected,
    summarize, verify_critical, run as analysis_run)
from src.pilot import file_sha256, fingerprints
from check_pubmed_replication_empirical import empirical as independent_check


def configuration():
    path = Path(__file__).resolve().parents[1]/"configs/pubmed_replication_v1.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"stage":"pubmed_replication_v1","focal_selection":"all_eligible",
        "history_years":[2018,2019,2020],"focal_years":[2021],"sources":["PLOSCB","PLOSGEN"],
        "source_contrast":["PLOSCB","PLOSGEN"],"rates":[.25,.5],"masks_per_condition":200,
        "mask_preferred_source":"PLOSCB","structured_weight":8.,"top_k":20,"block_size":128,
        "control_matching":{"source":["year"]}}


def fixture(directory, full_design=False):
    p=Path(directory);config=configuration()
    if not full_design:config.update(rates=[.25],masks_per_condition=2)
    rows=[]
    for year in (2018,2019,2020,2021):
        for source in ("PLOSCB","PLOSGEN"):
            for j in range(5):
                title=f"SYNTHETIC title {j}";abstract=f"SYNTHETIC abstract {year} {source} {j}"
                rows.append(dict(id=f"{year}-{source}-{j}",year=year,source=source,title=title,abstract=abstract,topic=0,
                    title_sha256=hashlib.sha256(title.encode()).hexdigest(),abstract_sha256=hashlib.sha256(abstract.encode()).hexdigest()))
    (p/"records.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rng=np.random.default_rng(39280)
    for mode in ('full','title'):np.save(p/(mode+'.npy'),rng.normal(size=(len(rows),11)).astype(np.float32))
    (p/'config.json').write_text(json.dumps(config));(p/'seeds.json').write_text(json.dumps(create_seed_manifest(config)))
    (p/'protocol.md').write_text('Synthetic replication protocol fixture, not an empirical freeze.\n')
    (p/'requirements.lock').write_text('numpy=='+np.__version__+'\n')
    (p/'replication').mkdir();(p/'src').mkdir()
    package_hashes={}
    for source in Path(score_module.__file__).parent.glob('*.py'):
        (p/'replication'/source.name).write_bytes(source.read_bytes())
        package_hashes['replication/'+source.name]=file_sha256(source)
    cohort={'quality_pass':True,'records_sha256':file_sha256(p/'records.jsonl'),
        'date_definition':'Synthetic exact-year fixture, not a PubMed rule','focal_count':10,'history_count':30,
        'focal':[r for r in rows if r['year']==2021],'history':[r for r in rows if r['year']<2021],
        'new_duplicate_exclusions':[]}
    (p/'cohort_manifest.json').write_text(json.dumps(cohort))
    if not (p/'replication/acquire.py').exists():
        (p/'replication/acquire.py').write_text('# Synthetic acquisition fixture only.\n')
    (p/'docs').mkdir();(p/'docs/replication_cohort_rules_v1.md').write_text('Synthetic fixture inclusion rules.\n')
    acquisition={'scope':'fabricated test records only','code_sha256':file_sha256(p/'replication/acquire.py'),
        'rules_sha256':file_sha256(p/'docs/replication_cohort_rules_v1.md'),
        'data_sha256':{name:file_sha256(p/name) for name in ('records.jsonl','cohort_manifest.json')}}
    (p/'acquisition.json').write_text(json.dumps(acquisition))
    enc={'status':'complete','fingerprint':{'schema':2,'model':'synthetic','revision':'synthetic-v1',
        'data_sha256':file_sha256(p/'records.jsonl'),'model_files':{'synthetic.bin':'0'*64},
        'requirements_lock_sha256':file_sha256(p/'requirements.lock')},
        'output_sha256':{mode:file_sha256(p/(mode+'.npy')) for mode in ('full','title')}}
    (p/'encoding_manifest.json').write_text(json.dumps(enc))
    args=argparse.Namespace(root=str(p),records=str(p/'records.jsonl'),full=str(p/'full.npy'),title=str(p/'title.npy'),
        config=str(p/'config.json'),seed_manifest=str(p/'seeds.json'),protocol=str(p/'protocol.md'),
        cohort_manifest=str(p/'cohort_manifest.json'),acquisition_manifest=str(p/'acquisition.json'),
        encoding_manifest=str(p/'encoding_manifest.json'),freeze_manifest=str(p/'freeze.json'),
        output=str(p/'results'),resume=False,dry_run=False,max_masks=None,synthetic_test=True)
    provenance=fingerprints(args)
    # Copy the shared source snapshots so the independent analysis can verify them.
    import src.pilot
    for name in provenance['source_code_sha256']:
        target=p/'src'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(src.pilot.__file__).parent/name).read_bytes())
    freeze={'status':'frozen','stage':'pubmed_replication_v1','git_commit':'0'*40,
        'frozen_utc':'2026-10-04T00:00:00Z','originality_gate':'PASS WITH NARROWER CLAIM',
        'source_code_sha256':provenance['source_code_sha256'],'replication_code_sha256':package_hashes,
        'analysis_code_sha256':{'replication/analysis.py':file_sha256(p/'replication/analysis.py')},
        'sha256':{key:file_sha256(path) for key,path in {
            'records':args.records,'full':args.full,'title':args.title,'config':args.config,'seeds':args.seed_manifest,
            'protocol':args.protocol,'cohort_manifest':args.cohort_manifest,'acquisition_manifest':args.acquisition_manifest,
            'encoding_manifest':args.encoding_manifest,'requirements_lock':p/'requirements.lock'}.items()}}
    (p/'freeze.json').write_text(json.dumps(freeze))
    return args,rows,config


class ReplicationContractTests(unittest.TestCase):
    def test_explicit800_base400_control_seeds_new_namespace(self):
        config=configuration();validate_config(config)
        manifest=create_seed_manifest(config);rows=validate_seed_manifest(manifest,config)
        self.assertEqual(len(rows),800);self.assertEqual(sum('control_seed' in row for row in rows),400)
        self.assertEqual({r['mechanism'] for r in rows},{'random','source'})
        vectors=[tuple(r[key]) for r in rows for key in ['mask_seed','control_seed'] if key in r]
        self.assertEqual(len(set(vectors)),1200)
        self.assertEqual(manifest['namespace'],'pubmed-2021-replication-v1')
        for old in (20261004,3216741901):
            with self.assertRaises(ValueError):create_seed_manifest(config,master_seed=old)
        modified=copy.deepcopy(manifest);modified['assignments'][0]['mask_seed'][0]+=1
        with self.assertRaises(ValueError):validate_seed_manifest(modified,config)

    def test_dynamic_cohort_date_ids_and_source_control(self):
        with tempfile.TemporaryDirectory() as directory:
            args,rows,config=fixture(directory)
            f,h=select_all_cohorts(rows,config)
            self.assertEqual(len(f),10);self.assertEqual(len(h),30)
            validate_cohort_manifest(rows,f,h,args)
            history=[rows[i] for i in h];mask=np.array([r['source']=='PLOSCB' for r in history])
            control,cells=matched_control(history,mask,np.random.default_rng(94),('year',))
            for year in (2018,2019,2020):
                ix=np.array([r['year']==year for r in history]);self.assertEqual(mask[ix].sum(),control[ix].sum())
            self.assertNotEqual(int(control[np.array([r['source']=='PLOSCB' for r in history])].sum()),int(mask.sum()))
            cohort=json.loads(Path(args.cohort_manifest).read_text());original=copy.deepcopy(cohort)
            cohort['focal'][0]['year']=2020
            Path(args.cohort_manifest).write_text(json.dumps(cohort))
            with self.assertRaisesRegex(ValueError,'date/source'):validate_cohort_manifest(rows,f,h,args)
            cohort=original;del cohort['date_definition']
            Path(args.cohort_manifest).write_text(json.dumps(cohort))
            with self.assertRaisesRegex(ValueError,'date|year'):validate_cohort_manifest(rows,f,h,args)

    def test_replication_code_acquisition_and_lock_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=fixture(directory);p=Path(directory)
            provenance=replication_provenance(args)
            self.assertIn('replication/score.py',provenance['replication_code_sha256'])
            self.assertIn('acquisition_manifest',provenance['inputs'])
            for field in ('acquisition.json','cohort_manifest.json','requirements.lock','replication/analysis.py','replication/acquire.py','docs/replication_cohort_rules_v1.md'):
                path=p/field;old=path.read_bytes();path.write_bytes(old+b' ')
                with self.subTest(field=field),self.assertRaises(ValueError):replication_provenance(args)
                path.write_bytes(old)

    def test_checkpoint_clean_outputs_and_stage18(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=fixture(directory);args.max_masks=1
            self.assertEqual(score_run(args)['status'],'partial')
            args.max_masks=None;args.resume=True
            self.assertEqual(score_run(args)['status'],'complete')
            resumed=Path(args.output)
            summary=json.loads((resumed/'summary.json').read_text())
            self.assertEqual(summary['stage'],'pubmed_replication_v1')
            self.assertEqual(summary['primary_inference']['family_size'],18)
            args.output=str(Path(directory)/'clean');args.resume=False;score_run(args)
            for name in ('baseline.json','mask_results.jsonl','summary.json'):
                self.assertEqual((resumed/name).read_bytes(),(Path(args.output)/name).read_bytes())


class ReplicationAnalysisTests(unittest.TestCase):
    def test_dynamic_selection_denominator_ties_and_margins(self):
        n=21;b=np.arange(n,dtype=float);ids=np.array([f'id{i:02}' for i in range(n)])
        sources=['PLOSCB']*10+['PLOSGEN']*11
        ctx=baseline_context(b,ids,sources);self.assertEqual(ctx['k'],3)
        changed=b.copy();changed[0],changed[-1]=changed[-1],changed[0]
        result,_,_=mask_metrics(changed,ctx);self.assertAlmostEqual(result['T'],1/3)
        self.assertAlmostEqual(result['Q'],2/21)
        tied=np.array([1.,1.,0.]);chosen=selected(tied,np.array(['b','a','c']))
        self.assertTrue(chosen[1]);self.assertEqual(int(chosen.sum()),1)

    def test_18_family_critical_and_missing_data_guard(self):
        result=verify_critical();self.assertEqual(result['family'],18)
        self.assertAlmostEqual(result['critical'],3.029133739375753,places=13)
        values=np.linspace(0,1,200);out=summarize(values,True)
        expected=float(values.mean()+result['critical']*values.std(ddof=1)/np.sqrt(200))
        self.assertAlmostEqual(out['ci_high_raw'],expected)
        with self.assertRaises(ValueError):summarize([None]*200,True)

    def test_full_synthetic18_family_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=fixture(directory,full_design=True)
            completed=score_run(args);self.assertEqual(completed['completed_assignments'],800)
            analysis_args=argparse.Namespace(**vars(args))
            analysis_args.results=args.output;analysis_args.output=str(Path(directory)/'analysis_output')
            analysis_args.report=str(Path(directory)/'synthetic_report.md')
            result=analysis_run(analysis_args)
            self.assertEqual(result['family_size'],18)
            manifest=json.loads((Path(analysis_args.output)/'analysis_manifest.json').read_text())
            self.assertEqual(manifest['n_assignments'],800);self.assertEqual(manifest['n_matched_controls'],400)
            self.assertEqual(len(list(Path(analysis_args.output).glob('figure*.png'))),3)
            cutoff=json.loads((Path(analysis_args.output)/'baseline_cutoff.json').read_text())
            self.assertEqual(cutoff['k'],1)
            construction=json.loads((Path(analysis_args.output)/'cohort_construction_metadata.json').read_text())
            self.assertEqual(construction['new_duplicate_exclusion_ledger_entries_entire_frame'],0)
            self.assertNotIn('exclusions_new',(Path(analysis_args.output)/'table1_cohort_construction.csv').read_text().splitlines()[0])
            checking=argparse.Namespace(**vars(args))
            checking.results=args.output;checking.analysis=analysis_args.output
            checking.atol=2e-6;checking.focals_per_source=1
            checking.output=str(Path(directory)/'synthetic_independent_validation.json')
            verified=independent_check(checking)
            self.assertEqual(verified['analysis_intervals_checked'],18)
            self.assertEqual(verified['movement_rows_checked'],1200)
            self.assertEqual(verified['cutoff_bins_checked'],120)


if __name__=='__main__':unittest.main()
