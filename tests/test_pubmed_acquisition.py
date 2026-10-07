"""Synthetic biomedical acquisition audit. No network requests are permitted."""
from pathlib import Path
import contextlib, copy, importlib.util, io, json, tempfile, unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('biomedical_acquisition_reviewed',HERE.parent/'replication/acquire.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def article(pmid='1',identifier='10.1371/journal.pcbi.0000001',source='PLOSCB',year=2018,title='A <i>careful</i> study',sections='<AbstractText Label="BACKGROUND">A result with <i>inline</i> text.</AbstractText><AbstractText Label="RESULTS">Another finding.</AbstractText>',dates=None,relationship='',ptypes='<PublicationType>Journal Article</PublicationType>'):
    if dates is None:dates=f'<ArticleDate DateType="Electronic"><Year>{year}</Year><Month>05</Month><Day>01</Day></ArticleDate>'
    return ET.fromstring(f'''<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Journal><Title>{a.SOURCES[source][0]}</Title></Journal><ArticleTitle>{title}</ArticleTitle><Abstract>{sections}<CopyrightInformation>Copyright-only attribution, not abstract.</CopyrightInformation></Abstract><Language>eng</Language><PublicationTypeList>{ptypes}</PublicationTypeList><AuthorList><Author><LastName>Smith</LastName><ForeName>Anne Marie</ForeName><Initials>AM</Initials><AffiliationInfo><Affiliation>Must not enter author name</Affiliation></AffiliationInfo></Author><Author><CollectiveName>Study Consortium</CollectiveName></Author></AuthorList>{dates}</Article>{relationship}</MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">{identifier}</ArticleId></ArticleIdList></PubmedData></PubmedArticle>''')


def test_corpus():
    docs={};xml={};discovery={};ids={};n=100
    for year in a.YEARS:
        for source in a.SOURCES:
            n+=1;pmid=str(n);identifier=f'10.1371/journal.{"pcbi" if source=="PLOSCB" else "pgen"}.{year}001'
            ids[(source,year)]=(identifier,pmid)
            docs[(source,year)]={'id':identifier,'journal':a.SOURCES[source][0],'article_type':'Research Article','doc_type':'full','publication_date':f'{year}-05-01T00:00:00Z','author_display':['Anne Marie Smith'],'copyright':'CC BY'}
            xml[pmid]=article(pmid,identifier,source,year,title=f'Paper {source} {year}',sections=f'<AbstractText>Unique findings {source} {year}.</AbstractText>')
            discovery[(source,year)]=[pmid]
    # True absent join, undated record found by DOI rescue, all-placeholder abstract,
    # ordinary linked erratum, current retraction, cross-year exact-text duplicate.
    discovery[('PLOSGEN',2018)]=[]
    d,p=ids[('PLOSGEN',2019)];discovery[('PLOSGEN',2019)]=[]
    xml[p]=article(p,d,'PLOSGEN',2019,dates='')
    d,p=ids[('PLOSGEN',2020)]
    xml[p]=article(p,d,'PLOSGEN',2020,sections='<AbstractText Label="BACKGROUND">N/A</AbstractText><AbstractText Label="RESULTS">none</AbstractText>')
    d,p=ids[('PLOSCB',2019)]
    xml[p]=article(p,d,'PLOSCB',2019,title='Ordinary erratum original',relationship='<CommentsCorrectionsList><CommentsCorrections RefType="ErratumIn"><PMID>999</PMID></CommentsCorrections></CommentsCorrectionsList>')
    d,p=ids[('PLOSCB',2020)]
    xml[p]=article(p,d,'PLOSCB',2020,relationship='<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><PMID>998</PMID></CommentsCorrections></CommentsCorrectionsList>')
    d,p=ids[('PLOSCB',2021)]
    canonical=xml[ids[('PLOSCB',2018)][1]]
    xml[p]=copy.deepcopy(canonical)
    xml[p].find('./MedlineCitation/PMID').text=p
    xml[p].find('./PubmedData/ArticleIdList/ArticleId').text=d
    xml[p].find('./MedlineCitation/Article/ArticleDate/Year').text='2021'
    xml[ids[('PLOSGEN',2021)][1]].find('./MedlineCitation/Article/ArticleDate/Day').text='02'
    return docs,xml,discovery,ids


def fake_client(docs,xml,discovery,ids):
    class FakeClient:
        instance=None
        def __init__(self,folder):
            self.calls=[];self.rescue_queries=[];self.fetches=[];FakeClient.instance=self
        def plos(self,source,year,start):
            return {'numFound':1,'start':start,'docs':[docs[(source,year)]]}
        def ids(self,source,year):return discovery[(source,year)]
        def doi_ids(self,dois,start):
            self.rescue_queries.extend(dois)
            undated_doi,undated_pmid=ids[('PLOSGEN',2019)]
            return [undated_pmid,ids[('PLOSCB',2018)][1]] if undated_doi in dois else []
        def articles(self,source,year,pmids,start):
            self.fetches.append((source,tuple(pmids)))
            return [copy.deepcopy(xml[p]) for p in pmids]
    return FakeClient


class BiomedicalAcquisitionTests(unittest.TestCase):
    def setUp(self):
        p=patch.object(a,'urlopen',side_effect=AssertionError('Network forbidden in synthetic audit'));p.start();self.addCleanup(p.stop)

    def test_structured_text_placeholders_doi_and_author_attribution(self):
        r=a.parse_pubmed(article(identifier=' HTTPS://doi.org/10.1371/JOURNAL.PCBI.0000001 '))
        self.assertEqual(r['dois'],['10.1371/journal.pcbi.0000001'])
        self.assertEqual(r['title'],'A careful study')
        self.assertEqual(r['abstract'],'BACKGROUND: A result with inline text. RESULTS: Another finding.')
        self.assertTrue(r['has_substantive_abstract'])
        self.assertEqual(r['authors'],['Anne Marie Smith','Study Consortium'])
        self.assertNotIn('Copyright',r['abstract']);self.assertNotIn('Affiliation',str(r['authors']))
        r=a.parse_pubmed(article(sections='<AbstractText Label="BACKGROUND">N/A</AbstractText><AbstractText Label="RESULTS"> none </AbstractText>'))
        self.assertFalse(r['has_substantive_abstract']);self.assertEqual(r['abstract'],'BACKGROUND: N/A RESULTS: none')
        self.assertEqual(a.doi('doi: 10.1371/ABC.1'),'10.1371/abc.1')
        self.assertEqual(a.doi('https://doi.org/ 10.1371/ABC.1'),'10.1371/abc.1')

    def test_electronic_dates_require_unique_valid_full_date_without_fallback(self):
        valid='<ArticleDate DateType="Electronic"><Year>2020</Year><Month>02</Month><Day>29</Day></ArticleDate>'
        self.assertEqual(a.parse_pubmed(article(dates=valid+valid))['date'],'2020-02-29')
        malformed=valid.replace('2020','2019')
        other=valid.replace('<Day>29</Day>','<Day>28</Day>')
        for dates,reason in [('', 'missing_electronic_date'),(malformed,'malformed_electronic_date'),(valid+other,'ambiguous_electronic_date'),(valid+malformed,'malformed_electronic_date'),('<ArticleDate DateType="Print"><Year>2018</Year><Month>01</Month><Day>01</Day></ArticleDate>','missing_electronic_date')]:
            with self.subTest(reason=reason):
                r=a.parse_pubmed(article(dates=dates));self.assertIsNone(r['date']);self.assertEqual(r['date_error'],reason)

    def test_queries_and_api_cardinality_checks(self):
        c=a.Client.__new__(a.Client);calls=[]
        def get(label,url):
            calls.append((label,url));return json.dumps({'esearchresult':{'count':'1','idlist':['1']}}).encode()
        c.get=get
        self.assertEqual(c.doi_ids(['10.1371/journal.pcbi.1'],0),['1'])
        query=parse_qs(urlsplit(calls[-1][1]).query)['term'][0]
        self.assertEqual(query,'"10.1371/journal.pcbi.1"[aid]');self.assertNotIn('epdat',query)
        c.get=lambda label,url:json.dumps({'esearchresult':{'count':'2','idlist':['1']}}).encode()
        with self.assertRaises(AssertionError):c.doi_ids(['10.1371/a'],0)
        c.get=lambda label,url:json.dumps({'esearchresult':{'count':'2','idlist':['1','1']}}).encode()
        with self.assertRaises(AssertionError):c.ids('PLOSCB',2018)
        c.get=lambda label,url:b'<PubmedArticleSet>'+ET.tostring(article())+ET.tostring(article())+b'</PubmedArticleSet>'
        with self.assertRaises(AssertionError):c.articles('PLOSCB',2018,['1','1'],0)
        def plos_get(label,url):
            calls.append((label,url));return b'{"response":{"numFound":0,"start":0,"docs":[]}}'
        c.get=plos_get;c.plos('PLOSCB',2018,0)
        query=parse_qs(urlsplit(calls[-1][1]).query)
        self.assertIn('doc_type:full',query['q'][0]);self.assertEqual(query['rows'],['100']);self.assertEqual(query['sort'],['id asc']);self.assertIn('copyright',query['fl'][0])

    def test_cache_binds_url_and_bytes_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=p/'cached.response';meta=p/'cached.json';url='https://api.plos.org/search?q=fixture'
            raw.write_bytes(b'fixture');meta.write_text(json.dumps({'url':url,'sha256':a.sha(raw)}))
            c=a.Client(p);self.assertEqual(c.get('cached',url),b'fixture')
            with self.assertRaises(AssertionError):c.get('cached',url+'&changed=1')
            raw.write_bytes(b'corrupted')
            with self.assertRaises(AssertionError):c.get('cached',url)

    def run_fixture(self,docs,xml,discovery,ids):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);p=Path(directory.name)
        (p/'docs').mkdir();(p/'docs/replication_cohort_rules_v1.md').write_text('Synthetic fixed rule fixture\n')
        client=fake_client(docs,xml,discovery,ids)
        with patch.object(a,'Client',client),contextlib.redirect_stdout(io.StringIO()):a.run(p)
        return p/'replication/data',client.instance

    def test_rescue_left_join_denominators_dates_and_temporal_dedup(self):
        docs,xml,discovery,ids=test_corpus();p,client=self.run_fixture(docs,xml,discovery,ids)
        ledger={r['doi']:r for r in json.loads((p/'join_ledger.json').read_text())};manifest=json.loads((p/'cohort_manifest.json').read_text())
        self.assertEqual(len(ledger),8)
        self.assertIn('unmatched_pubmed_doi',ledger[ids[('PLOSGEN',2018)][0]]['exclusion_reasons'])
        undated=ledger[ids[('PLOSGEN',2019)][0]]
        self.assertIn('missing_electronic_date',undated['exclusion_reasons']);self.assertNotIn('unmatched_pubmed_doi',undated['exclusion_reasons'])
        self.assertEqual([v for source,v in client.fetches if source=='doi_rescue'],[(ids[('PLOSGEN',2019)][1],)])
        profile=next(r for r in manifest['profile'] if r['source']=='PLOSGEN' and r['year']==2020)
        self.assertEqual((profile['publisher_candidates'],profile['date_type_id_valid'],profile['valid_with_title'],profile['valid_with_both']),(1,1,1,0))
        self.assertEqual(profile['joint_availability_valid_frame'],0);self.assertEqual(profile['complete_coverage_publisher_frame'],0)
        self.assertEqual(profile['text_distribution_valid_frame']['abstract_chars']['n'],1)
        records=[json.loads(line) for line in (p/'records.jsonl').read_text().splitlines()]
        kept={r['id'] for r in records}
        self.assertIn(ids[('PLOSCB',2019)][0],kept) # Ordinary linked erratum retains original.
        self.assertNotIn(ids[('PLOSCB',2020)][0],kept) # RetractionIn excludes original.
        self.assertNotIn(ids[('PLOSCB',2021)][0],kept)
        duplicate=json.loads((p/'duplicate_exclusions.json').read_text())[0]
        self.assertEqual(duplicate['canonical_id'],ids[('PLOSCB',2018)][0])
        self.assertEqual(len(manifest['same_year_date_disagreements']),1)
        self.assertFalse(manifest['quality_pass']) # Tiny fixtures never authorize a real cohort.
        self.assertIn('doi_rescue.json',json.loads((p/'acquisition_manifest.json').read_text())['data_sha256'])

    def test_conflicting_pubmed_doi_cross_year_and_publisher_fragment_excluded(self):
        docs,xml,discovery,ids=test_corpus()
        d,p=ids[('PLOSCB',2019)];xml[p].find('./MedlineCitation/Article/ArticleDate/Year').text='2020'
        d,p=ids[('PLOSGEN',2021)];xml['999']=copy.deepcopy(xml[p]);xml['999'].find('./MedlineCitation/PMID').text='999';discovery[('PLOSGEN',2021)].append('999')
        folder,_=self.run_fixture(docs,xml,discovery,ids);ledger={r['doi']:r for r in json.loads((folder/'join_ledger.json').read_text())}
        self.assertIn('publisher_electronic_year_mismatch',ledger[ids[('PLOSCB',2019)][0]]['exclusion_reasons'])
        self.assertIn('multiple_pubmed_records_for_doi',ledger[ids[('PLOSGEN',2021)][0]]['exclusion_reasons'])
        docs,xml,discovery,ids=test_corpus();docs[('PLOSCB',2018)]['doc_type']='abstract'
        with self.assertRaisesRegex(AssertionError,'Publisher frame mismatch'):self.run_fixture(docs,xml,discovery,ids)

if __name__=='__main__':unittest.main(verbosity=2)
