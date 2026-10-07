"""Read-only metadata reconstruction. Standard library; never imports acquisition code.

Consumes saved PLOS JSON, PubMed XML and request receipts. Refuses incomplete runs.
No network, embeddings, novelty scores, or outcome files are accessed.
"""
import argparse
import collections
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import unicodedata
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

JOURNALS = {'PLOSCB': 'PLOS Computational Biology', 'PLOSGEN': 'PLOS Genetics'}
EXCLUDED = {'Review', 'Systematic Review', 'Meta-Analysis', 'Editorial', 'Letter',
            'Comment', 'News', 'Published Erratum', 'Retraction of Publication',
            'Retracted Publication', 'Expression of Concern', 'Corrected and Republished Article'}
PLACEHOLDERS = {'n/a', 'na', 'none', 'not available', 'no abstract available', 'abstract not available'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def norm(value):
    return ' '.join(unicodedata.normalize('NFC', html.unescape(value or '')).split())


def content(node):
    return norm(''.join(node.itertext())) if node is not None else ''


def normdoi(value):
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', norm(value).lower()).strip()


def meaningful(value):
    return bool(value) and value.casefold() not in PLACEHOLDERS


def parse_article(node):
    article = node.find('MedlineCitation/Article')
    date_values, malformed = [], False
    for element in article.findall('ArticleDate'):
        if element.attrib.get('DateType') != 'Electronic':
            continue
        try:
            date_values.append(dt.date(*(int(content(element.find(k))) for k in ('Year', 'Month', 'Day'))).isoformat())
        except (TypeError, ValueError):
            malformed = True
    if len(set(date_values)) > 1:
        date_error = 'ambiguous_electronic_date'
    elif malformed:
        date_error = 'malformed_electronic_date'
    elif not date_values:
        date_error = 'missing_electronic_date'
    else:
        date_error = None
    raw_dois = [content(e) for e in node.findall('PubmedData/ArticleIdList/ArticleId') if e.attrib.get('IdType') == 'doi']
    sections = [(norm(e.attrib.get('Label')), content(e)) for e in article.findall('Abstract/AbstractText')]
    title = content(article.find('ArticleTitle'))
    abstract = norm(' '.join((label + ': ' if label else '') + value for label, value in sections if value))
    author_details = [{k: content(a.find(k)) for k in ('LastName', 'ForeName', 'Initials', 'Suffix', 'CollectiveName')}
                      for a in article.findall('AuthorList/Author')]
    authors = [a['CollectiveName'] or ' '.join(v for v in (a['ForeName'] or a['Initials'], a['LastName'], a['Suffix']) if v)
               for a in author_details]
    return {
        'pmid': content(node.find('MedlineCitation/PMID')),
        'dois': sorted(set(map(normdoi, raw_dois))), 'raw_dois': raw_dois,
        'title': title, 'abstract': abstract,
        'has_title': meaningful(title), 'has_abstract': any(meaningful(value) for _, value in sections),
        'date': date_values[0] if date_error is None else None, 'date_error': date_error,
        'raw_electronic_dates': date_values,
        'journal': content(article.find('Journal/Title')),
        'publication_types': [content(e) for e in article.findall('PublicationTypeList/PublicationType')],
        'languages': [content(e) for e in article.findall('Language')],
        'correction_relationships': [{'ref_type': e.attrib.get('RefType'), 'pmid': content(e.find('PMID')),
                                      'ref_source': content(e.find('RefSource'))}
                                     for e in node.findall('MedlineCitation/CommentsCorrectionsList/CommentsCorrections')],
        'authors': authors, 'author_details': author_details,
        'copyright': content(article.find('Abstract/CopyrightInformation')),
    }


def audit(project, output):
    data = project / 'replication/data'
    progress = load(data / 'progress.json')
    if progress.get('stage') != 'complete':
        raise SystemExit('Metadata acquisition is incomplete; no audit executed.')
    acquisition = load(data / 'acquisition_manifest.json')
    cohort = load(data / 'cohort_manifest.json')
    errors, checks = [], collections.Counter()

    def check(ok, label, detail=None):
        checks[label] += 1
        if not ok:
            errors.append({'check': label, 'detail': detail})

    hashes = {name: digest(data / name) for name in acquisition['data_sha256']}
    check(hashes == acquisition['data_sha256'], 'derived_manifest_hashes')
    check(digest(project / 'replication/acquire.py') == acquisition['code_sha256'], 'acquisition_code_hash')
    rules_hash = digest(project / 'docs/replication_cohort_rules_v1.md')
    check(acquisition['rules_sha256'] == cohort['rules_sha256'] == rules_hash, 'rules_hash')
    check(hashes['records.jsonl'] == cohort['records_sha256'], 'cohort_records_hash')

    publisher_pages = collections.defaultdict(list)
    search_ids, fetched_ids = {}, collections.defaultdict(list)
    pubmed, initial_pmids, rescue_pmids = {}, set(), set()
    receipts, rescue_searches = [], []
    for call in acquisition['calls']:
        label = call['label']
        raw = data / 'raw' / (label + '.response')
        meta = load(data / 'raw' / (label + '.json'))
        body = raw.read_bytes()
        check(meta == call, 'raw_receipt_manifest_equality', label)
        check(hashlib.sha256(body).hexdigest() == call['sha256'] and len(body) == call['bytes'], 'raw_bytes_hash_size', label)
        check(call['status'] == 200, 'http_status', label)
        url = urlsplit(call['url'])
        params = parse_qs(url.query)
        receipts.append({'label': label, 'host': url.hostname, 'retrieved_utc': call['retrieved_utc'], 'sha256': call['sha256']})
        if label.startswith('plos_'):
            _, source, year, offset = label.split('_')
            page = json.loads(body)['response']
            check('doc_type:full' in params['q'][0] and 'article_type:"Research Article"' in params['q'][0], 'publisher_query_types', label)
            check(params['rows'] == ['100'] and params['sort'] == ['id asc'] and params['start'] == [str(int(offset))], 'publisher_query_pagination', label)
            publisher_pages[(source, int(year))].append((int(offset), page))
        elif 'esearch.fcgi' in url.path:
            result = json.loads(body)['esearchresult']
            ids = result['idlist']
            check(int(result['count']) == len(ids) == len(set(ids)), 'search_complete_unique', label)
            search_ids[label] = set(ids)
            if label.startswith('pubmed_rescue_ids_'):
                dois = re.findall(r'"([^\"]+)"\[aid\]', params['term'][0])
                check(0 < len(dois) <= 20 and params['term'][0] == ' OR '.join('"' + d + '"[aid]' for d in dois), 'rescue_query_date_free', label)
                rescue_searches.append({'dois': dois, 'returned_pmids': ids, 'offset': int(label.rsplit('_', 1)[1])})
        elif 'efetch.fcgi' in url.path:
            xml = ET.fromstring(body)
            articles = [parse_article(e) for e in xml.findall('PubmedArticle')]
            expected = params['id'][0].split(',')
            actual = [r['pmid'] for r in articles]
            check(sorted(actual) == sorted(expected) and len(actual) == len(set(actual)) and len(actual) <= 200,
                  'fetched_ids_complete_unique', label)
            rescue_batch = label.startswith('pubmed_xml_doi_rescue_')
            if rescue_batch:
                rescue_pmids.update(actual)
            else:
                initial_pmids.update(actual)
                match = re.fullmatch(r'pubmed_xml_(PLOSCB|PLOSGEN)_(\d{4})_\d+', label)
                check(match is not None, 'fetch_label_pattern', label)
                if match:
                    fetched_ids[(match[1], int(match[2]))].extend(actual)
            for row in articles:
                if row['pmid'] in pubmed:
                    check(pubmed[row['pmid']] == row, 'repeated_pubmed_record_consistency', row['pmid'])
                pubmed[row['pmid']] = row
        else:
            check(False, 'unexpected_acquisition_endpoint', label)

    frame = []
    for year in range(2018, 2022):
        for source in JOURNALS:
            pages = sorted(publisher_pages[(source, year)])
            check(bool(pages), 'publisher_cell_present', [source, year])
            total, ids = pages[0][1]['numFound'], []
            for start, page in pages:
                check(page['numFound'] == total and page['start'] == start == len(ids), 'publisher_page_start_total', [source, year, start])
                check(len(page['docs']) == min(100, total-start), 'publisher_page_cardinality', [source, year, start])
                for doc in page['docs']:
                    d = dt.date.fromisoformat(doc['publication_date'][:10])
                    identifier = normdoi(doc['id'])
                    check(bool(identifier) and doc['journal'] == JOURNALS[source] and doc['doc_type'] == 'full'
                          and doc['article_type'] == 'Research Article' and d.year == year,
                          'publisher_record_domain', identifier)
                    ids.append(identifier)
                    frame.append({'doi': identifier, 'source': source, 'year': year, 'publisher_date': str(d), 'publisher': doc})
            check(len(ids) == len(set(ids)) == total, 'publisher_cell_count_unique', [source, year])
            expected = search_ids['pubmed_ids_%s_%s' % (source, year)]
            actual = fetched_ids[(source, year)]
            check(len(actual) == len(set(actual)) and set(actual) == expected, 'search_fetch_cell_completeness', [source, year])
    check(len(frame) == len({r['doi'] for r in frame}), 'publisher_global_doi_uniqueness')
    check(frame == load(data / 'publisher_frame.json'), 'publisher_frame_raw_reconstruction')

    initial_by_doi = collections.defaultdict(list)
    by_doi = collections.defaultdict(list)
    for row in pubmed.values():
        for identifier in row['dois']:
            by_doi[identifier].append(row)
            if row['pmid'] in initial_pmids:
                initial_by_doi[identifier].append(row)
    initial_unmatched = sorted(r['doi'] for r in frame if not initial_by_doi[r['doi']])
    rescue_searches.sort(key=lambda r: r['offset'])
    rescue_union = set().union(*(set(r['returned_pmids']) for r in rescue_searches))
    check([d for batch in rescue_searches for d in batch['dois']] == initial_unmatched, 'rescue_all_initially_unmatched')
    check(rescue_pmids == rescue_union - initial_pmids and not rescue_pmids.intersection(initial_pmids), 'rescue_only_new_complete_fetch')
    rescue_expected = {'initial_unmatched_dois': initial_unmatched,
                       'lookups': [{k: r[k] for k in ('dois', 'returned_pmids')} for r in rescue_searches],
                       'new_pmids_fetched': sorted(rescue_pmids),
                       'unmatched_after_rescue': [d for d in initial_unmatched if not by_doi[d]]}
    check(load(data / 'doi_rescue.json') == rescue_expected, 'rescue_ledger_raw_reconstruction')

    rows, eligible, disagreements = [], [], []
    for item in frame:
        hits = by_doi[item['doi']]
        r = hits[0] if len(hits) == 1 else None
        failures = []
        if not hits:
            failures.append('unmatched_pubmed_doi')
        elif len(hits) != 1:
            failures.append('multiple_pubmed_records_for_doi')
        if r:
            if len(r['dois']) != 1:
                failures.append('conflicting_pubmed_dois')
            if r['journal'].casefold() != JOURNALS[item['source']].casefold():
                failures.append('journal_mismatch')
            if r['date_error']:
                failures.append(r['date_error'])
            elif int(r['date'][:4]) != item['year']:
                failures.append('publisher_electronic_year_mismatch')
            elif r['date'] != item['publisher_date']:
                disagreements.append({'doi': item['doi'], 'publisher': item['publisher_date'], 'electronic': r['date']})
            if 'Journal Article' not in r['publication_types']:
                failures.append('not_pubmed_journal_article')
            if EXCLUDED.intersection(r['publication_types']):
                failures.append('excluded_publication_type')
            if any(rel['ref_type'] in {'RetractionIn', 'ExpressionOfConcernIn'} for rel in r['correction_relationships']):
                failures.append('retraction_or_concern')
            if r['languages'] != ['eng']:
                failures.append('nonunique_english_language')
        valid = not failures
        has_title, has_abstract = (r['has_title'], r['has_abstract']) if r else (False, False)
        if r and not has_title:
            failures.append('missing_title')
        if r and not has_abstract:
            failures.append('missing_abstract')
        expected = {'doi': item['doi'], 'source': item['source'], 'publisher_year': item['year'],
                    'publisher_date': item['publisher_date'], 'matched_pmids': [h['pmid'] for h in hits],
                    'date_type_id_valid': valid, 'has_title': has_title, 'has_abstract': has_abstract,
                    'exclusion_reasons': failures}
        if r:
            expected.update(electronic_date=r['date'], publication_types=r['publication_types'], languages=r['languages'],
                            correction_relationships=r['correction_relationships'], raw_pubmed_dois=r['raw_dois'],
                            title_chars=len(r['title']), abstract_chars=len(r['abstract']),
                            title_tokens=len(r['title'].split()), abstract_tokens=len(r['abstract'].split()))
        rows.append(expected)
        if not failures:
            eligible.append((item, r))
    check(rows == load(data / 'join_ledger.json'), 'join_ledger_raw_reconstruction')
    canonical, duplicates, retained = {}, [], []
    for item, r in sorted(eligible, key=lambda pair: (pair[1]['date'], pair[0]['doi'])):
        key = (r['title'], r['abstract'])  # Compare actual normalized text, not just exported hashes.
        if key in canonical:
            duplicates.append({'id': item['doi'], 'canonical_id': canonical[key], 'reason': 'exact_title_abstract_duplicate'})
        else:
            canonical[key] = item['doi']
            retained.append((item, r))
    check(duplicates == load(data / 'duplicate_exclusions.json') == cohort['new_duplicate_exclusions'], 'exact_text_dedup_and_canonical')
    actual_records = [json.loads(line) for line in (data / 'records.jsonl').read_text().splitlines() if line.strip()]
    actual_by_id = {r['id']: r for r in actual_records}
    check(len(actual_records) == len(actual_by_id) == len(retained) and set(actual_by_id) == {i['doi'] for i, r in retained}, 'retained_membership_unique')
    rights = collections.Counter()
    for item, r in retained:
        exported = actual_by_id.get(item['doi'], {})
        expected = {'id': item['doi'], 'doi': item['doi'], 'pmid': r['pmid'], 'year': item['year'], 'source': item['source'],
                    'title': r['title'], 'abstract': r['abstract'], 'electronic_date': r['date'],
                    'publisher_date': item['publisher_date'], 'authors': r['authors'], 'author_details': r['author_details'],
                    'publisher_authors': item['publisher'].get('author_display', []), 'copyright': r['copyright'],
                    'publisher_copyright': item['publisher'].get('copyright'),
                    'source_url': 'https://pubmed.ncbi.nlm.nih.gov/' + r['pmid'] + '/',
                    'publisher_url': 'https://doi.org/' + item['doi'], 'publication_types': r['publication_types'], 'languages': r['languages']}
        for field in ('title', 'abstract'):
            expected[field + '_sha256'] = hashlib.sha256(r[field].encode()).hexdigest()
        check(all(exported.get(k) == v for k, v in expected.items()), 'retained_text_metadata_provenance', item['doi'])
        check(exported.get('topic') == 0, 'unused_topic_constant', item['doi'])
        rights['retained_records'] += 1
        rights['publisher_authors_present'] += bool(expected['publisher_authors'])
        rights['pubmed_authors_present'] += bool(expected['authors']) and all(expected['authors'])
        rights['publisher_copyright_present'] += bool(expected['publisher_copyright'])
        rights['pubmed_copyright_present'] += bool(expected['copyright'])
        rights['publisher_creative_commons_attribution_text'] += 'creative commons attribution' in str(expected['publisher_copyright']).casefold()
        rights['publisher_cc0_public_domain_text'] += 'creative commons cc0 public domain dedication' in str(expected['publisher_copyright']).casefold()
        rights['publisher_explicit_4_0_text'] += bool(re.search(r'(?:4\.0|by/4)', str(expected['publisher_copyright'])))
        rights['publisher_rights_text_unclassified'] += not any(term in str(expected['publisher_copyright']).casefold() for term in ('creative commons attribution', 'creative commons cc0 public domain dedication'))
    check(all(r['year'] in {2018, 2019, 2020, 2021} and int(r['electronic_date'][:4]) == r['year'] for r in actual_records), 'retained_chronology')
    check(actual_records == sorted(actual_records, key=lambda r: (r['year'], r['source'], r['id'])), 'record_order')

    profile, losses_by_cell = [], {}
    for year in range(2018, 2022):
        for source in JOURNALS:
            cell = [r for r in rows if (r['source'], r['publisher_year']) == (source, year)]
            valid = [r for r in cell if r['date_type_id_valid']]
            both = sum(r['has_title'] and r['has_abstract'] for r in valid)
            stats = {'year': year, 'source': source, 'publisher_candidates': len(cell),
                     'doi_matched_unique': sum(len(r['matched_pmids']) == 1 for r in cell),
                     'date_type_id_valid': len(valid), 'valid_with_title': sum(r['has_title'] for r in valid),
                     'valid_with_abstract': sum(r['has_abstract'] for r in valid), 'valid_with_both': both,
                     'joint_availability_valid_frame': both/len(valid) if valid else 0,
                     'complete_coverage_publisher_frame': both/len(cell) if cell else 0,
                     'eligible_final': sum((r['source'], r['year']) == (source, year) for r in actual_records)}
            saved = next(r for r in cohort['profile'] if (r['source'], r['year']) == (source, year))
            check(all(saved.get(k) == v for k, v in stats.items()), 'cell_denominators_and_coverage', [source, year])
            distribution = {}
            for field in ('title_chars', 'abstract_chars', 'title_tokens', 'abstract_tokens'):
                vals = [r[field] for r in valid]
                distribution[field] = {'n': len(vals), 'min': min(vals) if vals else None,
                                       'median': statistics.median(vals) if vals else None, 'max': max(vals) if vals else None}
            check(distribution == saved['text_distribution_valid_frame'], 'text_distribution', [source, year])
            stats['gate_pass'] = stats['joint_availability_valid_frame'] >= .95 and stats['complete_coverage_publisher_frame'] >= .95
            profile.append(stats)
            losses_by_cell[source + '_' + str(year)] = dict(collections.Counter(reason for row in cell for reason in row['exclusion_reasons']))
    focal = collections.Counter(r['source'] for r in actual_records if r['year'] == 2021)
    history = sum(r['year'] < 2021 for r in actual_records)
    gate = all(r['gate_pass'] for r in profile) and all(focal[s] >= 200 for s in JOURNALS) and history >= 2000
    check(cohort['quality_pass'] == progress['quality_pass'] == gate, 'quality_gate_exact')
    check(cohort['focal_count'] == sum(focal.values()) and cohort['history_count'] == history
          and cohort['publisher_frame_count'] == len(frame) and cohort['retrieved_pubmed_records'] == len(pubmed), 'manifest_summary_counts')
    reasons = dict(collections.Counter(reason for row in rows for reason in row['exclusion_reasons']))
    check(reasons == cohort['exclusions'], 'exclusion_reason_counts')
    check(disagreements == cohort['same_year_date_disagreements'], 'same_year_date_disagreement_ledger')
    for section, selected in [('history', [r for r in actual_records if r['year'] < 2021]), ('focal', [r for r in actual_records if r['year'] == 2021])]:
        keys = ['id', 'doi', 'pmid', 'year', 'source', 'topic', 'electronic_date', 'publisher_date', 'title_sha256', 'abstract_sha256']
        check(cohort[section] == [{k: r[k] for k in keys} for r in selected], 'manifest_membership_' + section)
    retained_types = collections.Counter(t for r in actual_records for t in r['publication_types'])
    excluded_rows = [r for r in rows if r['exclusion_reasons']]
    excluded_type_counts = collections.Counter(t for r in excluded_rows for t in r.get('publication_types', []) if t in EXCLUDED)
    exclusion_patterns = collections.Counter(' + '.join(r['exclusion_reasons']) for r in excluded_rows)
    rights_inventory = collections.Counter(r['publisher_copyright'] for r in actual_records)
    report = {
        'audit_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'project': str(project),
        'auditor': 'Independent standard-library raw-response reconstruction; acquisition module not imported',
        'no_network_no_novelty_outcomes': True, 'audit_integrity_pass': not errors, 'errors': errors,
        'checks': dict(checks), 'cohort_gate_pass': gate, 'gate_thresholds_unchanged': True,
        'profile': profile, 'history_count': history, 'focal_by_source': dict(focal),
        'exclusion_reason_counts': reasons, 'exclusion_reasons_by_cell': losses_by_cell,
        'excluded_pubmed_type_counts': dict(excluded_type_counts), 'mutually_exclusive_reason_patterns': dict(exclusion_patterns),
        'excluded_publisher_records': sum(bool(r['exclusion_reasons']) for r in rows),
        'exact_text_duplicate_exclusions': duplicates, 'same_year_date_disagreement_count': len(disagreements),
        'rescue': {k: v for k, v in rescue_expected.items() if k != 'lookups'},
        'rescue_query_batches': len(rescue_searches), 'pubmed_unique_records': len(pubmed),
        'retained_publication_types': dict(retained_types), 'rights_provenance_counts': dict(rights),
        'publisher_rights_statement_distinct_count': len(rights_inventory),
        'retained_language_states': dict(collections.Counter('|'.join(r['languages']) for r in actual_records)),
        'rights_limit': 'Item statements and attribution retained; no redistribution authorized. No license version inferred when item text does not specify it.',
        'raw_receipt_count': len(receipts), 'retrieval_utc_min': min(r['retrieved_utc'] for r in receipts),
        'retrieval_utc_max': max(r['retrieved_utc'] for r in receipts),
        'data_sha256': hashes, 'rules_sha256': rules_hash, 'acquisition_code_sha256': acquisition['code_sha256'],
        'auditor_code_sha256': digest(__file__),
        'limitations': ['Current snapshots do not reconstruct historical visibility.',
                        'DOI rescue targets initially unmatched publisher records, not undiscovered duplicates of already matched records.',
                        'Exclusion reasons overlap; their sum is not a distinct-record loss count.',
                        'No biomedical score or embedding was accessed.']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'independent_metadata_audit.json').write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    lines = ['# Independent biomedical metadata audit', '',
             'Saved-source audit; no API calls or novelty outcomes. Acquisition code was not imported.', '',
             '**Integrity reconstruction: %s. Metadata gate: %s.**' % ('PASS' if not errors else 'FAIL', 'PASS' if gate else 'FAIL'), '',
             '| Journal | Year | Publisher frame | Unique DOI match | Date/type/ID valid | Complete before dedup | Valid-frame availability | Publisher coverage | Final | Gate |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in profile:
        lines.append('| {source} | {year} | {publisher_candidates} | {doi_matched_unique} | {date_type_id_valid} | {valid_with_both} | {va:.3%} | {pc:.3%} | {eligible_final} | {g} |'.format(**r, va=r['joint_availability_valid_frame'], pc=r['complete_coverage_publisher_frame'], g='PASS' if r['gate_pass'] else 'FAIL'))
    lines += ['', 'Historical retained papers: **%s**. Focal retained by journal: **%s**.' % (history, dict(focal)), '',
              'The gate retains both 95% per-cell thresholds, at least 200 focal papers in each journal, and at least 2,000 historical papers. A failed gate is preserved; this audit does not lower it.', '',
              'Distinct excluded publisher records before exact-text dedup: %s. Exclusion reasons (may overlap): `%s`.' % (report['excluded_publisher_records'], json.dumps(reasons, sort_keys=True)), '',
              'Excluded publication types: `%s`. Distinct reason combinations: `%s`.' % (json.dumps(dict(excluded_type_counts), sort_keys=True), json.dumps(dict(exclusion_patterns), sort_keys=True)), '',
              'Unique DOI matches: %s/%s publisher candidates. Missing/malformed/ambiguous/cross-year electronic date exclusions: %s; language exclusions: %s; missing Journal Article type exclusions: %s. Every retained record is English and has the Journal Article type. Meta-analysis/review/retraction classifications follow the declared exclusions even when PLOS labels the item a Research Article.' % (sum(r['doi_matched_unique'] for r in profile), len(frame), sum(reasons.get(k, 0) for k in ('missing_electronic_date', 'malformed_electronic_date', 'ambiguous_electronic_date', 'publisher_electronic_year_mismatch')), reasons.get('nonunique_english_language', 0), reasons.get('not_pubmed_journal_article', 0)), '',
              'Exact-text duplicate exclusions: %s. Same-year publication-date differences: %s.' % (len(duplicates), len(disagreements)), '',
              'DOI rescue: %s initially unmatched DOIs; %s date-free search batches; %s new PMIDs fetched; %s DOIs still unmatched.' % (len(initial_unmatched), len(rescue_searches), len(rescue_pmids), len(rescue_expected['unmatched_after_rescue'])), '',
              'Rights/provenance counts: `%s`. Original item statements, author names and source links were compared against raw responses. A generic Creative Commons Attribution statement does not itself establish a numbered license version. No redistribution is authorized.' % json.dumps(dict(rights), sort_keys=True), '',
              'All raw request receipts and response hashes, publisher pagination/membership, discovery/fetch/rescue IDs, PubMed chronology and exclusions, exact normalized scoring text, cross-year canonical duplicates, both completeness denominators, retained attribution, and exported manifest memberships were independently reconstructed. Full checks, cell-level loss reasons and SHA-256 values are in `independent_metadata_audit.json`.', '',
              'Audit discrepancies: %s.' % len(errors), '',
              'Limits: current snapshots are not historical visibility; unmatched-only rescue is not exhaustive bibliographic deduplication; overlapping exclusion counts must not be summed as distinct papers. No biomedical outcomes or embeddings were inspected.', '',
              'Sources: saved `replication/data/raw` response/receipt pairs, publisher frame, DOI-rescue and join ledgers, records, cohort/acquisition manifests, and `docs/replication_cohort_rules_v1.md`.']
    if errors:
        lines += ['', '## Integrity discrepancies', '', '```json', json.dumps(errors, indent=2, ensure_ascii=False), '```']
    (output / 'independent_metadata_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ('audit_integrity_pass', 'cohort_gate_pass', 'history_count', 'focal_by_source', 'errors', 'exclusion_reason_counts', 'rights_provenance_counts')}, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.project, args.output))
