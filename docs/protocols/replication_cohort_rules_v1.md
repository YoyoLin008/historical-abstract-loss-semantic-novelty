# Biomedical metadata eligibility rules v1 — fixed before bulk retrieval

Prepared 2026-10-04. This is a prospective metadata feasibility rule, before biomedical novelty scores. Root may revise it only before scores and must retain a dated reason. Counts-only and two-record schema probes preceded this rule; no cohort bulk metadata has been retrieved by this agent.

## Exact target frame

One research paper is one normalized DOI. Source groups are exactly PLOS Computational Biology (`PLOSCB`) and PLOS Genetics (`PLOSGEN`). Use publisher PLOS API records with `doc_type:full` and **exact returned** `article_type == "Research Article"`. A query without `doc_type:full` also returns article fragments and must never define counts or IDs. Exclude all other publisher types: reviews, corrections, editorial, education, perspective, correspondence, opinion, interviews, front matter, issue images and any unrecognized type. Research articles about methods/software are retained when the publisher labels them Research Article; no topic selection.

Retrieve the publisher research-article DOI frame for each of2018,2019,2020,2021 using `publication_date` bounded to the calendar year. This is a current index snapshot, not reconstructed historical metadata availability. Retain publisher counts and every candidate DOI before completeness exclusions. Sort by `id asc`, pages at most100 rows, at least6.2seconds between starts, no concurrent requests. Check every page's numFound, exact DOI uniqueness, returned date/year/journal/type/doc_type and consistent total; stop on a moving total or inconsistent pagination.

## Date definition

For each DOI matched to PubMed, read `MedlineCitation/Article/ArticleDate[@DateType="Electronic"]` and require a valid full Gregorian year-month-day. If multiple such dates exist, require all to be identical after parsing; otherwise quarantine as ambiguous. The **electronic publication date** defines the analytic year. Require its year to equal the publisher `publication_date` year. Record both exact dates and any day/month disagreement; same-year day differences do not alter the year-based design. Exclude missing, malformed, ambiguous or cross-year dates with reasons and counts, rather than imputing January1 or substituting issue/Entrez/indexing/MeSH/accepted dates. The accepted target is therefore the explicitly concordant publisher/PubMed electronic-year frame, not every paper the journals may have published.

Focal cohort: all eligible2021 papers, no outcome-dependent sampling. History: all eligible2018–2020 papers from both journals; every history year is strictly earlier than2021. Do not mix same-year predecessors into history. PubMed discovery uses `[EPDAT]`; `[DP]` includes print/electronic dates and is unsuitable as the sole chronology definition. Indexing and revision dates are provenance only.

## Exact DOI join and article exclusions

Normalize DOI by Unicode/HTML decoding, whitespace trim, lower case and removal of a leading DOI URL or `doi:` label; do not remove internal punctuation. Extract DOI from `PubmedData/ArticleIdList/ArticleId[@IdType="doi"]`; retain original DOI/PMID/source strings. Match the publisher frame by exact normalized DOI. Quarantine DOI collisions, several conflicting PubMed records for one DOI, unexpected journal mismatch, and publisher DOI/year duplicates. Do not fuzzy-match titles or invent identifiers. Retain a left-join ledger of all publisher candidates, including missing PubMed records.

Exclude non-JournalArticle PubMed records, publication types explicitly identifying reviews/editorials/letters/comments/news/corrections/retraction notices, and records flagged as retracted publications or expressions of concern. Record every publication type and correction relationship. Original papers with ordinary linked errata remain eligible; the correction notices themselves are excluded by publisher type. Exclude records with `RetractionIn` or `ExpressionOfConcernIn` relationships. These are current-snapshot quality exclusions, not claims about knowledge available in2021. Retain English records (`Language=eng`); log unknown/multiple language states. Do not silently use PubMed JournalArticle as proof of the publisher ResearchArticle category.

## Text construction and completeness

Use PubMed `ArticleTitle` and `Abstract/AbstractText` only. Preserve inline text with itertext; collapse whitespace deterministically. Join structured abstract sections in their XML order, retaining each supplied section Label followed by its text. Do not append publisher author summaries, keywords, affiliations, or the copyright statement to the abstract. Preserve the original XML bytes separately. Normalize to NFC for the exported scoring text and hash it; avoid case-folding the actual encoder input.

Define nonempty substantive title and abstract before complete-case exclusion; reject empty/whitespace-only strings and exact placeholders `N/A`, `NA`, `none`, `not available`, `no abstract available`, `abstract not available` case-insensitively. Record token/character distributions and very short texts for review without selecting by future outcomes. Audit exact normalized title+abstract duplicates across all dates and journals and retain a deterministic exclusion ledger (earliest electronic date, then normalized DOI as canonical). Repeated titles alone are not sufficient to merge different papers. No PDF extraction or imputation.

## Acceptance rule and denominator

Require at least200 complete eligible2021 papers in each journal and at least2,000 complete eligible historical papers combined. Compute title availability, abstract availability and joint availability **before text-completeness exclusions**, per journal-year, in the date/type/ID-valid joined frame. Also report the stricter completeness fraction relative to the entire publisher ResearchArticle frame, so PubMed join losses cannot be concealed. Prefer at least95% joint title/abstract availability in each journal-year and at least95% publisher-frame coverage; failure is a metadata gate to resolve before freezing biomedical outcomes. Do not use the post-exclusion complete cohort as its own completeness denominator.

The proposed outcome design remains all eligible papers, G/T/Q and cutoff checks,25%/50% random and source-preferred masks, PLOSCB preference8:1,200base masks/condition, year-matched source controls, independent new seeds. This file authorizes no biomedical scores or public redistribution.

## Rights and retrieval boundary

NCBI publicly permits low-rate E-utilities access without an API key. Use at most2requests/second and batch retrieval; no email or API key is fabricated. PLOS Search API has a stricter10requests/minute,300/hour,7,200/day limit and at most100rows per request. NLM does not own publishers' abstract copyrights. PLOS's current policy permits mining and reuse under CC BY4.0 or equivalently permissive licenses with author/source attribution; retain publisher copyright/license statements, author list, DOI and source links. No external release is currently authorized. Before any later abstract/embedding redistribution, retain item-level rights/provenance; identifiers, hashes, queries and non-text results are the default public package.
