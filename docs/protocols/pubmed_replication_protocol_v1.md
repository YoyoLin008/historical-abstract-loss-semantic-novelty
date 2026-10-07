# PubMed/PLOS replication v1 — prospective protocol

Prepared 2026-10-04 after the completed, frozen ACL confirmation. No biomedical novelty outcomes have been inspected. This is an independent-collection replication, not an externally registered preregistration or independent validation of the encoder. The same SPECTER revision is intentionally retained. No second encoder or diagnostic model search is authorized by this protocol.

## Question and locked scope

Does the aggregate-versus-ranking/selection stability distinction observed in the ACL study recur in a separately acquired biomedical collection under the same operational thresholds and a limited mask design? The second collection is not chosen by its observed novelty response.

Use all eligible complete original research articles in PLOS Computational Biology (`PLOSCB`) and PLOS Genetics (`PLOSGEN`): 2021 focal cohort, 2018–2020 strictly prior history. Retain all eligible historical records and all eligible focal records. The metadata audit accepted on 2026-10-05 retains 1,471 focal papers (934 PLOSCB; 537 PLOSGEN) and 3,391 historical papers from 4,896 publisher candidates. The 34 distinct type/retraction/concern exclusions overlap across reason categories. There are no DOI, date or text-completeness losses or exact text duplicates. All eight journal-year cells have 100% joint text availability in the eligible joined frame and 98.168–100% complete publisher-frame coverage. Ordered IDs, original dates, normalized text hashes and exclusions are bound by companion manifests; the data gate passes unchanged. The baseline top-decile denominator will be 148, computed as ceil(0.1 × 1471). The scorer refuses a missing date definition or inconsistent frozen ID/count manifests. PubMed `Journal Article` alone is not sufficient evidence of an original research article; the acquisition audit must state and verify the publisher/PMC type rule.

Do not change years or choose between the journals based on novelty outcomes. A source/date/license barrier is resolved prospectively in an explicit amendment or blocks the replication. Existing papers' public metadata/abstracts are the records; no patient-level or privileged clinical dataset is used.

Freeze source requests/raw-file hashes, acquisition code and manifest, PubMed/DOI/article identifiers, date fields and chosen year, type/missingness/exclusion states, title/abstract hashes, all ordered IDs, source labels, embedding manifest v2, model/tokenizer file hashes, dependency lock, config, protocol, seed list, scoring code and analysis code. Preserve the completed ACL source inventory and outputs. New code lives under `replication/` and imports unchanged low-level numerical utilities from `src/`.

## Four conditions, no search

Rates are 25% and 50% historical abstract masking; zero masking is a deterministic reference. Both are synthetic interventions, with 50% explicitly a stress test. There is no claim that these rates reproduce real PubMed omissions.

At each rate use uniform random masking and source-preferred masking. The preferred source is fixed prospectively to PLOS Computational Biology, with weight 8 versus 1; use the same weighted-without-replacement exponential-clock method as ACL. Exact mask count is `floor(n_history × rate + 0.5)`. The field `topic=0` may be carried solely for compatibility with the shared record validator; no topic labels, topic mask or topic effect is defined in this replication.

For each source-preferred mask, draw one independently seeded uniform control with exactly the same missing count in every historical year. This tests source location and its associated within-year content differences beyond historical age composition. Do not match source counts in that control, because source preference is the intended contrast. All focal records are from the same later year, so all declared history is eligible for every focal paper. Report exact control cells and degeneracy.

Exactly 200 masks per rate/mechanism: **800 base masks and 400 matched controls**. Master seed `2785196430`, namespace `pubmed-2021-replication-v1`. Store all 1,200 independently derived 128-bit seed vectors before outcomes; exclude the earlier pilot/confirmation master seeds. No adaptive extension or favorable condition selection.

## Same measurement, thresholds and cutoff rules

The primary policy is fixed full title-plus-abstract focal embeddings and mixed full/title-only history. Complete-case historical removal, all-title on both sides, and authentic restoration are unchanged ablations/checks. Score `1 − max earlier cosine` exactly, excluding self/non-earlier pairs; undefined histories remain undefined.

Let b be the complete selected-collection baseline, s0 its all-focal population SD (`ddof=0`), and A/B the fixed `PLOSCB`/`PLOSGEN` groups. The three primary endpoints remain:

- **G:** absolute mask-specific change in the A-minus-B mean score contrast, divided by s0; then average over complete masks.
- **T:** fraction of baseline top-decile papers lost, with `k = ceil(0.10 × n_focal)` recomputed for this collection. Sort scores descending with lexical ID breaking exact selection ties. Never reuse ACL's k=156.
- **Q:** mean absolute change in average score rank divided by `n_focal−1`; exact score ties receive average ranks, independent of lexical ID.

Use tolerances G=0.10 SD, T=0.05 of the original k selections, Q=0.01 of rank range. They are the same unvalidated operational choices as ACL, not disciplinary standards. A condition supports the joint mean-level claim only when `U_G < 0.10` and `L_T > 0.05`; the additional `L_Q > 0.01` criterion is reported separately. Equality/touching is inconclusive. Do not compare the units by ratio. Report baseline journal contrast, signed drift, mean intervals, mask SD and distribution quantiles, and counts lost out of k.

Apply the unchanged predeclared threshold grid: G `{0.05,0.10,0.15}`, T `{0.025,0.05,0.10}`, Q `{0.005,0.01,0.02}`. The descriptive proportion of masks jointly satisfying G<0.10 and T>0.05 gets a pointwise 95% Wilson interval, explicitly outside the primary mean family. Mean-level coexistence does not imply coexistence in most masks or stability of every paper.

Mandatory margin analysis uses the midpoint of the baseline kth and (k+1)th scores, their gap in raw/SD units, and the exact cutoff tie block. The cutoff band contains descending ordinal positions r whose `(r−0.5)/n` lies in inclusive `[0.09,0.11]`, expanded through exact-score ties at both edges. Signed score-margin bins have edges `−infinity, −.10, −.05, −.025, −.01, 0, .01, .025, .05, .10, infinity`, left-inclusive/right-exclusive; zero is in `[0,.01)`. Report empty bins and mask-level retention/entry probabilities.

For every mask report baseline selections lost inside and outside that band, each divided by k; the loss rate among selections outside the band gets its own stated denominator. Far-margin loss is `count{i in S0 minus Sm : (b_i−c0)/s0 > delta}/k` for delta `{.01,.025,.05,.10}`. Report rank-movement median, 90th percentile, and fraction moving more than five percentile points. Near-boundary exchange alone is not broad rank disorder or a change in true scientific originality.

## Eighteen-mean inference family

Use one approximate simultaneous 95% family: 12 base-condition means (G/T/Q × random/source × 25%/50%) plus six paired source-minus-year-matched-control means. Every complete mask or structured/control pair is one intervention unit. Do not use iid paper-level inference.

For each mean use `mean ± t_(199,1−.05/(2×18)) × sample_SD/sqrt(200)`, with critical value **3.029133739375753**. The readout verifies this value with pinned mpmath's Student-t tail. Bonferroni does not require endpoint independence; the individual Student intervals still rely on a large-sample approximation for bounded/skewed mask outcomes. Retain untruncated intervals and do not claim exact finite-sample coverage. Primary structured effects use each pair's endpoint difference before estimating mean/SD/SE. Report signed differences even if source preference reduces instability.

This family concerns the new corpus alone. Do not pool the prior ACL experiment and replication into a retrospectively chosen overall test. Compare patterns and uncertainty across the two frozen studies descriptively. No success requirement is imposed: a different direction, smaller effect, or absence of joint coexistence is informative and must be retained.

## Outputs, tests and stop conditions

The readout produces cohort, aggregate, rank/selection and paired tables; cutoff/movement supplements; threshold and descriptive joint-event tables; and three figures (G/T/Q panels, mask-level G/T scatter, margin retention/entry). Figure units and k are computed from this cohort, not copied from ACL. No predictive diagnostic is fitted.

Before outcome execution require both the accepted data/rights audit and a new protocol freeze manifest containing an actual Git commit and all content hashes. Use bounded checkpoint/resume, new output directories, and clean reproduction. Tests must cover new source names, both rates, year-only controls, all-focal selection, seed cardinality/uniqueness/exclusions, frozen biomedical dates/IDs, complete provenance, n-dependent selection denominators and cutoff ties, the 18-family critical value and table cardinality, restoration/all-title checks, and interrupted/clean equality on synthetic data. Root must independently inspect rendered synthetic figures before treating the analysis as ready.

Actual data counts, research-article inclusion rules, date definition and access findings were accepted prospectively on 2026-10-05 (docs/pubmed_data_quality_gate.md). Empirical execution still requires completed encoding and a new companion freeze manifest with the actual Git commit, time and content hashes. A scientific disagreement with the ACL pattern does not authorize changing those rules or searching more datasets until a favorable replication appears.

## Encoding audit before freeze

All 4,862 full/title representations completed locally, including a deliberate two-chunk checkpoint and resume. The fixed tokenizer/encoder cap truncates 67 full inputs (1.378%) and no title-only inputs; no record is excluded for length. This source-dependent limitation is recorded in results/pubmed_input_token_audit.json and reconciled with the actual completed chunk manifest. Full/title array shape, float32 values, finite entries, unit norms and output hashes were checked before scores. The same pinned model, numerical settings and dependency lock are retained.
