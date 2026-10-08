# R3/R5 numerical resource dictionary

These outputs contain revision-stage summaries of original saved assignments (R3) and a post hoc descriptive neighbor diagnostic (R5). All analysis units are original masks or matched pairs. No inferential intervals or new testing family are introduced. Source labels ACL/EMNLP and PLOSCB/PLOSGEN mean the original ACL collection and PLOS Computational Biology/PLOS Genetics frame. `topic` is the frozen ACL title-substring lexical preference, not a validated topic.

## Common keys and summary conventions

`study`, `condition`, `rate`, and `arm_role` identify a unique arm condition. `assignment_id` is the original base/pair ID; `arm_id` includes the original `_matched_random` suffix for controls. Rates are proportions. Conditions `source_matched_control` and `topic_matched_control` match historical year and historical year × source respectively. Every condition has 200 assignments. Means and SDs first summarize assignment-level quantities. SD is sample SD (`ddof=1`); p05/p95 use NumPy linear interpolation. No rounding occurs before aggregation; display rounding is separate. A blank CSV value is undefined (usually an empty conditional cell), not zero. `*_defined_masks` records how many mask values define a conditional summary. Unconditional fixed-denominator contributions are zero for empty cells and remain defined in all 200 assignments.

## R3 summary files and private replay companions

- `history_strata.csv`: authoritative historical denominators `N_s`. ACL sources: 1,819 ACL and 1,980 EMNLP; biomedical sources: 1,854 PLOSCB and 1,537 PLOSGEN. ACL lexical strata: 289 `translat` and 3,510 `nontranslat`. Preferred refers to ACL/PLOSCB source or the `translat` lexical stratum.
- `intensity_per_assignment.csv` (private replay companion, outside ESM2): all original base/control source strata and ACL lexical-condition/control lexical strata. `c_m_s` is the original mask count in the stratum. `total_masked` is the complete mask count. `within_stratum_fraction=c_m_s/N_s`. `masked_set_composition=c_m_s/total_masked`. These fractions have different denominators. 10,800 rows cover 4,200 arms; the extra rows are lexical-stratum views, not extra masks.
- `intensity_summary.csv`: 54 stratum/condition rows. Each count, within-stratum fraction and masked-set share has `_mean`, `_assignment_sd`, `_p05`, `_p95`; `N_s` and `n_masks` give denominators.
- `intensity_table.tex`: means (SDs) of within-source fractions in percent, including controls and all original conditions. Preferred means ACL for ACL collection and PLOSCB for biomedical collection.
- `intensity_lexical_table.tex`: analogous preferred/nonpreferred lexical fractions for all three original ACL rates and matched controls. Full percentiles/composition remain in the CSV.

Mask IDs are authoritative saved originals; no draws were reconstructed. All original total counts and 6,600 matched cells pass exact equality. Frozen `topic==0` membership was checked against the original archived title substring `translat` solely to verify the existing classification.

## R5 files

Scope is original SPECTER source-preferred 25%/50% and year-matched controls in both collections: 800 pairs, 1,600 arms, 2,422,400 focal-arm evaluations. Existing exact float32 similarity caches reproduce all saved original scores bit-for-bit. Original full focal inputs and strictly prior eligible histories are unchanged.

`neighbor_per_assignment.csv` (private replay companion, outside ESM2) contains 4,800 rows: each arm has ALL plus its two focal-source views. Views overlap and must not be pooled as independent observations. `fixed_focal_denominator` is the cohort/source population; `fixed_selected_denominator` is the number of baseline top-decile selections in that source. `exposed_fraction` means baseline winning history masked. `identity_switch_fraction` means the winning historical paper changed. `winner_title_fraction` means the perturbed winner uses the title-only representation; this does not imply identity changed. Increase/decrease/unchanged fractions compare exactly preserved scalar values. `mean_shift_raw` is the mean masked-minus-baseline score; `mean_shift_sd` divides by original full-cohort population SD (`ddof=0`, float64 readout). Selection loss fractions use the fixed baseline-selected denominator, and loss counts are raw counts. Same-source fractions compare focal source with winning historical source. Tie flags identify exact cosine maxima without tolerance rounding.

`neighbor_summary.csv` reports assignment distributions for every numeric per-assignment quantity, including constant denominators (their SD is zero). `neighbor_table.tex` displays the source-preferred source-level summary with selected loss rates; `neighbor_controls_table.tex` supplies an optional concise full-cohort structured/control summary. Full source-specific control values remain in CSV.

`neighbor_transition_per_assignment.csv` (private replay companion, outside ESM2) partitions each fixed focal cohort/source into eight cells: baseline winner masked yes/no × historical identity switched yes/no × perturbed winner title-only yes/no. `cell_count` is the cell size; `cell_fraction` divides by the fixed focal denominator. `shift_contribution_sd` sums the cell's standardized score changes and divides by the fixed focal denominator; summing all eight cells reconstructs the source/cohort shift. `conditional_shift_sd` instead averages within the cell, and is undefined for empty cells. Selection counts/losses within cells partition the fixed selected set; `conditional_selected_loss_fraction` uses the number originally selected within that cell and is undefined when zero. `neighbor_transition_summary.csv` gives mask means, sample SDs, percentiles and defined-mask counts for all cells. These are descriptive accounting identities, not causal mediation or evidence that labels cause the effects.

No baseline/masked exact-max ties occurred in this fixed scope. The original scorer defined maximum values without saving historical winner IDs; this new diagnostic uses lexical historical ID for exact winner-identity ties only. Original focal top-decile selection retains its lexical focal-ID tie rule and never rounds scores first. If a baseline winning full representation remains unmasked, novelty cannot increase; every row satisfies this bound.

## Private records and provenance

Private audit component files contain hashes, receipts, benchmark resource measurements, synthetic test logs and the R1 bridge. `neighbor_records/` holds numeric lossless NPZ arrays with CSV focal/history/assignment indices. They contain no text or embeddings, but scalar changes can recover scores when combined with baselines; they are private and excluded from ESM2 pending explicit file-level author review. They are not covered by a claim that the old v1.0.0 DOI contains new outputs.

`neighbor_R1_bridge.csv` compares 4,800 per-assignment direct source/cohort shifts against independently reconstructed R1 values. Private NPZ arrays are reloaded with `allow_pickle=False`; all 4,800 source/cohort means reconcile with the written summaries. Runtime/code/input/output hashes are in `intensity_verification.json`, `neighbor_verification.json`, `neighbor_R1_and_export_verification.json`, and `fragment_manifest.json` under private audits.

## Core provenance map
The companion core/claim_provenance.csv identifies output_file and one-based csv_line, JSON output_key, source_files (relative to data/derived), source_filter, formula, generating_code (analysis implementation path) and generated_columns. It contains no new masked scalar vectors. Its scope is the seven included core summary CSVs only.

Version 1.1.0 includes aggregate summaries and the actual analysis source, but not private audit files, per-assignment mechanism companions or focal-neighbor records. Historical execution descriptions above are not claims of a fresh saved-cache run.
