# Fixed post hoc policy comparison

Specified UTC: 2026-10-11T02:25:32.351668+00:00

Question: For the same historical records lacking abstracts, how does retaining authentic titles differ from dropping those records? This is the one principal added experiment. It addresses the policy specificity of the empirical result, not novelty validity or observed missingness. Specified after original findings; not preregistered confirmation.

Fixed inputs: saved SPECTER float32 full/full and full/title similarities, frozen ordered 2018–2020 histories and 2021 focal papers, complete-history baseline, exact saved 25% masks. Both collections, random (200), source-preferred (200), and original year-matched source controls (200), each evaluated under both policies: 1,200 arms, 2,400 policy evaluations. No new masks, encoders, text, cohorts or embeddings. Removal intentionally reduces the candidate count; it is not a count-preserving control.

Readouts: G = absolute change in source contrast / baseline population SD; T = baseline top-ceil(0.1n) fraction replaced using score-descending/lexical-ID membership ties; Q = mean absolute average-rank change/(n−1). Also report signed source contrast, whole-cohort/source-specific mean shifts, positive/negative score-change proportions, and R = baseline-top-k mean baseline score minus perturbed-top-k mean baseline score, divided by baseline SD. R is a descriptive companion distinguishing turnover from baseline score shortfall; it is not lost scientific quality. No other shortlist sizes.

Pairing: removal minus title on each identical saved mask; source minus original matched control within each policy on the original assignment pair. Report every condition and endpoint, including null/adverse outcomes. No preferred-policy selection or new tolerance classifications.

Uncertainty: descriptive assignment mean, sample assignment SD, empirical 5th/95th percentiles, and SD/sqrt(200) Monte Carlo SE for each condition and paired contrast. These are not population intervals or formal significance tests. No new confidence intervals or hypothesis family; original 45/18/12 families unchanged. Assignment or original pair is the unit. No focal-paper pseudo-replication.

Validation: all title scores must exactly reproduce saved float32 scores after float64 cast; baselines must match; cache hashes must match the prior frozen diagnostic manifest. Mask counts, pair-cell counts and chronology checked. Removal must never lower nearest-prior distance; source-weighted shifts recover drift; R nonnegative within numerical tolerance. Synthetic tests cover ties, sign directions, zero perturbation and R. Fail closed on any discrepancy.

Resources: previous saved neighbor benchmark suggests ~0.03 s per two-arm evaluation on this host, but not a fresh estimate. Measure first selected arm in each collection, require conservative fourfold projected computation under600s, hard cap600s, checkpoint every100 evaluations. No GPU/network needed for computation. Execute once under exclusive lock; completed receipt prevents rerun. Outputs remain private under outputs/policy; append method/results only after all selected conditions complete and original inputs rehash unchanged.

Completion: 2,400 rows, 200 per condition/policy, all validations pass; report exact counts/hashes/timing and all results. Do not extend to50%, other models or further analyses based on observed outcomes.
