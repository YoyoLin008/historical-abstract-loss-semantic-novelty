# Post hoc whole-mask sensitivity plan — not executed

Prepared on 2026-10-06 after the frozen study outcomes and a descriptive distribution audit. This is a separately labelled post hoc robustness analysis; it is not part of either original prospective family. Latest author request authorizes evaluation of such checks. Archive this plan/hash before executing any bootstrap.

## Question and immutable inputs

Does relaxing the marginal Student approximation materially change the original endpoint threshold classifications or the directions of paired effects? Use only `mask_primary_and_movement.csv` and the original primary/paired tables from the ACL and biomedical frozen analyses. Input SHA-256 values are in `distribution_audit.json`; verify them against frozen analysis manifests before proceeding. No score recalculation, new mask, new seed manifest for scientific assignments, encoder, corpus, threshold or tuned resampling method.

## Fixed procedure

Use a studentized nonparametric bootstrap of complete independent assignments within each mechanism/rate block. Fixed B=200000 bootstrap replicates for every block, n=200 sampled rows with replacement, NumPy PCG64, root resampling seed 20261006 and stable block-specific seed derivation from SHA-256 of `["major-revision-studentized-bootstrap-v1",20261006,study,rate,mechanism]`. This seed belongs to resampling existing outcomes; it does not create new scientific masks.

For a random block resample its whole(G,T,Q) vector. For a structured block join each control to the structured row by exact assignment_id and jointly resample all six outcomes. Form each sampled row's three structured-minus-control differences before calculating means and variances. Preserve the complete endpoint vector and pairing; never resample endpoints or arms separately. Blocks from different conditions are resampled independently. Keep the original 45 ACLmeans and 18 biomedicalmeans as two separate inferential scopes; control means alone are not added to either scope.

For each endpoint mean let original estimate/SE be xbar,se. Each bootstrap replicate yields xbar*,se* and t*=(xbar*−xbar)/se*. For family size J, set a=.05/(2 J), using empirical quantiles q_a and q_(1−a) with NumPy `method="linear"`. Report the interval `[xbar−q_(1−a)*se, xbar−q_a*se]`. These are approximate bootstrap-Bonferroni sensitivity intervals, not exact finite-sample guarantees. Do not clip limits to mathematical bounds for comparison. If original SE is zero or any resampled SE is zero, record it and stop that endpoint rather than inject jitter or substitute a different method.

Compute in batches of 2000 bootstrap replicates to limit memory; no parallel workers or remote compute. Save complete methods/runtime/library/input/seed hashes, mean/SE and tail quantiles, sensitivity intervals and all classifications in a new posthoc directory. Store first-half and second-half quantile estimates as a fixed Monte Carlo diagnostic, not as opportunities to rerun until favorable. Extreme-tail quantiles have finite-B uncertainty; report differences between halves. Predicted resampling work is local and modest relative to encoding, but measure actual memory/time and checkpoint if necessary. This plan does not claim measured execution time.

## Fixed comparison/decision rule

Classify all base G,T,Q means at the original tolerances(.10,.05,.01), all 13 joint G/T rules, and the sign of each 24 paired-effect intervals. Use the same strict inequalities and unresolved/touching convention as the original. Report every concordance/discordance and numerical interval difference; no selective reporting or replacement of original intervals.

If any joint rule changes, describe that condition as sensitive to interval approximation. If a paired direction changes from directional to unresolved, narrow only that directional claim. If any of the three illustrative 25% conditions(ACL random; biomedical random; biomedical source) changes its central aggregate/selection interpretation, explicitly state that the corresponding robustness inference is sensitivity-dependent. The fragile ACL source 10% condition is highlighted a priori because its frozen lower T bound is 5.0112%, not because bootstrap outcomes are favorable. Differences confined to it do not erase well-separated 25% findings.

Retain the original frozen inference as primary. Do not increase B, masks or sample size after seeing a threshold crossing. A discordance is an informative limitation. This check cannot adjudicate true originality, actual database omission mechanisms, metadata snapshot bias or alternate-model transportability.
