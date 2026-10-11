# Scalar-score reconstruction package

This 37-file package is distributed with public release v1.1.1. It supports independent reconstruction from the original saved scalar scores. The root LICENSE and manifests/license_scope.csv specify the scoped MIT/CC BY 4.0 grants. It is not contained in the earlier v1.0.0 or v1.1.0 archives. Preparation and test receipts describe when the inputs were generated; they do not change current access. No third-party text, embeddings or weights are included.

The four datasets contain the existing saved mixed-policy vectors: SPECTER ACL (3,000 × 1,557), SPECTER biomedical (1,200 × 1,471), SPECTER2 ACL (200 × 1,557), and SPECTER2 biomedical (600 × 1,471). That is 5,000 scored arms and 7,630,200 masked values. Matched control arms are included, while repeated restored/all-title policies and SPECTER2 paired endpoint-only records are excluded. Baseline arrays add 6,056 scalar values. No score was newly generated from an embedding.

With the listed dependencies already installed, from this directory run:

```
python reconstruct_endpoints_from_scalar_scores.py --package . --output verification
python -m unittest discover -s tests -v
```

For the executed file-access/network guard, use `python -I -B isolated_runner.py reconstruct` and `python -I -B isolated_runner.py tests`. The guard permits this extraction and the installed Python/dependency directories; it denies other file reads and network/subprocess calls. This is a documented Python audit-hook guard, not an operating-system sandbox or new isolated dependency installation. A clean extraction was used for the recorded run.

The verifier implements endpoints independently using only exported scalars, ordered IDs/source labels, masks, and reference projections. It does not import the historical scoring/readout functions. Reference columns are comparison targets, never calculation inputs. It reconstructs 63 original means (45 ACL and 18 biomedical) and 12 separate SPECTER2 means, paired endpoint differences, SD/MCSE, historical critical values and simultaneous intervals, exact lost/selected/cutoff IDs, tie-aware ranks, strict-margin subsets, signed contrasts/drift, and direct source shifts. The exported masks identify the existing intervention and permit exact reused-mask checks; they do not permit fresh text-to-score reconstruction.

The historical readout tolerance for reconciliation with float32 scoring aggregation is G=2e-5, T=2e-10, Q=2e-10, as documented before this revision in `analysis/confirmatory_readout.py` and `replication/analysis.py`. Independent float64 family means/intervals and readout arm metrics are checked at 2e-10, consistent with the original independent interval verification. Counts, pair keys and selected/lost IDs require exact equality. No threshold is relaxed after failure.

Each `scores.npz` has `baseline` (focal,), `scores` (arm,focal), and `missing_history` (arm,history). Scores use float64 to preserve every parsed JSON scalar exactly; originating scores were calculated using float32 and serialized as JSON numbers. Exporting float64 does not add numerical information. NPZ files use no object arrays and are loaded with `allow_pickle=False`; arrays are verified exactly on reload. `focal.csv` and `history.csv` define zero-based ordered axes. `arms.csv` defines rows, original arm IDs, canonical assignment/pair IDs, mechanism, exact rate, role, seeds, mask count and original row content binding. Control arm IDs in the original SPECTER records carry `_matched_random`; the canonical pair ID is the base assignment ID. SPECTER2 reuses the original assignments and masks.

`reference_arms.jsonl`, `reference_baseline.json`, `reference_membership.csv`, and `reference_summaries.csv` are scoped comparison projections. `manifest.json` binds source hashes, dimensions, families and licensing status; `export_receipt.json` records exact scalar reload checks and exported payload hashes. `file_manifest.csv` separately binds all candidate files without self-hashing. Run the commands above to generate verification reports, numeric comparisons and command/environment receipts locally.

Availability layers: public v1.0.0 supports reporting-table reconstruction and cutoff-membership verification. This package makes the previously private complete-score endpoint reconstruction available for public execution. Exact scoring from frozen arrays is a separate layer; raw-text acquisition and fresh encoder reproduction are not executed by this scalar route and are not supplied by this package. No source text, embeddings or model weights are included.
