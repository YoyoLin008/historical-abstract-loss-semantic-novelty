# Reproduction levels and executed scope

## 1. Reporting-level route (supported by this export)

From the repository root, after selecting the reporting interpreter:

```sh
python scripts/validate_release_data.py
python scripts/rebuild_tables.py
python scripts/rebuild_figures.py --project . --out outputs/figures
python scripts/verify_cutoff_membership.py --data data/derived/cutoff_verification
python -m unittest discover -s tests -v
python -m unittest discover -s analysis -p 'test_*.py' -v
```

No command above downloads models, queries a provider, reads source text/arrays or launches the original experiments. They validate all released CSV/JSONL schemas, recompute the original 39 mixed-policy primary means and 24 paired means, SDs, Monte Carlo SEs and original Student limits from released mask-level outcomes, check cutoff count identities, reconstruct table views and recreate all four figures from original summary rows. The SPECTER2 and bootstrap numerical outputs are read as separately labelled results; neither analysis is rerun. OpenAlex coverage is reconstructed from released field-state identifiers, preserving failed calibration. Pilot held-out errors are reconstructed from preserved prediction/target arrays, without refitting. Figures retain the 13 original conditions, eight paired contrasts and original 45/18-mean interval families.

Data inputs are under `data/derived/` and bibliographic IDs under `data/identifiers/`. The shortest table/figure route is indexed in `docs/results_map.csv`. CSV blanks mean an endpoint is undefined (empty bins) or no inferential interval was specified for that non-primary/descriptive endpoint; they are never silently zero. Full-precision source values lead; printed rounded percentages/counts are presentation only. See `docs/data_dictionary.md`, `docs/csv_dictionary.csv`, and `manifests/csv_schemas.json` for every released CSV column.

## 2. Cached-score or cached-embedding reconstruction (not executed from this export)

The original study previously reproduced ACL and biomedical scoring/readouts in a separate Python 3.12.14 environment from exact retained full/title embeddings, matching three scoring files per collection and 15 readout CSV/PNG/cutoff files per collection. This public candidate does not distribute those embeddings, original text-bearing records, saved full score vectors or raw response caches. No complete cached-score rerun is possible from the public export alone. That limitation is not repaired by a checksum or a PDF rebuild.

For an authorized private input owner, prepare a separate full reproduction tree with all files bound by `configs/confirmation_freeze_v1.json` / `configs/pubmed_replication_freeze_v1.json`, including original source-text record files, full/title arrays and original encoding/cohort/acquisition manifests. Restore original paths from each freeze `artifact_paths` and relevant manifest, preserving byte identity. Retain original byte-identical `src`, `analysis`, `replication`, configs and root `requirements.lock` supplied here; original protocols are in `docs/protocols/` and must also be copied to the historical `docs/` paths named by the freeze. Never place outputs over the original frozen files.

Primary ACL scoring, then readout (parameterized instructions, not tested from this public candidate):

```sh
python -m src.confirm --root "$PWD" --records data/confirmation_v1/records.jsonl --full data/confirmation_v1/embeddings/full.npy --title data/confirmation_v1/embeddings/title.npy --encoding-manifest data/confirmation_v1/embeddings/encoding_manifest.json --config configs/confirmation_v1.json --seed-manifest configs/confirmation_seeds_v1.json --freeze-manifest configs/confirmation_freeze_v1.json --protocol docs/confirmatory_protocol_v1.md --output results/review_acl
python -m analysis.confirmatory_readout --root "$PWD" --results results/review_acl --records data/confirmation_v1/records.jsonl --config configs/confirmation_v1.json --seed-manifest configs/confirmation_seeds_v1.json --freeze-manifest configs/confirmation_freeze_v1.json --protocol docs/confirmatory_protocol_v1.md --cohort-manifest data/confirmation_v1/cohort_manifest.json --encoding-manifest data/confirmation_v1/embeddings/encoding_manifest.json --output results/review_acl/analysis --report results/review_acl/readout.md
```

Biomedical route substitutes `python -m replication.score` / `python -m replication.analysis`, biomedical config/seeds/freeze/protocol paths, `replication/data/records.jsonl`, the three `replication/data/embeddings/` inputs, `--cohort-manifest replication/data/cohort_manifest.json`, `--acquisition-manifest replication/data/acquisition_manifest.json`, and fresh `results/review_biomedical` / `results/review_biomedical/analysis` outputs. The acquisition manifest also requires its bound metadata ledgers; text-bearing ledgers are absent here. Inspect `python -m replication.score --help` and `python -m replication.analysis --help` before using exact arguments. These help commands need NumPy but do not launch experiments.

Post-confirmation source is supplied for inspection at `src/post_confirmation/`. Its original functions are unchanged; only personal root constants were adapted, with original/export hashes and function-AST comparison in `manifests/original_to_release.csv`. Historical freezes still bind original source bytes and cannot be substituted silently by this path-adapted export. The companion's full private frozen work tree/model metadata/manifests must be obtained lawfully; its original historical code identity and transparent runtime path/Student-solver overrides are documented in original provenance. Packaging syntax/endpoint fixtures are the executed verification here. No new SPECTER2 cached scoring, encoding or bootstrap is claimed.

## 3. Raw-to-encoder reconstruction (not executed from this export)

Use `docs/acquisition.md` and original byte-identical acquisition/encoder sources to acquire the pinned ACL XML and a separately identified biomedical snapshot. Live biomedical queries can change source records, corrections and rights; changed records must not be called the frozen experiment. Acquire each original model at its recorded revision, verify terms/hashes, then use explicit records/output/model paths. Full/title CLS vectors are L2-normalized float32 with a 512-token limit. Never pass fixture `--synthetic-test` for empirical reconstruction. A clean download-to-encoding-to-analysis run has not been verified on a second machine. Releasing source hashes does not make omitted input bytes downloadable.

## Resources and limitations

Packaging reporting checks and fixture tests use local CPU only and ordinarily fit within a few minutes and well under 1 GB additional storage; actual packaging time/RSS/storage are in the accompanying reproduction receipt. This is a planning estimate until measured by that receipt. Historical SPECTER2 encoding measured 480.3 seconds (ACL) and 734.5 seconds (biomedical) on the original CPU; no GPU/remote spending occurred. Original full dependency/model installation and acquisition transfer size are external. Plan within 40 GB working storage and 30 GPU-hours; there is no measured need for paid/remote computing. The fixed cohorts are 5,356 ACL and 4,862 biomedical records with 768-dimensional full/title arrays; those original float32 arrays alone total about 62.78 MB across both collections (excluding cache/model overhead). Do not interpret measured original-host times as guaranteed cross-platform runtimes.

Targeted cutoff proof is also released: `python scripts/verify_cutoff_membership.py --data data/derived/cutoff_verification`. It reconstructs selected sets, tie-expanded rank bands and positive margins from the four original/SPECTER2 scalar baseline score lists, then verifies 16 baseline comparisons and 20,000 saved loss-ID intersections. The 5,000 saved scored arms consist of 4,200 original arms and 800 limited SPECTER2 arms; the latter reuse the original 25% mask assignments and are not new independent draws. The lean public JSONLs selectively retain numeric endpoints and ID memberships from saved scientific outputs; later private movement-field additions are not part of this public projection. Full masked-score vectors remain omitted, so this supports reporting-level membership verification. Baseline scalar scores are original non-text numerical outputs included specifically for inspectable cutoff proof. Their origin hashes and transformation scope are in `manifests/cutoff_inputs.json`.
