# Reproduction levels and commands

The historical reporting commands below use public derived summaries; the root README additionally documents the new scalar route. Run the reporting commands in a clean checkout with Python 3.12, NumPy 2.3.5, matplotlib 3.10.7 and mpmath 1.3.0. `requirements.txt` pins the figure dependency set. The release preparation records the exact observed versions, commands, runtimes and outcomes privately; a clean directory is distinct from a newly installed environment. Generated output stays under `outputs/` and is ignored by Git.

| Layer | Inputs and operation | Supported scope |
|---|---|---|
| Original reporting | Saved per-assignment endpoints, configs, frozen summaries | 39 primary plus 24 paired original mean/SD/MCSE/Student summaries; cutoff arithmetic; existing bootstrap/encoder views; failed calibration and pilot readout |
| Saved membership | Four baseline scalar lists and 5,000 saved-arm loss-ID projections | Reconstruct baseline ordering/ties/sets; check 20,000 positive-margin intersections. Masked scores are not recomputed. |
| New core reporting | Public outcomes, probabilities and loss IDs | Seven original revision summary CSVs: signed drift, conditional margin, capacity, nesting and cross-encoder agreement/composition/context; no scoring. Derived per-mask reporting output is generated locally. |
| Mechanism presentation | Frozen intensity/neighbor aggregate CSVs | Reformat Tables S16–S18 only; no new neighbor or intensity computation |
| Current figures | Original means/intervals and conditional-margin summary | Four main plus two supplementary figures, using `build_submission_figures.py`; original `rebuild_figures.py` preserves old-version presentation |
| Complete masked scalar endpoints | Public verification/scalar_scores package | Independently reconstructs5,000 arms/75 means from complete saved scores, IDs and pair mappings; see its README. Does not perform fresh acquisition/encoding. |
| Saved embedding/similarity scoring | Omitted original text metadata, saved masks and float32 similarity caches | Source is inspectable; complete replay not performed in this packaging pass |
| Full acquisition-to-encoder | Independently obtained provider records and pinned models; encoder environment | Not rerun; rights, network, compute and exact numerical environment require separate work |

`analysis/revision_mask_intensity.py` reads authoritative saved missing-history IDs, original metadata, completed run manifests and seeds under the original project layout. `analysis/revision_neighbors.py` additionally needs exact float32 similarity caches and an approved post hoc specification/benchmark fingerprint. Their fixture tests exercise functions without those inputs. The entrypoints are retained byte-for-byte from the actual rc.2 implementation; the public command list does not attempt to fabricate the missing inputs. Aggregate neighbor rows are overlapping fixed-source views, not independent focal observations; all post hoc labels and denominator definitions remain in the dictionaries.

Numerical identity should be checked against `manifests/export_files.json`; original byte bindings remain in `manifests/original_to_release.csv` and `manifests/history/`. No original scientific code/config/result file was changed in the update. Full raw-to-encoder reproducibility and construct validity are not established by these checks.


## Historical reporting commands (preserved)

```sh
python -B scripts/validate_release_data.py
python -B scripts/rebuild_tables.py
python -B scripts/verify_cutoff_membership.py --data data/derived/cutoff_verification
python -B analysis/revision_descriptive_outputs.py --data-dir data/derived --output-dir outputs/revision_core
python -B scripts/rebuild_revision_mechanism_tables.py --out outputs/revision_mechanism_tables
python -B scripts/build_submission_figures.py --project . --descriptive data/derived/revision_descriptive/core --out outputs/current_figures
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s analysis -p "test_*.py" -v
```
