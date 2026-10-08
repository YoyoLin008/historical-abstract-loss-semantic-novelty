# Historical abstract loss and use-dependent robustness of a semantic novelty indicator

Research software and original non-text numerical materials, version **1.1.0**. Author: Yunya Lin, School of Information Sciences, University of Illinois Urbana-Champaign. This software archive is not an article preprint, journal submission or acceptance.

The study holds focal title-plus-abstract inputs fixed and substitutes authentic title-only representations for selected historical inputs. It distinguishes source-contrast distortion, ranking displacement, selection replacement and score drift. Original frozen ACL and biomedical experiments, the unsuccessful pilot and separately labelled post-confirmation checks remain unchanged. Revision-stage descriptive additions are post hoc: signed contrasts and drift, cutoff capacity and conditional-margin loss, complete-history cross-encoder agreement, realized masking intensity and bounded neighbor accounting. No new masks, encoders, thresholds or inference are introduced by this release.

## Reproduce the supported layers

Use Python 3.12 and the pinned reporting dependencies in `environment/requirements_reporting.lock`; `requirements.txt` adds exact figure-package pins. These commands work from the repository root without source abstracts, embeddings, weights or private project paths:

```bash
python -B scripts/validate_release_data.py
python -B scripts/rebuild_tables.py
python -B scripts/verify_cutoff_membership.py --data data/derived/cutoff_verification
python -B analysis/revision_descriptive_outputs.py --data-dir data/derived --output-dir outputs/revision_core
python -B scripts/rebuild_revision_mechanism_tables.py --out outputs/revision_mechanism_tables
python -B scripts/build_submission_figures.py --project . --descriptive data/derived/revision_descriptive/core --out outputs/current_figures
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s analysis -p 'test_*.py' -v
```

`docs/results_map.csv` maps all current displays. `scripts/rebuild_figures.py` is retained for historical v1.0.0 figures; use `build_submission_figures.py` for the four current main and two supplementary figures. New core summaries are reconstructed from saved outcomes and membership evidence. Mechanism tables are formatted from frozen aggregates; that command does not repeat their full saved-cache analyses. Existing baseline scalar lists and saved loss identifiers support membership checks; **complete masked per-paper vectors and focal-neighbor records are not in this release**. No public or reviewer-only access to omitted records is promised. See `docs/reproduction.md` for input and test boundaries.

Raw provider text, embedding arrays, model weights, licensed literature, manuscript/source, private audit receipts and private project history are excluded. Original and revision intensity/neighbor source code is included for inspection, but its full execution requires those documented private inputs. Full fresh raw-text-to-encoder reproduction remains incomplete. Checksums establish consistency, not scientific originality or construct validity. Fixed-corpus assignment intervals do not describe uncertainty across fields. The finite 2018–2020 history, failed OpenAlex calibration, near-zero/changed-sign biomedical SPECTER2 contrast and unknown final SemNovel methods remain limitations.

## Versions, citation and rights

The immutable historical v1.0.0 release (commit `82bd064fae8f4aff9a2f0b56bb9ec850bd649e89`) has [version DOI 10.5281/zenodo.23222368](https://doi.org/10.5281/zenodo.23222368). It does not contain the revision additions. Version **1.1.0** was published on 8 October 2026: [version DOI 10.5281/zenodo.23242635](https://doi.org/10.5281/zenodo.23242635), [GitHub release](https://github.com/YoyoLin008/historical-abstract-loss-semantic-novelty/releases/tag/v1.1.0), frozen commit `9870031b3fcfe62943d15b4ad6d74404b2d9fc6e`. All 146 downloaded Zenodo payloads match the approved commit by path, size and SHA-256. The [concept DOI 10.5281/zenodo.23222367](https://doi.org/10.5281/zenodo.23222367) identifies the version series. This subsequent documentation commit records publication without moving either release tag. Preparatory approval labels in the archived snapshot record its prepublication state; the author approved its exact scope before publication.

MIT and CC BY 4.0 apply only to their explicitly named files in `manifests/license_scope.csv`; `LICENSE` and `THIRD_PARTY_NOTICES.md` define exclusions. The author explicitly approved the exact new scoped grants before publication on 8 October 2026. All 47 optional scalar/neighbor package files remain private by the author's decision. Third-party libraries and source materials retain their own terms. AI assisted coding, checks, manuscript organization/wording and descriptive-analysis/interpretive suggestions during revision; the human author retains responsibility.
