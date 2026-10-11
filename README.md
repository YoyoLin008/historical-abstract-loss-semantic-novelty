# Historical abstract loss and use-dependent robustness of a semantic novelty indicator

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23299162.svg)](https://doi.org/10.5281/zenodo.23299162)

Research code, derived numerical outputs, an author manuscript and supplements for an independent study by **Yunya Lin**.

## What this study asks

Can a semantic novelty indicator remain stable for comparing publication groups while changing which individual papers receive high scores?

The study evaluates an established nearest-prior semantic-distance score: one minus the maximum cosine similarity between a focal paper and earlier papers. It simulates historical abstract loss by replacing affected historical title-and-abstract representations with authentic title-only representations, retaining every historical record and leaving focal texts unchanged.

The primary SPECTER experiments use 2021 focal papers and 2018–2020 histories in two selected complete-text collections:

- **ACL/EMNLP:** 1,557 focal papers and 3,799 historical records.
- **PLOS Computational Biology/PLOS Genetics:** 1,471 focal papers and 3,391 historical records.

Frozen protocols, seeds and opportunity-matched controls define the experiments. Readouts measure source-contrast distortion, rank displacement and replacement of the baseline top decile. Later analyses include cutoff diagnostics, a limited SPECTER2 sensitivity check and a separately specified **post hoc** comparison of title fallback with record removal.

## Main findings and limits

- Under random 25% historical abstract loss, small source-contrast changes coexist with **14.10% and 14.90% top-decile replacement** in the two collections, respectively.
- Source-preferred loss reduces replacement relative to matched controls in the conference collection and increases it in the biomedical collection.
- In the tested 25% same-mask comparisons, removal produces more replacement than title fallback, but the readouts do **not** establish uniform policy superiority.

These are conditional synthetic-mask results for the selected collections and representations. They do not estimate real database missingness, validate scientific originality or establish universal selection costs. Similar masking responses across encoders do not establish interchangeable baseline rankings. See the [manuscript](preprint/manuscript.pdf) and [full methods and limitations](preprint/ESM_1.pdf).

**Publication status:** the manuscript was submitted to Scientometrics on 11 October 2026. The public author manuscript is unreviewed; submission does not imply acceptance.

## Find the materials

| Location | Contents |
| --- | --- |
| [preprint/](preprint/) | Manuscript PDF, methods supplement (`ESM_1.pdf`) and numerical supplement (`ESM_2.zip`) |
| [verification/scalar_scores/](verification/scalar_scores/) | Complete saved scalar scores, masks, ordered identifiers, pairing mappings and independent endpoint verifier |
| [verification/policy_comparison/](verification/policy_comparison/) | Fixed post hoc specification, implementation, 2,400 policy rows and complete descriptive summaries |
| [data/derived/](data/derived/) | Original and revision-stage summaries, cutoff membership projections and diagnostics |
| [scripts/](scripts/) | Reporting-table, figure and membership checks |
| [src/](src/), [analysis/](analysis/), [replication/](replication/) | Scientific implementations, readouts and biomedical replication code |
| [docs/](docs/) and [manifests/](manifests/) | Dictionaries, protocols, result-to-file map, provenance and license inventories |

## Verify the saved-score results

The public scalar package independently reconstructs **5,000 scored arms and 75 summary means** from saved scores. It does not require source text, embeddings, model weights or network access during reconstruction.

From the repository root, with Python 3.12 installed, use this macOS/Linux setup:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r verification/scalar_scores/requirements.txt
.venv/bin/python -B verification/scalar_scores/reconstruct_endpoints_from_scalar_scores.py --package verification/scalar_scores --output outputs/scalar_verification
```

The output includes `outputs/scalar_verification/verification_report.json` and numerical comparisons. Dependency installation requires network access. This setup is guidance; it is not a claim that a fresh dependency installation was verified during release preparation.

See the [scalar package guide](verification/scalar_scores/README.md) for definitions, tolerances, fixture tests and the optional Python audit-hook guard. See [reproduction levels and reporting commands](docs/reproduction.md) and [environment notes](environment/README.md) to rebuild tables and figures.

**Reproducibility boundary:** saved-score endpoint reconstruction and reporting are publicly runnable. Fresh acquisition/encoding and the title-fallback/removal scoring code require omitted inputs. In particular, removal scoring needs private similarity caches and metadata; its published code and summaries do not supply a second standalone scalar reconstruction. Source texts, embeddings, model weights, detailed focal-neighbor records and editable manuscript templates are excluded.

## Cite and reuse

Cite the fixed [v1.2.0 archive](https://doi.org/10.5281/zenodo.23299162); machine-readable citation metadata is in [CITATION.cff](CITATION.cff). Earlier releases remain available through [release history](https://github.com/YoyoLin008/historical-abstract-loss-semantic-novelty/releases) and [CHANGELOG.md](CHANGELOG.md). The original 37-file scalar package first appeared in [v1.1.1](https://doi.org/10.5281/zenodo.23298679) and is unchanged in v1.2.0. Later README clarifications on the default branch do not alter deposited versions.

Named original code/tests use **MIT**; named author-controlled numerical outputs, documentation and manuscript/supplement works use **CC BY 4.0**. Exact assignments and exclusions are in [LICENSE](LICENSE), [manifests/license_scope.csv](manifests/license_scope.csv) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). These grants do not relicense third-party materials.

For questions or reproducibility issues, please [open a GitHub issue](https://github.com/YoyoLin008/historical-abstract-loss-semantic-novelty/issues).
