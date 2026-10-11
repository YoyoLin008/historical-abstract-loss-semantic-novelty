# Historical abstract loss and use-dependent robustness of a semantic novelty indicator

**Public-access release v1.1.1: author preprint + reproducibility materials.** The manuscript is not peer reviewed or journal accepted. Yunya Lin, University of Illinois Urbana-Champaign.

- [Read the manuscript](preprint/manuscript.pdf)
- [Read the methods/results supplement](preprint/ESM_1.pdf)
- [Download numerical supporting information](preprint/ESM_2.zip)
- [Reconstruct endpoints from the 37-file scalar package](verification/scalar_scores/README.md)

With Python 3.12 and the pinned dependencies installed:

```sh
python -B verification/scalar_scores/reconstruct_endpoints_from_scalar_scores.py --package verification/scalar_scores --output outputs/scalar_verification
python -B -m unittest discover -s verification/scalar_scores/tests -v
```

The independent verifier computes from complete saved scores, ordered identifiers and source labels rather than using saved endpoints as calculation inputs. Expected outputs:5,000 scored arms,7,630,200 masked scalar values,75 family-summary rows and116,898 checks;15 synthetic fixtures. This does not perform fresh scoring from embeddings or reacquisition/encoding.

The original 146-file research/reporting release is preserved in [v1.1.0](https://doi.org/10.5281/zenodo.23242635). This additive version preserves its scientific payloads and adds the preprint/supplements and complete scalar-verification package. The complete v1.1.1 archive, including the preprint and all 37 scalar-verification files, is publicly available at [Zenodo DOI 10.5281/zenodo.23298679](https://doi.org/10.5281/zenodo.23298679). All 188 downloaded payloads match the frozen release.

The root LICENSE and manifests/license_scope.csv provide exact scoped MIT/CC BY 4.0 grants. No third-party raw text, embeddings, model weights, manuscript TeX templates, private audits, correspondence or focal-neighbor package is included. The 37scalar files supersede their previous private-access status; the 10neighbor files remain private. See [docs/public_access_v111.md](docs/public_access_v111.md) for provenance and scope.

Existing reporting commands remain available in [docs/reproduction.md](docs/reproduction.md). Statements explicitly about historicalv1.1.0 retain that historical scope. A public preprint is not journal publication or acceptance.
