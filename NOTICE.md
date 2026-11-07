# License scope and provenance

The root MIT license applies to the author-owned experiment code, build and verification scripts, and software documentation in this repository. It does not grant a new license to the manuscript text or scientific figure artwork, which remain copyright El Mustapha Mansouri, or to third-party materials and datasets.

`camera_ready_v3_10page/neurips_2026.sty` is the supplied January 2026 NeurIPS style. Its original attribution header is retained unchanged. This third-party style is supplied for compiling the accompanying manuscript; the root MIT license does not relicense it. Third-party Python and TeX dependencies retain their respective licenses and are installed separately. `checklist.tex` is an unfinished author placeholder and is not included in the PDF; acknowledgment and funding disclosures also await author completion. The release supplies the current manuscript rather than asserting submission readiness.

LendingClub source records and prepared borrower-level data are not redistributed. Obtain the source files under their source terms as described in `docs/data.md`. Saved experimental tables contain aggregate evaluation results, not individual loan records.

This public release is a separate snapshot of the manuscript and reproducibility material. It does not contain the private working repository's history. Historical manifests preserve the seeds, scientific configuration, input/source/output hashes, and software/hardware information. Host-specific paths have been replaced with repository-relative paths or explicit portable placeholders; this changes the affected metadata-file hashes, not the recorded numerical results. The release manifest records the hashes of the public copies.

Portability edits to helper scripts are documented in `docs/experiments.md` and `docs/figures.md`. The saved source hashes describe the implementations that produced the recorded experiments, rather than a claim that every documentation or helper file is unchanged. New runs must use a new output directory. Existing historical checkpoints are evidence of the reported runs, not portable resume state.
