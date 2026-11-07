# Data access and preparation

Raw LendingClub records, prepared cohorts, cached predictions, and fitted models are not included. The repository contains derived scientific tables, metadata, source hashes, and deterministic preparation code. The release and figure checks use the included results and do not need any borrower-level records.

## LendingClub source files

Obtain the files from the publisher's [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) dataset (`wordsforthewise/lending-club`). Access may require a Kaggle account and agreement to its current download terms. The historical project metadata identifies the dataset as CC0 Public Domain; the repository's MIT software license does not grant rights to third-party data. No credentials or automatic data download are needed for the bundled checks.

The accepted-file acquisition URL is recorded in [the sequential source manifest](../camera_ready_v3_10page/revision_experiments/sequential_plugin/data/source_manifest.json). The rejected-file version-3 URL is recorded in [download_provenance.json](../camera_ready_v3_10page/review/empirical_repair_qa/raw_support_check/download_provenance.json). The exact downloaded gzip bytes matter: a recompressed or changed snapshot can have the same filename but a different hash.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `accepted_2007_to_2018Q4.csv.gz` | 392,582,231 | `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d` |
| `rejected_2007_to_2018Q4.csv.gz` | 255,470,782 | `c1f9f2a4ec5537ef0af44b24b2fb2f41f58fcaec45363ef32c6a2fb27347f6c5` |

Use `sha256sum FILE` on Linux, `shasum -a 256 FILE` on macOS, or `Get-FileHash -Algorithm SHA256 -LiteralPath FILE` in PowerShell. The preparation helper below checks the accepted-file hash itself.

## Accepted-only reconstruction for the current sequential experiment

Install the pinned environment described in the root README, then run:

```bash
cd camera_ready_v3_10page/revision_experiments/sequential_plugin
python prepare_accepted.py --source /data/accepted_2007_to_2018Q4.csv.gz --check-only
python prepare_accepted.py --source /data/accepted_2007_to_2018Q4.csv.gz --output data/accepted_2012_reconstructed.parquet
python run.py --prepared data/accepted_2012_reconstructed.parquet --check-only
```

Replace `/data/...` with the file's actual location on your machine; the same arguments work in PowerShell. Preparation itself is a CPU operation. The first command verifies source size, SHA-256 and column mapping and writes nothing. The second uses the original loader, original order and `A`-prefixed row identifiers. Before saving, it enforces 53,367 accepted mature loans, 8,644 defaults, and the 20 historical mask metadata records at absolute tolerance 1e-12. It writes a new sibling `accepted_2012_reconstructed.manifest.json`; the published `source_manifest.json` remains unchanged. The third command checks the pinned GPU training prerequisites; the full rerun command is in [experiments.md](experiments.md).

The historical reconstructed Parquet file was 736,542 bytes with SHA-256 `fcd06d383adff616699ab4a79130a38cd4e5ce1f7821867697b1679aff3d952e`. Its 12-column, ordered accepted-population hash is `9b252b44866f94f03fa74423595302e925495a24428f35011567c8991a2d6021`, computed with `pandas.util.hash_pandas_object(frame, index=True)` using the pinned pandas version. Parquet serialization can differ with software versions; preserve the generated manifest and check the logical row identity as well as file bytes when investigating differences.

No rejected file is required for this sequential study. Three descriptive subgroup columns from the combined cohort (`subgroup_fico`, `subgroup_dti`, `subgroup_amount`) are unused by all sequential fits, masks, splits and costs, so the accepted-only reconstruction excludes them.

## Full historical cohort

The historical Parts A/B require both accepted and rejected sources. Follow the `real_cvar.cli prepare` command in [experiments.md](experiments.md), using a separate reproduction configuration so the published metadata is preserved.

The observed cohort is **390,644 applications: 53,367 mature accepted loans and 337,277 rejected applications**, with **8,644 defaults among accepted loans**. The `expected_counts` entries 27,157/206,161 in the frozen configuration are older illustrative values, not observed counts. The historical loader warns about that difference and records actual counts; those entries are retained to preserve the scientific configuration's hash.

Accepted loans are filtered by 2012 issue date and rejected applications by 2012 application date. Accepted labels are 1 for `Charged Off`/`Default` and 0 for `Fully Paid`; other loan statuses are excluded. Rejected labels remain missing. Model features are requested amount, DTI, FICO/risk score, employment length, state, three-digit ZIP prefix and normalized purpose. Calendar month, Policy Code, underwriting fields and repayment fields are not predictors. See the source metadata for the precise schema mappings. Part A's extremely limited overlap and unsupported applicant regions are empirical diagnostics, not evidence that its assumptions hold everywhere.

The restored original combined Parquet file had 390,644 rows, 15 columns, 5,215,764 bytes and SHA-256 `6a9a44ee9338068d50bf3e0e773a0debab9cd1816369c8dbed639a0f15817f86`. Its accepted rows match the reconstructed input exactly in values, missingness, dtypes and order on all 12 shared columns. [cohort_identity_evidence.json](../camera_ready_v3_10page/review/cohort_identity_20261107/cohort_identity_evidence.json) records this local comparison and all 20 mask/split hashes. It compared local restored files and supplied historical cluster hashes; it was not a new remote read or a rerun of the experiments.

## Synthetic sources

The finite-cell and population simulations generate their own data from committed seeds. The semi-synthetic case uses scikit-learn's bundled `load_diabetes()` dataset; no separate network acquisition is performed. It imposes a synthetic label/selection model for demonstration and is not an evaluation of clinical deployment.
