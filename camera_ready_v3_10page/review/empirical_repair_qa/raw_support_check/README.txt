Raw-source follow-up to historical_support_audit.txt (2026-11-07 JST)

The original audit's raw-data availability limitation is superseded by this new
read-only check. The explicitly specified historical extrapolation model is retained.
No original experiment files, data, predictions or manuscript files were modified.

Downloaded rejected_2007_to_2018Q4.csv.gz from Kaggle dataset version 3, with source
URL and full download metadata in download_provenance.json. File size: 255470782
bytes. SHA-256: c1f9f2a4ec5537ef0af44b24b2fb2f41f58fcaec45363ef32c6a2fb27347f6c5.
The local compressed source is excluded from Git by this directory's .gitignore.

audit_rejected.py scanned all 27648741 source rows in chunks of 100000 using the
original data.py standardization functions, imported read-only without bytecode
writes. Python 3.12.4, pandas 2.3.2 and NumPy 1.26.4 were available in the system
environment; this audit did not touch the separate sequential-experiment environment.

Directly verified rejected cohort:
  2012 rejections                  337277
  FICO below 660                   180903
  FICO 660-699                      82090
  FICO 700-739                      39420
  FICO 740+                         11767
  Standardized score missing        23097
The last count comprises 4338 raw missing values and 18759 zero sentinels. All other
raw 2012 scores are integer-valued within [385,850]. Schema mappings and every FICO
stratum exactly reconcile with the saved historical export and its reconstruction.

compare_accepted.py independently read the other agent's reconstructed accepted
Parquet, without modifying it. It verifies 53367 mature loans, 8644 defaults,
FICO midpoint range [662,847.5], zero missing scores and zero scores below 660.
All three accepted FICO subgroup counts match the saved export. Combined N=390644;
numeric missing counts (amount=2, DTI=4226, score=23097, employment=5509) exactly
match every corresponding saved preparation-metadata fraction.

The FICO finding is thus directly confirmed in reconstructed source records,
in addition to the contemporaneous 660 minimum documented in the earlier audit.
This supports the manuscript's explicit extrapolation interpretation. It does not
establish a supported replacement cohort or validate the learned reference risks.

Additional measurement/support diagnostics are deliberately not used to redefine
the analysis:
  * Rejected amounts range from 1000 to 1200000; 169 exceed 35000. The two missing
    standardized amounts remain missing. Accepted loan_amnt can be reduced during
    underwriting, so the fields do not establish identical predecision amounts.
  * Only normalized purpose-category aggregates are exported. No individual titles,
    ZIPs, source-row identifiers or other applicant-level examples are printed.
  * Rejected Policy Code is uniformly 0.0 in 2012; the generic first-party dictionary
    describes codes 1/2. This field was excluded from modeling and is not used to
    assert eligibility.
  * Accepted records include IN=26, NE=1, MS=1 despite dated state exclusion lists.
    Therefore no blanket state-based structural claim or state filter is justified
    by this audit. Exact eligibility timing and these discrepancies need separate
    resolution before such a claim. Rejected counts by state and DTI are saved as
    diagnostics, not as an eligibility-defined supported population.

Follow-up on 7 November 2026: the July raw-source hashes supplied from the cluster
match newly recomputed local SHA-256 values for both raw files. Bitwise source
identity with those reported cluster copies is therefore established. The original
prepared cohort has also been restored at the repository-relative path
real_data_evaluation/artifacts/prepared/lendingclub_2012.parquet (5,215,764 bytes;
390,644 rows; 15 columns), SHA-256
6a9a44ee9338068d50bf3e0e773a0debab9cd1816369c8dbed639a0f15817f86.
Its 53,367 accepted rows exactly equal the reconstructed rows on all 12 shared
columns, in the original order, and it confirms the rejected support counts above.
The three extra columns are unused subgroup labels. This establishes input identity,
not bitwise reproduction of fitted predictions across computing environments.
The cluster was not reread during this follow-up; its hashes came from the supplied
restoration report. Evidence is under review/cohort_identity_20261107/ relative to
the camera-ready directory. Original run and download manifests remain unchanged.

Artifacts:
  download_provenance.json             URL, timing, bytes and source SHA-256
  raw_rejected_audit.json/.txt          complete rejected aggregate audit
  combined_source_reconciliation.json  accepted provenance and joint reconciliation
  download_rejected.py                 source retrieval script
  audit_rejected.py                    unchanged-loader chunk audit
  compare_accepted.py                  accepted/combined aggregate checks
