EDITABLE SCIENTIFIC FIGURES

Main sources / vector PDFs
  fig01_concept        Shared observations, compatible hidden worlds, certificate
  fig02_learning       Oracle interval widths and learned-endpoint RMSE
  fig03_assessment     Historical applicants and empirical masking coverage
  fig04_revelation      Fixed-objective and sequential audit precision

Supplementary sources / PDFs
  supp_frontier        Exact synthetic frontier and all baseline costs
  supp_diagnostics     Observed-tail sample scale and synthetic sensitivity
  supp_sensitivity     All historical sensitivity curves
  supp_coverage        Coverage by predictor and tail level
  supp_revelation      All six acquisition rules, width and predictive CVaR

All artwork is native TikZ/PGFPlots; no raster plots are embedded. Each source
uses figure_style.tex and produces a vector PDF 5.5 inches wide, the exact
manuscript text width (397.48499 TeX pt / 396 PDF bp). Essential labels are
at least 8 PDF pt. Main figures have two panels. Figure 1 is a conceptual
diagram with no plotted axes. Blue / gray / orange roles are shared throughout;
shapes and line styles preserve distinctions in grayscale.

BUILD (from camera_ready_v3_10page)
  powershell -ExecutionPolicy Bypass -File figures/build_figures.ps1
The script compiles all nine standalone sources and updates the PDFs used by
main.tex. It needs pdflatex, TikZ/PGFPlots, standalone, newtx, and helvet.
The included CSVs suffice for rebuilding figures without the original data.

Optional data regeneration from original results, without running experiments:
  python figures/assessment_extract.py
  python figures/revelation_extract.py
  python figures/scripts/prepare_diagnostics.py
The first two scripts write source hashes and transformation details to
assessment_provenance.json and revelation_provenance.json. The original
experiment and real_data_evaluation directories are read-only inputs.

SCIENTIFIC CONVENTIONS
Figure 1: squared loss, fixed f=1/5, pi=1/2, p1=1/3, Gamma=2, alpha=1/2.
Hidden endpoint probabilities are 1/5 and 1/2. Their population probabilities
are 4/15 and 5/12, giving CVaR 9/25=0.36 and 27/50=0.54. Identical blue
sections denote the same observed law; the two parametric surfaces denote
alternative hidden completions. Their silhouettes, projected cross-sections,
and mesh curves share one parameterization in fig01_concept.tex. Shapes,
areas, and volumes do not encode probabilities or additional risk dimensions.
A transparent oblique sleeve with closed end rims surrounds a dark risk segment for
[0.36,0.54]. Pale longitudinal extensions depict statistical enlargement,
with A/B mapped to the dark segment's endpoints. Sleeve depth and curvature
are illustrative only. Blue identifies A / 0.36 and burnt orange identifies
B / 0.54 in the world headings, mapping curves, endpoint markers, and interval
notation. The interval notation is 10.5 TeX pt. Surface geometry is unchanged.
This is the selected abstract geometric option, copied unchanged from
the final manuscript source. All geometry is defined in fig01_concept.tex;
earlier alternative concepts are not required or bundled in this release.

Figure 2: finite_sample_summary.csv supplies the original 4,800 replications
per sample size. Fill is the difference of mean widths, not a confidence band.
The n^-1/2 guide is scaled to 0.055 at n=1,000 for comparison only.

Figure 3: all-applicant estimates at alpha=.95, Gamma=2; coverage is descriptive
over 20 shared masking repetitions and three predictors, not 60 independent
masks. At Gamma3: 40/60 plug-in versus 59/60 fold-enlarged coverage.

Figure 4: fixed gaps are normalized within each mask before taking medians.
Final reductions are 61.1177%, 16.0230%, and 9.0305% for TIV, random, and
lowest propensity. This replaces the old table's 10% lowest-propensity figure,
which used a different aggregation. Nominal final budget is 16%; actual
fractions include integer batch rounding (about 16.013% fixed, 16.018% sequential).
Sequential widths are medians of the recorded absolute intervals. Fixed budgets use
the original hidden pool; sequential budgets use the initially hidden 80% training
pool. Both batch sizes and plotted fractions use their respective initial counts.

Supplement bands and bars are explicitly described in manuscript captions:
IQR across masks for selected sequential curves; IQR across worlds for the
synthetic frontier and within-world cost ratios. They are not confidence bands.
No smoothed or fabricated trajectories are used. Diagnostic OLS is a guide
fitted to the 54 recorded simulations, not an additional theoretical guarantee.

PREVIEW / PRESERVATION
Optional preview and vector checks, after compiling the nine figures:
  python figures/scripts/preview_and_check.py
This creates color/grayscale PNGs in previews/, a four-page main-figure
figures_overview.pdf, and final_qa.json with size, text-bound, and raster checks.
The current paper is ../main.pdf. The guide and Appendices A-E are in
../appendices/. See ../../docs/figures.md for cross-platform commands,
dependencies, source-to-figure mapping, and reproducibility checks.
