"""Audit the original rejected snapshot with the unchanged historical standardizer."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO / 'real_data_evaluation' / 'src'))
import numpy as np
import pandas as pd
from real_cvar.data import inspect_schema, _standardize_chunk, _numeric, add_subgroups

SOURCE = ROOT / 'rejected_2007_to_2018Q4.csv.gz'
ORIGINAL_CODE = REPO / 'real_data_evaluation' / 'src' / 'real_cvar' / 'data.py'
mapping = inspect_schema(SOURCE, accepted=False)
metadata = json.loads((REPO / 'real_data_evaluation' / 'artifacts' / 'preparation_metadata.json').read_text())
started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cohort_parts = []
raw_score_parts = []
total_read = 0
raw_purpose_missing = 0
raw_purpose_exact_category = 0
known_categories = {'debt_consolidation','credit_card','home_improvement','small_business',
                    'major_purchase','medical','moving','vacation','wedding','car',
                    'education','renewable_energy','house','other'}
for chunk_number, chunk in enumerate(pd.read_csv(SOURCE, usecols=sorted(set(mapping.values())),
                                               chunksize=100000, low_memory=False), 1):
    total_read += len(chunk)
    dates = pd.to_datetime(chunk[mapping['date']], errors='coerce', format='mixed')
    mask = dates.dt.year.eq(2012)
    if mask.any():
        raw = chunk.loc[mask]
        raw_score_parts.append(_numeric(raw[mapping['risk_score']]).reset_index(drop=True))
        raw_purpose = raw[mapping['purpose']].astype('string').str.lower().str.strip()
        raw_purpose_missing += int(raw_purpose.isna().sum())
        raw_purpose_exact_category += int(raw_purpose.isin(known_categories).sum())
    standardized = _standardize_chunk(chunk, mapping, accepted=False, year=2012)
    if not standardized.empty:
        cohort_parts.append(standardized)
    if chunk_number % 25 == 0:
        print(f'Inspected {total_read:,} source rows; retained {sum(len(x) for x in cohort_parts):,} 2012 rows', flush=True)

data = add_subgroups(pd.concat(cohort_parts, ignore_index=True))
raw_score = pd.concat(raw_score_parts, ignore_index=True)

def counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}

def numeric_summary(series):
    return {'missing': int(series.isna().sum()), 'min': float(series.min()),
            'max': float(series.max()), 'unique_nonmissing': int(series.nunique()),
            'quantiles': {str(k): float(v) for k,v in series.quantile([0,.01,.25,.5,.75,.99,1]).items()}}

fico_counts = counts(data['subgroup_fico'])
expected_rejected_fico = {
    'fico_<660': 180903,
    'fico_660_699': 111100-29010,
    'fico_700_739': 56597-17177,
    'fico_740_plus': 18947-7180,
    'fico_missing': 23097,
}
report = {
    'audit_started_utc': started,
    'audit_completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
    'source_download_provenance': json.loads((ROOT/'download_provenance.json').read_text()),
    'original_standardizer': str(ORIGINAL_CODE.relative_to(REPO)),
    'original_standardizer_sha256': hashlib.sha256(ORIGINAL_CODE.read_bytes()).hexdigest(),
    'method': 'Imported unchanged _standardize_chunk, _numeric, inspect_schema and add_subgroups; chunk size 100000; rejected Application Date year 2012. No fitting.',
    'source_rows_inspected': total_read,
    'source_schema_mapping': mapping,
    'mapping_matches_recorded_export': mapping == metadata['rejected_schema_mapping'],
    'rejected_2012_count': len(data),
    'rejected_count_matches_recorded_export': len(data) == metadata['rejected_count'],
    'fico_group_counts': fico_counts,
    'expected_fico_counts_from_saved_export': expected_rejected_fico,
    'all_fico_counts_match_recorded_export': fico_counts == expected_rejected_fico,
    'raw_risk_score': numeric_summary(raw_score),
    'raw_score_missing': int(raw_score.isna().sum()),
    'raw_score_zero': int(raw_score.eq(0).sum()),
    'raw_score_below_300_nonmissing': int(raw_score.lt(300).sum()),
    'raw_score_above_850_nonmissing': int(raw_score.gt(850).sum()),
    'raw_score_noninteger': int(((raw_score % 1 != 0) & raw_score.notna()).sum()),
    'standardized_risk_score': numeric_summary(data['risk_score']),
    'standardized_numeric_missing_counts': {k:int(data[k].isna().sum()) for k in ['requested_amount','dti','risk_score','employment_years']},
    'state_counts': counts(data['state']),
    'states_excluded_in_early_2012_filing_counts': {s:int(data['state'].eq(s).sum()) for s in ['ID','IN','IA','ME','MS','NE','ND','TN']},
    'amount_summary': numeric_summary(data['requested_amount']),
    'amount_below_1000': int(data['requested_amount'].lt(1000).sum()),
    'amount_above_35000': int(data['requested_amount'].gt(35000).sum()),
    'dti_summary': numeric_summary(data['dti']),
    'dti_ge_35': int(data['dti'].ge(35).sum()),
    'purpose_mapping_summary': {
        'source_field':'Loan Title',
        'normalized_category_counts': counts(data['purpose']),
        'raw_title_missing':raw_purpose_missing,
        'raw_title_exactly_an_accepted_style_category':raw_purpose_exact_category,
        'privacy':'Only normalized category aggregates retained; no free text titles, ZIPs or row identifiers exported.'},
    'application_month_counts': counts(data['application_month']),
    'policy_code_counts': counts(data['policy_code']),
    'limitations': [
        'Count reconciliation does not establish bitwise identity to the original July 2026 input: its raw-source hash was not recorded.',
        'Rejected file raw audit only; all 53367 accepted records at score >=660 are established independently by saved subgroup counts until accepted-source audit is available.',
        'State/DTI counts are diagnostics, not a redefined supported target; effective dates and joint underwriting eligibility are not verified.',
        'No original data/predictions/results modified and no model fitted. Explicit extrapolation interpretation retained.',
    ],
}
(ROOT/'raw_rejected_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
lines = [
    'Raw rejected LendingClub support check',
    f"2012 rejected records: {len(data):,}; recorded export: {metadata['rejected_count']:,}; exact match: {report['rejected_count_matches_recorded_export']}",
    f"FICO strata: {fico_counts}; exact export reconstruction match: {report['all_fico_counts_match_recorded_export']}",
    f"Raw score missing/zero/outside [300,850]: {report['raw_score_missing']}/{report['raw_score_zero']}/{report['raw_score_below_300_nonmissing']+report['raw_score_above_850_nonmissing']}",
    f"Standardized score range: {data['risk_score'].min()}--{data['risk_score'].max()}",
    f"Schema matches saved export: {report['mapping_matches_recorded_export']}",
    'See raw_rejected_audit.json for source SHA-256, chronology, original-code hash, complete aggregate counts and limits.',
    'No re-fitting, supported-cohort redesign, original-source mutation or exposure of individual free-text values.',
]
(ROOT/'raw_rejected_audit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines), flush=True)
