"""Reconcile raw rejected aggregates against the independently rebuilt accepted cohort."""
from pathlib import Path
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO/'real_data_evaluation'/'src'))
import pandas as pd
from real_cvar.data import add_subgroups

rejected = json.loads((ROOT/'raw_rejected_audit.json').read_text())
accepted_path = REPO/'camera_ready_v3_10page'/'revision_experiments'/'sequential_plugin'/'data'/'accepted_2012_reconstructed.parquet'
accepted = add_subgroups(pd.read_parquet(accepted_path))
accepted_manifest = json.loads((accepted_path.parent/'source_manifest.json').read_text())
original = json.loads((REPO/'real_data_evaluation'/'artifacts'/'preparation_metadata.json').read_text())
expected_accepted_fico = {'fico_660_699':29010,'fico_700_739':17177,'fico_740_plus':7180}
accepted_fico = {str(k):int(v) for k,v in accepted['subgroup_fico'].value_counts().sort_index().items()}
numeric_missing = {}
for key in ['requested_amount','dti','risk_score','employment_years']:
    rejected_n = rejected['standardized_numeric_missing_counts'][key]
    accepted_n = int(accepted[key].isna().sum())
    expected = round(original['total_count']*original['feature_missing_fraction'][key])
    numeric_missing[key] = {'accepted':accepted_n,'rejected':rejected_n,
                            'combined':accepted_n+rejected_n,'saved_export':expected,
                            'matches':accepted_n+rejected_n==expected}
early_excluded = ['ID','IN','IA','ME','MS','NE','ND','TN']
state_comparison = {s:{'accepted':int(accepted['state'].eq(s).sum()),
                       'rejected':rejected['state_counts'].get(s,0)} for s in early_excluded}
report = {
    'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'accepted_prepared_path':str(accepted_path.relative_to(REPO)),
    'accepted_prepared_sha256':hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
    'accepted_source_sha256':accepted_manifest['source_sha256'],
    'accepted_source_bytes':accepted_manifest['source_bytes'],
    'accepted_loader_sha256':accepted_manifest['loader_sha256'],
    'accepted_count':len(accepted),
    'accepted_defaults':int(accepted['Y'].sum()),
    'accepted_count_and_defaults_match_saved_export':len(accepted)==original['accepted_mature_count'] and int(accepted['Y'].sum())==original['default_count_among_accepted'],
    'accepted_score_min':float(accepted['risk_score'].min()),
    'accepted_score_max':float(accepted['risk_score'].max()),
    'accepted_missing_scores':int(accepted['risk_score'].isna().sum()),
    'accepted_below_660':int(accepted['risk_score'].lt(660).sum()),
    'accepted_fico_counts':accepted_fico,
    'accepted_fico_counts_match_saved_export':accepted_fico==expected_accepted_fico,
    'combined_count':len(accepted)+rejected['rejected_2012_count'],
    'combined_numeric_missing_reconciliation':numeric_missing,
    'early_2012_excluded_state_counts':state_comparison,
    'accepted_amount_min':float(accepted['requested_amount'].min()),
    'accepted_amount_max':float(accepted['requested_amount'].max()),
    'accepted_dti_max':float(accepted['dti'].max()),
    'accepted_dti_ge_35':int(accepted['dti'].ge(35).sum()),
    'accepted_policy_code_counts':{str(k):int(v) for k,v in accepted['policy_code'].value_counts(dropna=False).items()},
    'interpretation':[
        'Direct source reconstruction confirms the prior aggregate FICO arithmetic, including zero accepted below 660.',
        'Rare accepted records in IN, NE and MS conflict with a naive application of the dated exclusion lists; do not claim all state-listed records structurally excluded or define support from a blanket state filter.',
        'The rejected snapshot has Policy Code=0.0 throughout 2012, whereas the generic dictionary describes codes 1/2. The feature was excluded from original modeling; do not use it as eligibility proof.',
        'Actual accepted requested_amount is loan_amnt, potentially reduced in underwriting per the dictionary; it does not establish the rejected original amount under the same measurement.',
        'Original July 2026 raw source hashes are unavailable: exact count/missingness/FICO reconciliation confirms compatibility, not bitwise identity or exact reproduction of original model predictions.',
        'The explicitly specified extrapolation model remains the historical design; this audit does not introduce a supported cohort or new fitted models.',
    ],
}
(ROOT/'combined_source_reconciliation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['accepted_count','accepted_defaults','accepted_count_and_defaults_match_saved_export','accepted_score_min','accepted_score_max','accepted_missing_scores','accepted_below_660','accepted_fico_counts_match_saved_export','combined_count','combined_numeric_missing_reconciliation','early_2012_excluded_state_counts']},indent=2))
