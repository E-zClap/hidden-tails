"""Export recorded audit/frontier data after validating a completed plug-in rerun.

Run from any directory: python camera_ready_v3_10page/figures/revelation_extract.py
No experiment is executed. No output is written until all input checks succeed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PACKAGE = ROOT / 'camera_ready_v3_10page/revision_experiments/sequential_plugin'
DEFAULT_RESULTS = PACKAGE / 'results/optimized'
POLICIES = ['tiv_per_cost', 'random', 'low_propensity', 'predicted_loss', 'uncertainty', 'sensitivity_width']
METRICS = {'width': 'estimated_retained_width', 'brier': 'mean_test_brier', 'cvar': 'true_test_cvar'}
KEYS = ['repetition', 'policy', 'round']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(directory, name):
    # Archived manifests were recorded on Windows; accept their separators on
    # POSIX as well, while retaining the containment check below.
    path = (directory / name.replace('\\', '/')).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f'Manifest path leaves its directory: {name}')
    return path


def validate_grid(frame, description):
    expected = pd.MultiIndex.from_product([range(20), POLICIES, range(9)], names=KEYS).sort_values()
    if len(frame) != 1080 or frame.duplicated(KEYS).any():
        raise ValueError(f'{description} must have 1080 unique mask/policy/round rows.')
    actual = pd.MultiIndex.from_frame(frame[KEYS]).sort_values()
    if not actual.equals(expected):
        raise ValueError(f'{description} must contain all 20 masks, six policies, and rounds 0--8 (1080 rows).')


def load_completed_run(results, package=PACKAGE):
    """Check provenance, the complete trajectory grid, and stored terminal pairs."""
    manifest_path = results / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Sequential manifest is not complete; existing figure data are retained.')
    if manifest.get('rows') != 1080 or sorted(manifest.get('completed_repetitions', [])) != list(range(20)):
        raise ValueError('Completed manifest must record 20 masks and 1080 rows.')
    identity = manifest['identity']
    if identity.get('estimator') != 'retained_plugin' or identity.get('orthogonal') is not False:
        raise ValueError('Sequential manifest must specify retained_plugin with orthogonal=False.')
    cfg = identity['config']['part_b']
    if (cfg['repetitions'] != 20 or cfg['adaptive_rounds'] != 8
            or set(cfg['policies']) != set(POLICIES) or cfg['audit_cost_amount_weight'] != 0):
        raise ValueError('Sequential configuration must retain 20 masks, six policies, eight rounds, and unit costs.')
    sources = identity.get('sources_sha256', {})
    if not sources:
        raise ValueError('Sequential manifest has no source hashes.')
    for name, expected in sources.items():
        if sha256(checked_path(package, name)) != expected:
            raise ValueError(f'Sequential source hash mismatch: {name}')
    hashes = manifest.get('outputs_sha256', {})
    required_outputs = {'sequential_plugin_runs.csv', 'trajectory_summary.csv', 'terminal_pairs.csv',
                        'terminal_comparison.csv', 'constant_reference.csv'}
    if not required_outputs.issubset(hashes):
        raise ValueError('Completed manifest is missing required output hashes.')
    for name, expected in hashes.items():
        if sha256(checked_path(results, name)) != expected:
            raise ValueError(f'Sequential output hash mismatch: {name}')

    runs = pd.read_csv(results / 'sequential_plugin_runs.csv').sort_values(KEYS)
    required_columns = set(KEYS + list(METRICS.values()) + [
        'alpha', 'gamma', 'audited_labels', 'audited_fraction_hidden', 'cumulative_cost',
        'estimated_retained_lower', 'estimated_retained_upper', 'estimator'])
    if not required_columns.issubset(runs.columns):
        raise ValueError(f'Missing sequential columns: {sorted(required_columns.difference(runs.columns))}')
    validate_grid(runs, 'Sequential data')
    if set(runs.estimator) != {'retained_plugin'} or not np.allclose(runs.alpha, .95) or not np.allclose(runs.gamma, 3):
        raise ValueError('Sequential rows must use retained_plugin, alpha=.95, and Gamma=3.')
    numeric = list(METRICS.values()) + ['estimated_retained_lower', 'estimated_retained_upper',
                                     'audited_labels', 'audited_fraction_hidden', 'cumulative_cost']
    if not np.isfinite(runs[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Sequential data contain non-finite metrics or budgets.')
    bounded = list(METRICS.values()) + ['estimated_retained_lower', 'estimated_retained_upper', 'audited_fraction_hidden']
    if ((runs[bounded] < -1e-12) | (runs[bounded] > 1 + 1e-12)).any().any():
        raise ValueError('Sequential risks, widths, and audit fractions must lie in [0,1].')
    if not np.allclose(runs.estimated_retained_upper - runs.estimated_retained_lower,
                       runs.estimated_retained_width, rtol=0, atol=1e-12):
        raise ValueError('Sequential endpoint differences disagree with recorded widths.')
    if (runs.mean_test_brier > runs.true_test_cvar + 1e-12).any():
        raise ValueError('Held-out mean Brier loss exceeds its upper-tail CVaR.')
    if (not np.allclose(runs.audited_labels, np.round(runs.audited_labels), rtol=0, atol=1e-12)
            or (runs.audited_labels < 0).any()
            or not np.allclose(runs.cumulative_cost, runs.audited_labels, rtol=0, atol=1e-12)):
        raise ValueError('Sequential unit-cost budgets disagree with audited label counts.')
    for _, group in runs.groupby(['repetition', 'round']):
        for column in ['audited_labels', 'audited_fraction_hidden', 'cumulative_cost']:
            if not np.allclose(group[column], group[column].iloc[0], rtol=0, atol=1e-12):
                raise ValueError('Sequential policies have mismatched label budgets within a mask and round.')
    for _, group in runs.groupby(['repetition', 'policy']):
        if group.audited_labels.iloc[0] != 0 or (np.diff(group.audited_labels) < 0).any():
            raise ValueError('Sequential audit counts must start at zero and increase with rounds.')

    pairs = pd.read_csv(results / 'terminal_pairs.csv')
    expected_pairs = pd.MultiIndex.from_product([range(20), list(METRICS.values())], names=['repetition', 'metric']).sort_values()
    if (len(pairs) != 60 or pairs.duplicated(['repetition', 'metric']).any()
            or not pd.MultiIndex.from_frame(pairs[['repetition', 'metric']]).sort_values().equals(expected_pairs)
            or not pairs['round'].eq(8).all()):
        raise ValueError('Stored terminal pairs must contain all 20 masks and three metrics at round 8.')
    terminal = runs[runs['round'].eq(8)].set_index(['repetition', 'policy'])
    for row in pairs.itertuples():
        tiv = terminal.loc[(row.repetition, 'tiv_per_cost')]
        random = terminal.loc[(row.repetition, 'random')]
        expected = [tiv[row.metric], random[row.metric], tiv[row.metric] - random[row.metric],
                    tiv.audited_labels, tiv.audited_fraction_hidden, tiv.cumulative_cost, random.cumulative_cost]
        recorded = [row.tiv_value, row.random_value, row.tiv_minus_random, row.audited_labels,
                    row.audited_fraction_hidden, row.tiv_cumulative_cost, row.random_cumulative_cost]
        if not np.allclose(recorded, expected, rtol=0, atol=1e-12):
            raise ValueError('Stored terminal pair disagrees with the raw sequential trajectory.')
    comparison = pd.read_csv(results / 'terminal_comparison.csv').set_index('metric')
    if len(comparison) != 3 or set(comparison.index) != set(METRICS.values()):
        raise ValueError('Terminal comparison must contain each of the three metrics once.')
    for metric in METRICS.values():
        values = pairs[pairs.metric.eq(metric)]
        row = comparison.loc[metric]
        if (row.n_pairs != 20 or row['round'] != 8
                or not np.allclose([row.tiv_median, row.random_median, row.paired_difference_median],
                                   [values.tiv_value.median(), values.random_value.median(),
                                    values.tiv_minus_random.median()], rtol=0, atol=1e-12)):
            raise ValueError('Terminal comparison disagrees with validated terminal pairs.')
    paths = [manifest_path, *[results / name for name in hashes]]
    return runs, pairs, comparison, manifest, paths


def terminal_table(comparison):
    """Medians across masks, followed by the median of paired differences."""
    metrics = list(METRICS.values())
    lines = [r'\begin{tabular}{lccc}', r'\toprule',
             r'Acquisition rule & Estimated width & Mean Brier & CVaR\\', r'\midrule']
    for label, column in [('TIV', 'tiv_median'), ('Random', 'random_median')]:
        lines.append(label + ' & ' + ' & '.join(f'{comparison.loc[metric, column]:.4f}' for metric in metrics) + r'\\')
    lines.append(r'Paired TIV$-$Random & ' + ' & '.join(
        f'${comparison.loc[metric, "paired_difference_median"]:+.4f}$' for metric in metrics) + r'\\')
    lines.extend([r'\midrule', r'\multicolumn{4}{l}{\textit{Exact predictor reference}}\\',
                  r'Constant $f\equiv1/2$ & 0.0000 & 0.2500 & 0.2500\\', r'\bottomrule', r'\end{tabular}',
                  r'\par\smallskip{\footnotesize The constant forecast supplies an exact predictor reference.}', ''])
    return '\n'.join(lines)


def extract(results=None, check_only=False):
    results = DEFAULT_RESULTS if results is None else Path(results).resolve()
    a, pairs, comparison, run_manifest, sequential_sources = load_completed_run(results)
    real = ROOT / 'real_data_evaluation/artifacts/tables'
    syn = ROOT / 'experiments/results'
    sources = [real / 'part_b_fixed_audits.csv', real / 'part_b_fixed_audit_summary.csv',
               syn / 'experiment6_certification_frontier_curves.csv', syn / 'experiment6_cost_to_certify_ratios.csv',
               *sequential_sources]
    f = pd.read_csv(sources[0]).sort_values(KEYS)
    validate_grid(f, 'Fixed-audit data')
    initial = f[f['round'].eq(0)].set_index(['repetition', 'policy']).fixed_threshold_width
    if not np.isfinite(initial).all() or not (initial > 0).all():
        raise ValueError('Fixed-audit normalization requires positive finite initial gaps.')
    f['remaining_percent'] = [100 * row.fixed_threshold_width / initial.loc[(row.repetition, row.policy)] for row in f.itertuples()]
    outputs = {'revelation_fixed_runs.csv': f, 'revelation_sequential_runs.csv': a}
    endpoints, starting = {}, {}
    for policy in POLICIES:
        rows = []
        for round_id in range(9):
            fs = f[f.policy.eq(policy) & f['round'].eq(round_id)]
            aa = a[a.policy.eq(policy) & a['round'].eq(round_id)]
            row = {'round': round_id, 'fixed_budget': 100 * fs.audited_fraction_hidden.median(),
                   'sequential_budget': 100 * aa.audited_fraction_hidden.median()}
            for name, frame, column in [('fixed', fs, 'remaining_percent'), *[(name, aa, column) for name, column in METRICS.items()]]:
                for suffix, quantile in [('q25', .25), ('median', .5), ('q75', .75)]:
                    row[name + '_' + suffix] = frame[column].quantile(quantile)
            rows.append(row)
        summary = pd.DataFrame(rows)
        outputs[f'revelation_{policy}.csv'] = summary
        if policy in ['tiv_per_cost', 'random', 'low_propensity']:
            for metric in METRICS:
                outputs[f'revelation_band_{policy}_{metric}.csv'] = pd.concat([
                    pd.DataFrame({'budget': summary.sequential_budget, 'value': summary[metric + '_q25']}),
                    pd.DataFrame({'budget': summary.sequential_budget[::-1], 'value': summary[metric + '_q75'][::-1]})])
        starting[policy], endpoints[policy] = rows[0], rows[-1]

    c, ratios = pd.read_csv(sources[2]), pd.read_csv(sources[3])
    if c.world.nunique() != 120:
        raise ValueError('Synthetic frontier must contain 120 recorded worlds.')
    tiv = c[c.policy.eq('TIV/cost')].sort_values(['world', 'budget_fraction']).relative_debt.to_numpy()
    oracle = c[c.policy.eq('Oracle frontier')].sort_values(['world', 'budget_fraction']).relative_debt.to_numpy()
    if not np.array_equal(tiv, oracle):
        raise ValueError('Recorded synthetic TIV and oracle frontiers do not coincide.')
    outputs.update({'frontier_recorded_curves.csv': c, 'frontier_recorded_cost_ratios.csv': ratios})
    name_map = {'TIV/cost': 'tiv', 'Oracle frontier': 'oracle', 'Random': 'random', 'High density': 'density',
                'Low reveal prob.': 'low_propensity', 'High pred. loss': 'loss', 'Uncertainty': 'uncertainty',
                'Width only': 'width', 'Contrast only': 'contrast'}
    for policy, name in name_map.items():
        summary = pd.DataFrame([dict(budget=100 * budget, median=100 * group.relative_debt.median(),
                                    q25=100 * group.relative_debt.quantile(.25), q75=100 * group.relative_debt.quantile(.75))
                                for budget, group in c[c.policy.eq(policy)].groupby('budget_fraction')])
        outputs[f'frontier_{name}.csv'] = summary
        if name == 'tiv':
            outputs['frontier_tiv_band.csv'] = pd.concat([
                pd.DataFrame({'budget': summary.budget, 'value': summary.q25}),
                pd.DataFrame({'budget': summary.budget[::-1], 'value': summary.q75[::-1]})])
    cost_rows = []
    for index, policy in enumerate(list(name_map)[2:]):
        current = ratios[ratios.policy.eq(policy) & ratios.tolerance.eq(.25)].ratio
        median, low, high = current.median(), current.quantile(.25), current.quantile(.75)
        cost_rows.append(dict(policy=policy, y=6-index, median=median, q25=low, q75=high, minus=median-low, plus=high-median))
    outputs['frontier_target_costs.csv'] = pd.DataFrame(cost_rows)
    manifest = {
        'sources': {str(path.relative_to(ROOT)).replace('\\', '/'): {'sha256': sha256(path)} for path in sources},
        'sequential_run_identity': run_manifest['identity'],
        'fixed_aggregation': 'Normalize fixed_threshold_width by its round-0 value within each mask and policy; report medians and linear IQR across 20 masks at each recorded round.',
        'budget': (f'100 times the median recorded audited_fraction_hidden. Final fractions are '
                   f'{endpoints["tiv_per_cost"]["fixed_budget"]:.9g}% (fixed) and '
                   f'{endpoints["tiv_per_cost"]["sequential_budget"]:.9g}% (sequential), including integer batch rounding.'),
        'sequential_aggregation': 'Completed retained_plugin run with orthogonal=False. Medians and linear IQR across 20 masks; width uses estimated_retained_width, Brier uses mean_test_brier, and held-out CVaR uses true_test_cvar. All 1080 rows and source/output hashes are validated before export.',
        'terminal_comparison': 'TIV and random share each mask, split, terminal label count, and unit cost. Policy rows are medians across masks; the paired row is the median of the 20 within-mask TIV-minus-random differences, validated against terminal_pairs.csv and terminal_comparison.csv. The constant forecast is an exact predictor reference, not an acquisition policy.',
        'frontier_aggregation': 'Per-world relative_debt medians and linear IQR across 120 worlds; TIV and the exact oracle agree at every recorded coordinate. All nine raw and summary trajectories are exported.',
        'cost_target': 'First recorded cost reaching 25% of initial ambiguity divided by TIV cost within each world; report median and IQR across 120 worlds for all seven baselines.',
        'initials': starting, 'endpoints': endpoints,
        'main_caption': (f'Targeted revelation and assessment precision. Fixed-gap reductions at the terminal recorded budget: '
                         f'TIV {100-endpoints["tiv_per_cost"]["fixed_median"]:.0f}%, random {100-endpoints["random"]["fixed_median"]:.0f}%. '
                         f'Sequential median estimated retained width: initial TIV {starting["tiv_per_cost"]["width_median"]:.5f}, '
                         f'terminal TIV {endpoints["tiv_per_cost"]["width_median"]:.5f}, random {endpoints["random"]["width_median"]:.5f}, '
                         f'lowest propensity {endpoints["low_propensity"]["width_median"]:.5f}. '
                         'Twenty controlled masks, gradient boosting, alpha=.95, Gamma=3, unit label costs; points are recorded rounds.'),
        'supp_revelation_caption': 'Sequential outcomes across six acquisition rules. Lines are medians over 20 masks; light bands are IQR for TIV, random, and lowest propensity. Width uses retained plug-in objectives; predictive CVaR and mean Brier loss use the reserved 20% split. Alpha=.95 and Gamma=3.',
        'supp_frontier_caption': 'Revelation precision and audit cost across 120 synthetic finite-cell worlds. Median remaining fixed-threshold ambiguity is normalized within world. TIV per cost matches the exact fractional-knapsack frontier. Cost ratios compare first recorded cost reaching 25% of the initial gap to TIV within the same world; intervals are IQR.',
    }
    table = terminal_table(comparison)
    provenance_text = json.dumps(manifest, indent=2) + '\n'
    # All reads, validation, aggregation, and serialization preparation precede writes.
    if not check_only:
        raw_names = {'revelation_fixed_runs.csv', 'revelation_sequential_runs.csv', 'frontier_recorded_curves.csv', 'frontier_recorded_cost_ratios.csv'}
        for name, frame in outputs.items():
            frame.to_csv(OUT / name, index=False, float_format=None if name in raw_names else '%.10g')
        (OUT / 'sequential_terminal_table.tex').write_text(table, encoding='utf-8')
        (OUT / 'revelation_provenance.json').write_text(provenance_text, encoding='utf-8')
    return {'validated_sequential_rows': len(a), 'masks': a.repetition.nunique(),
            'written': not check_only, 'initials': starting, 'endpoints': endpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequential-results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--check-only', action='store_true', help='Validate all sources without writing figure data.')
    args = parser.parse_args()
    try:
        result = extract(args.sequential_results, args.check_only)
    except (ValueError, KeyError, OSError) as error:
        print(f'Extraction refused: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
