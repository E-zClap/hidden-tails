"""Ingestion regression tests using temporary, explicitly synthetic fixtures only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
POLICIES = ['tiv_per_cost', 'random', 'low_propensity', 'predicted_loss', 'uncertainty', 'sensitivity_width']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='revelation_ingestion_fixture_')
        self.root = Path(self.temp.name)
        self.out = self.root / 'camera_ready_v3_10page' / 'figures'
        self.out.mkdir(parents=True)
        shutil.copy2(HERE / 'revelation_extract.py', self.out / 'revelation_extract.py')
        self.package = self.root / 'camera_ready_v3_10page/revision_experiments/sequential_plugin'
        self.results = self.package / 'results/optimized'
        self.results.mkdir(parents=True)
        (self.package / 'run.py').write_text('# Synthetic test fixture; never a real experiment.\n')
        self.manifest = {
            'status': 'complete', 'rows': 1080, 'completed_repetitions': list(range(20)),
            'identity': {'estimator': 'retained_plugin', 'orthogonal': False,
                         'sources_sha256': {'run.py': digest(self.package / 'run.py')},
                         'config': {'part_b': {'repetitions': 20, 'adaptive_rounds': 8,
                                              'policies': POLICIES, 'audit_cost_amount_weight': 0.0}}},
        }
        rows = []
        for rep in range(20):
            for policy in POLICIES:
                width = ([.1, .1, .8] if policy == 'tiv_per_cost' else [.05, .2, .3])[rep % 3]
                for round_id in range(9):
                    rows.append(dict(repetition=rep, policy=policy, round=round_id,
                                     alpha=.95, gamma=3.0, audited_labels=2 * round_id,
                                     audited_fraction_hidden=.02 * round_id, cumulative_cost=2.0 * round_id,
                                     estimated_retained_lower=.05, estimated_retained_upper=.05 + width,
                                     estimated_retained_width=width, mean_test_brier=.15 + .001 * rep,
                                     true_test_cvar=.70 + .001 * rep, estimator='retained_plugin',
                                     n_train=42693, n_test=10674))
        self.runs = pd.DataFrame(rows)
        spec = importlib.util.spec_from_file_location('fixture_aggregation', PROJECT / 'camera_ready_v3_10page/revision_experiments/sequential_plugin/aggregation.py')
        aggregation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aggregation)
        self.runs.to_csv(self.results / 'sequential_plugin_runs.csv', index=False)
        for name, frame in aggregation.summarize(self.runs, 8, 20).items():
            frame.to_csv(self.results / f'{name}.csv', index=False)
        self.save_manifest()
        real = self.root / 'real_data_evaluation/artifacts/tables'
        real.mkdir(parents=True)
        fixed = self.runs[['repetition', 'policy', 'round', 'audited_fraction_hidden']].copy()
        fixed['fixed_threshold_width'] = 1 - fixed['round'] * .05
        fixed.to_csv(real / 'part_b_fixed_audits.csv', index=False)
        pd.DataFrame({'fixture': [1]}).to_csv(real / 'part_b_fixed_audit_summary.csv', index=False)
        syn = self.root / 'experiments/results'
        syn.mkdir(parents=True)
        names = ['TIV/cost', 'Oracle frontier', 'Random', 'High density', 'Low reveal prob.', 'High pred. loss', 'Uncertainty', 'Width only', 'Contrast only']
        pd.DataFrame([dict(world=world, policy=policy, budget_fraction=budget, relative_debt=1-budget)
                      for world in range(120) for policy in names for budget in [0.0, 1.0]]).to_csv(syn / 'experiment6_certification_frontier_curves.csv', index=False)
        pd.DataFrame([dict(world=world, policy=policy, tolerance=.25, ratio=2.0)
                      for world in range(120) for policy in names[2:]]).to_csv(syn / 'experiment6_cost_to_certify_ratios.csv', index=False)
        self.sentinel = self.out / 'revelation_random.csv'
        self.sentinel.write_text('existing production-shaped output must survive rejection\n')

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        self.manifest['outputs_sha256'] = {path.name: digest(path) for path in self.results.glob('*.csv')}
        (self.results / 'run_manifest.json').write_text(json.dumps(self.manifest))

    def run_extract(self):
        return subprocess.run([sys.executable, '-B', str(self.out / 'revelation_extract.py')],
                              capture_output=True, text=True)

    def assert_rejected_without_writes(self, expected):
        before = {path.name: digest(path) for path in self.out.iterdir()}
        result = self.run_extract()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(expected, result.stderr)
        self.assertEqual(before, {path.name: digest(path) for path in self.out.iterdir()})

    def test_incomplete_manifest_cannot_overwrite_existing_figure_data(self):
        self.manifest['status'] = 'running'
        self.save_manifest()
        self.assert_rejected_without_writes('manifest is not complete')

    def test_hash_mismatch_cannot_overwrite_existing_figure_data(self):
        with (self.results / 'sequential_plugin_runs.csv').open('a') as handle:
            handle.write('\n')
        self.assert_rejected_without_writes('hash mismatch')

    def test_incomplete_mask_policy_round_grid_is_rejected_even_with_valid_hash(self):
        self.runs.iloc[:-1].to_csv(self.results / 'sequential_plugin_runs.csv', index=False)
        self.save_manifest()
        self.assert_rejected_without_writes('1080')

    def test_changed_source_is_rejected(self):
        (self.package / 'run.py').write_text('# changed after run\n')
        self.assert_rejected_without_writes('source hash mismatch')

    def test_wrong_round_is_rejected_even_when_row_count_is_1080(self):
        self.runs.loc[self.runs.index[-1], 'round'] = 9
        self.runs.to_csv(self.results / 'sequential_plugin_runs.csv', index=False)
        self.save_manifest()
        self.assert_rejected_without_writes('rounds 0--8')

    def test_mismatched_policy_budget_is_rejected(self):
        self.runs.loc[self.runs.index[-1], ['audited_labels', 'cumulative_cost']] = 17
        self.runs.to_csv(self.results / 'sequential_plugin_runs.csv', index=False)
        self.save_manifest()
        self.assert_rejected_without_writes('mismatched label budgets')

    def test_terminal_pair_must_match_raw_trajectory(self):
        path = self.results / 'terminal_pairs.csv'
        pairs = pd.read_csv(path)
        pairs.loc[0, 'tiv_minus_random'] += .1
        pairs.to_csv(path, index=False)
        self.save_manifest()
        self.assert_rejected_without_writes('terminal pair')

    def test_complete_run_exports_compatible_columns_brier_and_paired_table(self):
        result = self.run_extract()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = pd.read_csv(self.out / 'revelation_tiv_per_cost.csv')
        self.assertTrue({'width_median', 'cvar_median', 'brier_median'}.issubset(summary.columns))
        self.assertAlmostEqual(summary.iloc[-1].width_median, .1)
        self.assertAlmostEqual(summary.iloc[-1].brier_median, .1595)
        self.assertTrue((self.out / 'revelation_band_tiv_per_cost_brier.csv').exists())
        table = (self.out / 'sequential_terminal_table.tex').read_text()
        self.assertIn('Paired TIV', table)
        self.assertIn('+0.0500', table)  # Paired median differs from the difference of medians (-.1).
        self.assertIn('exact predictor reference', table)
        self.assertIn('0.2500', table)
        provenance = json.loads((self.out / 'revelation_provenance.json').read_text())
        self.assertIn('retained_plugin', provenance['sequential_aggregation'])
        self.assertFalse(any('part_b_adaptive' in path for path in provenance['sources']))


if __name__ == '__main__':
    unittest.main()
