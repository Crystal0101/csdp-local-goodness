"""Catch silent evidence loss and changes to the published statistical summary."""
from pathlib import Path
import csv
import copy
import json
import shutil
import tempfile
import unittest

import numpy as np

from src.config import ROOT, load_config
from src.results import load_saved_results, export_training_csv
from src.stats import analyse_results, paired_summary


class EvidenceChecks(unittest.TestCase):
    def test_paired_interval_against_two_point_reference(self):
        result = paired_summary(np.array([-1., 1.]))
        self.assertAlmostEqual(result.mean_pp, 0.)
        self.assertAlmostEqual(result.upper_pp, 12.706204736, places=6)
        self.assertAlmostEqual(result.lower_pp, -12.706204736, places=6)

    def test_original_and_supplement_match_reported_intervals(self):
        results = analyse_results(load_saved_results())
        expected = {
            'primary': (-.153203, -.765369, .458962),
            'G8-G1': (-.640669, -1.682539, .401202),
            'epoch20': (-.250696, -.695768, .194376),
            'budget_interaction': (.389972, -.314218, 1.094163),
        }
        for name, reference in expected.items():
            result = results[name]
            np.testing.assert_allclose(
                [result.mean_pp, result.lower_pp, result.upper_pp], reference, atol=1e-5,
            )
        self.assertEqual(results['primary'].losses_over_one_pp, 4)

    def test_reject_missing_duplicate_or_inconsistent_epoch(self):
        with (ROOT / 'results/epochs.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        bad_accuracy = [dict(row) for row in rows]
        bad_accuracy[0]['accuracy'] = '0.123456'
        for name, altered in [('missing', rows[1:]), ('duplicate', rows + [rows[0]]),
                              ('mismatch', bad_accuracy)]:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary)
                for filename in ('labels.json', 'pairing.json'):
                    shutil.copy2(ROOT / 'results' / filename, target / filename)
                with (target / 'epochs.csv').open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(altered)
                with self.assertRaises(ValueError):
                    load_saved_results(target)

    def test_reject_truncated_or_misordered_fresh_curves(self):
        valid = {
            'group': 1, 'seed': 12001, 'labels': [1],
            'curves': [{'epoch': epoch, 'validation_accuracy': 1.} for epoch in range(1, 6)],
            'epoch_results': [
                {'epoch': epoch, 'predictions': [1], 'probabilities': [[.1, .9]]}
                for epoch in range(1, 6)
            ],
            'final_validation_accuracy': 1.,
        }
        truncated = copy.deepcopy(valid)
        truncated['curves'].pop()
        misordered = copy.deepcopy(valid)
        misordered['epoch_results'][2]['epoch'] = 2
        wrong_final = copy.deepcopy(valid)
        wrong_final['final_validation_accuracy'] = 0.
        for altered in (truncated, misordered, wrong_final):
            with self.subTest(record=altered), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = root / 'runs/G1_12001'
                run.mkdir(parents=True)
                (run / 'result.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    export_training_csv(root, 'primary')

    def test_g1_is_valid_and_unregistered_group_is_rejected(self):
        primary = load_config('primary')
        primary.validate_group(1)
        with self.assertRaises(ValueError):
            primary.validate_group(2)


if __name__ == '__main__':
    unittest.main()
