"""Recompute committed benchmark numbers without a model or training run."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from physcast_yaw.learning import metrics


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'docs' / 'benchmark'
    expected = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    table = np.genfromtxt(folder / 'predictions.csv', delimiter=',', names=True)
    for label, column in [('prior_speed_ms', 'prior_ms'), ('neural_speed_ms', 'ai_ms')]:
        actual = metrics(table['truth_ms'], table[column])
        for key, value in actual.items():
            np.testing.assert_allclose(value, expected[label][key], rtol=1e-10, atol=1e-12)
    coverage = np.mean((table['truth_ms'] >= table['lower90_ms']) &
                       (table['truth_ms'] <= table['upper90_ms']))
    width = np.mean(table['upper90_ms'] - table['lower90_ms'])
    np.testing.assert_allclose(coverage, expected['marginal_90_interval']['coverage'])
    np.testing.assert_allclose(width, expected['marginal_90_interval']['mean_width_ms'])
    summary = json.loads((folder / 'control_summary.json').read_text(encoding='utf-8'))
    with (folder / 'replay.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for method, values in summary.items():
        selected = [row for row in rows if row['method'] == method]
        energy = sum(float(row['energy_kwh']) for row in selected)
        movement = sum(float(row['movement_deg']) for row in selected)
        gain = 100 * (energy / summary['aligned']['energy_kwh'] - 1)
        np.testing.assert_allclose([energy, movement, gain], [values['energy_kwh'],
                                   values['total_heading_travel_deg'], values['gain_vs_alignment_pct']])
        assert sum(row['source'] == 'physics_fallback' for row in selected) == values['fallback_steps']
        assert sum(row['status'] == 'infeasible' for row in selected) == values['infeasible_steps']
    hashes = json.loads((folder / 'checksums.json').read_text(encoding='utf-8'))
    for relative, expected_hash in hashes.items():
        actual_hash = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f'Artifact hash mismatch: {relative}'
    reduction = 100 * (1 - expected['neural_speed_ms']['rmse'] / expected['prior_speed_ms']['rmse'])
    print(f'Verified {len(table)} prediction rows, {len(rows)} replay rows, and {len(hashes)} artifact hashes.')
    print(f'Speed RMSE reduction: {reduction:.2f}%; interval coverage: {coverage:.2%}.')
    print(f"AI replay gain: {summary['ai']['gain_vs_alignment_pct']:.2f}% over rate-limited alignment.")
    print('Synthetic benchmark only. This checks saved evidence, not independent scientific validity.')


if __name__ == '__main__':
    main()
