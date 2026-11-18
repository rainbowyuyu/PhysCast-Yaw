import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from physcast_yaw import Farm, GaussianWake, Wind
from physcast_yaw.control import ControllerSettings
from physcast_yaw.data import import_observations, load_dataset


def test_import_csv_preserves_complete_scenes_and_splits(tmp_path):
    path = tmp_path/'observations.csv'
    fields = 'layout_id scene_id turbine_id x_m y_m wind_speed_ms wind_from_deg ti yaw_deg effective_speed_ms'.split()
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in range(12):
            farm = Farm([[0, 0], [630+group*15, group*10]])
            wind = Wind()
            speeds = GaussianWake(farm).simulate(wind, [0, 0])['speed_ms'][0]
            for i in range(2):
                writer.writerow(dict(zip(fields, [f'L{group}', 'S0', f'T{i}', *farm.xy_m[i], 8, 270, .08, 0, speeds[i]])))
    import_observations(path, tmp_path/'data.npz', 'unit-test synthetic observations')
    arrays, meta = load_dataset(tmp_path/'data.npz')
    assert len(meta['farms']) == 12 and len(arrays['group']) == 24
    np.testing.assert_allclose(arrays['target_ratio'], arrays['prior_ratio'], atol=1e-7)


def test_cli_simulation_produces_valid_json(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(farm=dict(xy_m=[[0, 0], [756, 0]]))), encoding='utf-8')
    output = tmp_path/'result.json'
    result = subprocess.run([sys.executable, '-m', 'physcast_yaw.cli', 'simulate', '--config', str(config), '--out', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding='utf-8'))
    assert data['farm_power_kw'] > 0 and data['speed_ms'][1] < data['speed_ms'][0]


@pytest.mark.parametrize('kwargs', [{'seed': -1}, {'candidates': 8.0}, {'sweeps': 0}, {'risk_weight': -.1}])
def test_invalid_controller_settings(kwargs):
    with pytest.raises(ValueError):
        ControllerSettings(**kwargs)
