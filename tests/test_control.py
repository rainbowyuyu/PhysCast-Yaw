import numpy as np
import pytest

from physcast_yaw import Farm, Wind, GaussianWake
from physcast_yaw.config import wrap180
from physcast_yaw.control import ControllerSettings, optimize_yaw, replay


def settings(**kwargs):
    return ControllerSettings(candidates=8, sweeps=1, **kwargs)


def test_absolute_heading_rate_and_wrap():
    farm = Farm([[0, 0], [756, 0]])
    previous = np.array([359., 354.])
    s = settings(interval_s=10, direction_std_deg=0)
    result = optimize_yaw(farm, Wind(8, 1), current_heading_deg=previous, settings=s)
    assert result['status'] == 'ok'
    movement = np.abs(wrap180(np.array(result['heading_deg'])-previous))
    assert np.all(movement <= 3+1e-8)
    np.testing.assert_allclose(wrap180(1-np.array(result['heading_deg'])), result['yaw_deg'])


def test_feasible_incumbent_not_degraded_and_deterministic():
    farm, wind = Farm([[0, 0], [756, 0], [1512, 0]]), Wind()
    s = settings(direction_std_deg=0, risk_weight=0, movement_weight=0, load_weight=0)
    first = optimize_yaw(farm, wind, settings=s)
    second = optimize_yaw(farm, wind, settings=s)
    assert first == second
    assert first['predicted_power_kw'] >= GaussianWake(farm).simulate(wind, [0, 0, 0])['power_kw'].sum()-1e-7
    assert np.max(np.abs(first['yaw_deg'])) <= 30
    assert first['max_heading_rate_deg_s'] <= .3+1e-8


def test_infeasible_holds_heading_and_parked_does_not_optimize():
    farm = Farm([[0, 0]])
    result = optimize_yaw(farm, Wind(8, 90), current_heading_deg=[0], settings=settings(interval_s=1))
    assert result['status'] == 'infeasible' and result['heading_deg'] == [0]
    assert result['predicted_power_kw'] is None
    parked = optimize_yaw(farm, Wind(26), settings=settings())
    assert parked['status'] == 'parked' and parked['predicted_power_kw'] == 0


def test_out_of_domain_falls_back_whole_solve():
    class Unsupported:
        def domain_mask(self, x, *_):
            return np.zeros(len(x), dtype=bool)
        def predict(self, *_args, **_kwargs):
            raise AssertionError('Should not evaluate unsupported neural inputs')
    farm, wind, s = Farm([[0, 0], [756, 0]]), Wind(), settings()
    fallback = optimize_yaw(farm, wind, Unsupported(), settings=s)
    baseline = optimize_yaw(farm, wind, settings=s)
    assert fallback['source'] == 'physics_fallback'
    assert fallback['yaw_deg'] == baseline['yaw_deg']


def test_direction_scenarios_hold_the_same_headings():
    class Recorder:
        def __init__(self):
            self.calls = []
        def domain_mask(self, x, *_):
            return np.ones(len(x), dtype=bool)
        def predict(self, farm, wind, yaw, **_):
            self.calls.append((wind.direction_deg, yaw.copy()))
            result = GaussianWake(farm).simulate(wind, yaw)
            result['power_std_kw'] = np.zeros_like(result['power_kw'])
            return result
    model = Recorder()
    result = optimize_yaw(Farm([[0, 0]]), Wind(), model, settings=settings())
    assert result['source'] == 'neural_residual'
    for i in range(0, len(model.calls), 3):
        direction, yaw = model.calls[i]
        for d, y in model.calls[i+1:i+3]:
            np.testing.assert_allclose(direction-yaw, d-y, atol=1e-8)


def test_rejecting_one_candidate_does_not_discard_supported_actions():
    class NarrowDomain:
        def domain_mask(self, features, *_):
            return np.all(np.abs(features[:, :, 2]) <= .2, axis=1)

        def predict(self, farm, wind, yaw, **_):
            assert np.all(np.abs(np.sin(np.deg2rad(yaw))) <= .2 + 1e-7)
            result = GaussianWake(farm).simulate(wind, yaw)
            result['power_std_kw'] = np.zeros_like(result['power_kw'])
            return result

    result = optimize_yaw(Farm([[0, 0], [756, 0]]), Wind(), NarrowDomain(), settings=settings())
    assert result['source'] == 'neural_residual'
    assert result['rejected_ai_candidates'] > 0
    assert result['status'] == 'ok'


def test_replay_independent_states_and_energy_units(tmp_path):
    farm = Farm([[0, 0], [756, 0]])
    rows = replay(farm, [Wind(), Wind(8, 272)], None, tmp_path/'replay.csv', settings=settings())
    assert len(rows) == 6
    for r in rows:
        assert r['energy_kwh'] == pytest.approx(r['power_kw']/30)
        assert r['movement_deg'] <= farm.n*.3*120+1e-7
    for i in (0, 3):
        assert rows[i+1]['power_kw'] == pytest.approx(rows[i+2]['power_kw'])
