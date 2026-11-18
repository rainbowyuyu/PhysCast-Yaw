import numpy as np
import pytest

from physcast_yaw import Farm, GaussianWake, Turbine, Wind, load_config


def test_single_turbine_power_and_yaw_loss():
    farm = Farm([[0, 0]])
    result = GaussianWake(farm).simulate(Wind(8, 270, .08), [[0], [20]])
    assert np.allclose(result["speed_ms"], 8)
    assert result["power_kw"][1, 0] / result["power_kw"][0, 0] == pytest.approx(
        np.cos(np.deg2rad(20))**1.88)


def test_wakes_and_batch_identity():
    model = GaussianWake(Farm([[0, 0], [756, 0], [1512, 0]]))
    yaw = np.array([[0, 0, 0], [25, 10, 0]])
    batch = model.simulate(Wind(8, 270, .08), yaw)
    assert 0 < batch["speed_ms"][0, 2] < batch["speed_ms"][0, 0]
    assert batch["speed_ms"][0, 1] % 1 != 0  # integer wind input must not truncate
    for i in range(2):
        assert np.allclose(batch["power_kw"][i], model.simulate(Wind(), yaw[i])["power_kw"][0])


def test_permutation_translation_rotation():
    xy = np.array([[0, 0], [700, 30], [1400, -50]])
    yaw = np.array([20, -5, 2])
    wind = Wind(8, 267, .08)
    original = GaussianWake(Farm(xy.tolist())).simulate(wind, yaw)["power_kw"][0]
    perm = np.array([2, 0, 1])
    actual = GaussianWake(Farm((xy[perm]+[300, -900]).tolist())).simulate(wind, yaw[perm])["power_kw"][0]
    np.testing.assert_allclose(actual, original[perm], rtol=1e-10)
    angle = np.deg2rad(35)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    actual = GaussianWake(Farm((xy @ rot.T).tolist())).simulate(Wind(8, 232, .08), yaw)["power_kw"][0]
    np.testing.assert_allclose(actual, original, rtol=1e-10)


def test_yaw_reflection_and_field_direction():
    model = GaussianWake(Farm([[0, 0], [756, 0]]))
    wind = Wind()
    np.testing.assert_allclose(model.simulate(wind, [25, 0])["power_kw"],
                               model.simulate(wind, [-25, 0])["power_kw"])
    one = GaussianWake(Farm([[0, 0]]))
    assert one.field(wind, [0], -100, 0) == pytest.approx(8)
    assert one.field(wind, [25], 600, 60) < one.field(wind, [25], 600, -60)


def test_invalid_inputs_and_power_domain():
    with pytest.raises(ValueError):
        Farm([[0, 0], [10, 0]])
    with pytest.raises(ValueError):
        Wind(8, 0, 8)
    with pytest.raises(ValueError):
        GaussianWake(Farm([[0, 0]])).simulate(Wind(), [float("nan")])
    assert np.all(Turbine().power([0, 2.9, 25, 40], 0) == 0)
    assert Turbine().power(12, 0) == 5000


def test_features_respect_permutation():
    xy = [[0, 0], [700, 40], [1400, -30]]
    yaw = [25, 10, -3]
    features, _ = GaussianWake(Farm(xy)).features(Wind(), yaw)
    perm = [2, 0, 1]
    other, _ = GaussianWake(Farm([xy[i] for i in perm])).features(Wind(), [yaw[i] for i in perm])
    np.testing.assert_allclose(other, features[:, perm], atol=1e-6)


def test_config_unknown_keys(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text('{"farm":{"xy_m":[[0,0]]},"wimd":{}}')
    with pytest.raises(ValueError, match="Unknown"):
        load_config(f)
