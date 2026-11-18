import numpy as np
import pytest
import torch

from physcast_yaw.data import generate_dataset, load_dataset
from physcast_yaw.learning import ResidualMean, WakePredictor, train, evaluate


def test_split_is_by_layout_and_reproducible(tmp_path):
    a, _ = generate_dataset(tmp_path/"a.npz", layouts=12, samples_per_layout=8, seed=1)
    b, _ = generate_dataset(tmp_path/"b.npz", layouts=12, samples_per_layout=8, seed=1)
    np.testing.assert_array_equal(a["target_ratio"], b["target_ratio"])
    names = ("train", "validation", "calibration", "test")
    for i, name in enumerate(names):
        for other in names[i+1:]:
            assert not set(a["group"][a[name]]) & set(a["group"][a[other]])
    load_dataset(tmp_path/"a.npz")


def test_mean_physical_bounds_and_no_wake():
    torch.manual_seed(1)
    model = ResidualMean()
    result = model(torch.randn(32, 16), torch.linspace(0, 1, 32))
    assert torch.all((result >= 0) & (result <= 1))
    assert result[-1] == 1


def test_training_checkpoint_and_sampling(tmp_path):
    data = tmp_path/"data.npz"
    arrays, _ = generate_dataset(data, layouts=12, samples_per_layout=8)
    model = train(data, tmp_path/"model", epochs=3, diffusion_updates=4, members=2)
    loaded = WakePredictor(tmp_path/"model/model.pt")
    x, p = arrays["features"][:8], arrays["prior_ratio"][:8]
    np.testing.assert_allclose(model.ratios(x, p)[0], loaded.ratios(x, p)[0])
    first, second = loaded.samples(x, p, draws=4, seed=12), loaded.samples(x, p, draws=4, seed=12)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 8) and np.isfinite(first).all()
    assert np.all((first >= 0) & (first <= 1))
    result = evaluate(data, tmp_path/"model/model.pt", tmp_path/"eval", draws=4)
    assert 0 <= result["marginal_90_interval"]["coverage"] <= 1
    # Normalization is computed from training observations only.
    np.testing.assert_allclose(loaded.state['xmean'], arrays['features'][arrays['train']].mean(0))


def test_dataset_leakage_rejected(tmp_path):
    from physcast_yaw.data import save_dataset
    arrays, meta = generate_dataset(tmp_path/"a.npz", layouts=12, samples_per_layout=8)
    arrays["group"][arrays["test"][0]] = arrays["group"][arrays["train"][0]]
    save_dataset(tmp_path/"bad.npz", arrays, meta)
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(tmp_path/"bad.npz")


def test_prediction_fallback_is_per_scene_not_batch_wide():
    from dataclasses import asdict
    from physcast_yaw import Farm, Wind, GaussianWake
    from physcast_yaw.learning import Denoiser
    farm, wind = Farm([[0, 0], [756, 0]]), Wind()
    yaw = np.array([[0., 0.], [60., 60.]])
    features, _ = GaussianWake(farm).features(wind, yaw)
    state = dict(schema='physcast-yaw-model-v1', inputs=16,
                 mean_states=[ResidualMean().state_dict()], denoiser_state=Denoiser().state_dict(),
                 xmean=[0.]*16, xscale=[1.]*16, xmin=features[0].min(0).tolist(),
                 xmax=features[0].max(0).tolist(), turbine=asdict(farm.turbine), max_turbines=2)
    model = WakePredictor(state)
    together = model.predict(farm, wind, yaw)
    alone = model.predict(farm, wind, yaw[0])
    assert together['in_domain_by_scene'] == [True, False]
    assert together['source_by_scene'] == ['neural_residual', 'physics_fallback']
    np.testing.assert_allclose(together['speed_ms'][0], alone['speed_ms'][0])
    physics = GaussianWake(farm).simulate(wind, yaw[1])
    np.testing.assert_allclose(together['power_kw'][1], physics['power_kw'][0])
