"""Physics-guided residual ensemble and conditional residual DDPM."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Turbine, write_json
from .data import load_dataset
from .physics import GaussianWake


class ResidualMean(nn.Module):
    def __init__(self, inputs=16, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x, prior):
        # No wake => no learned mean correction. Bounded effective-speed closure.
        return torch.clamp(prior + .8*(1-prior)*torch.tanh(self.net(x).squeeze(-1)), 0, 1)


class Denoiser(nn.Module):
    def __init__(self, inputs=16, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs+4, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x, z, t):
        temporal = torch.stack((t, torch.sin(t*math.pi), torch.cos(t*math.pi)), dim=-1)
        return self.net(torch.cat((x, z[..., None], temporal), dim=-1)).squeeze(-1)


def schedule(steps=32, device="cpu"):
    t = torch.arange(steps+1, dtype=torch.float32, device=device)/steps
    alpha_bar = torch.cos((t+.008)/1.008*math.pi/2)**2
    beta = (1-alpha_bar[1:]/alpha_bar[:-1]).clamp(.0001, .999)
    return beta, torch.cumprod(1-beta, dim=0)


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth).ravel(), np.asarray(prediction).ravel()
    difference = prediction-truth
    r = float(np.corrcoef(truth, prediction)[0, 1]) if min(truth.std(), prediction.std()) > 1e-12 else None
    return dict(rmse=float(np.sqrt(np.mean(difference**2))), mae=float(np.abs(difference).mean()),
                bias=float(difference.mean()), pearson_r=r, n=int(len(truth)))


class WakePredictor:
    """A checkpoint holds tensors/primitives only and loads with weights_only=True."""
    def __init__(self, checkpoint, device="cpu"):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
        if checkpoint.get("schema") != "physcast-yaw-model-v1":
            raise ValueError("Unknown model schema")
        self.state = checkpoint
        self.device = device
        self.mean_models = []
        for state in checkpoint["mean_states"]:
            net = ResidualMean(checkpoint["inputs"]).to(device)
            net.load_state_dict(state)
            self.mean_models.append(net.eval())
        self.denoiser = Denoiser(checkpoint["inputs"]).to(device)
        self.denoiser.load_state_dict(checkpoint["denoiser_state"])
        self.denoiser.eval()

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def normalized(self, features):
        return (self._tensor(features)-self._tensor(self.state["xmean"])) / self._tensor(self.state["xscale"])

    @torch.inference_mode()
    def ratios(self, features, prior):
        x, p = self.normalized(features), self._tensor(prior)
        ensemble = torch.stack([m(x, p) for m in self.mean_models])
        return ensemble.mean(0).cpu().numpy(), ensemble.std(0, unbiased=False).cpu().numpy()

    def in_domain(self, features, farm):
        return bool(np.all(self.domain_mask(features, farm)))

    def domain_mask(self, features, farm):
        """One support flag per candidate farm scene, independent of batch peers."""
        features = np.asarray(features)
        if features.ndim != 3 or features.shape[1:] != (farm.n, self.state["inputs"]):
            raise ValueError("Domain checks expect (scenes, turbines, features)")
        if asdict(farm.turbine) != self.state["turbine"] or farm.n > self.state["max_turbines"]:
            return np.zeros(len(features), dtype=bool)
        lo, hi = np.array(self.state["xmin"]), np.array(self.state["xmax"])
        slack = .15*np.maximum(hi-lo, .01)
        outside = (features < lo-slack) | (features > hi+slack)
        return ~np.any(outside, axis=(1, 2))

    def predict(self, farm, wind, yaw_deg, fallback=True):
        base = GaussianWake(farm)
        features, prior = base.features(wind, yaw_deg)
        mask = self.domain_mask(features, farm)
        mean, std = self.ratios(features, prior)
        speed = mean*wind.speed_ms
        high = farm.turbine.power(np.clip(mean+std, 0, 1)*wind.speed_ms, yaw_deg)
        low = farm.turbine.power(np.clip(mean-std, 0, 1)*wind.speed_ms, yaw_deg)
        power = farm.turbine.power(speed, yaw_deg)
        std_power = (high-low)/2
        sources = np.full(len(mask), "neural_residual", dtype=object)
        if fallback and not np.all(mask):
            physics = base.simulate(wind, yaw_deg)
            speed[~mask], power[~mask] = physics["speed_ms"][~mask], physics["power_kw"][~mask]
            std_power[~mask] = 0
            sources[~mask] = "physics_fallback"
        return dict(speed_ms=speed, power_kw=power, power_std_kw=std_power,
                    source=str(sources[0]) if len(set(sources)) == 1 else "mixed",
                    source_by_scene=sources.tolist(), in_domain=bool(np.all(mask)),
                    in_domain_by_scene=mask.tolist())

    @torch.inference_mode()
    def samples(self, features, prior, draws=64, seed=42, chunk=512):
        """Marginal samples, shape (draws, points). Not joint turbine realizations."""
        features = np.asarray(features).reshape(-1, self.state["inputs"])
        prior = np.asarray(prior).ravel()
        if draws < 2 or len(prior) != len(features):
            raise ValueError("Need >=2 draws and matching points")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        beta, ab = schedule(self.state["diffusion_steps"], self.device)
        mean, _ = self.ratios(features, prior)
        results = []
        for start in range(0, len(features), chunk):
            count = min(chunk, len(features)-start)
            x = self.normalized(features[start:start+count]).repeat(draws, 1)
            z = torch.randn(draws*count, generator=generator, device=self.device)
            for step in reversed(range(len(beta))):
                t = torch.full_like(z, step/(len(beta)-1))
                noise = self.denoiser(x, z, t)
                previous_ab = ab[step-1] if step else torch.ones_like(ab[0])
                # Predict clipped x0, then use the exact DDPM posterior mean.
                x0 = ((z-torch.sqrt(1-ab[step])*noise)/torch.sqrt(ab[step])).clamp(-6, 6)
                coef1 = beta[step]*torch.sqrt(previous_ab)/(1-ab[step])
                coef2 = (1-previous_ab)*torch.sqrt(1-beta[step])/(1-ab[step])
                z = coef1*x0 + coef2*z
                if step:
                    variance = beta[step]*(1-previous_ab)/(1-ab[step])
                    z += torch.sqrt(variance)*torch.randn(z.shape, generator=generator, device=self.device)
            result = mean[None, start:start+count] + z.cpu().numpy().reshape(draws, count)*self.state["residual_scale"]
            results.append(np.clip(result, 0, 1))
        return np.concatenate(results, axis=1)

    def intervals(self, features, prior, draws=64, seed=42):
        samples = self.samples(features, prior, draws, seed)
        low, high = np.quantile(samples, [.05, .95], axis=0)
        correction = self.state.get("interval_correction", 0.0)
        return np.clip(low-correction, 0, 1), np.clip(high+correction, 0, 1)


def train(data_path, out, epochs=100, diffusion_updates=1200, members=3, seed=42, device="cpu"):
    if epochs < 1 or diffusion_updates < 1 or members < 1:
        raise ValueError("Positive training budgets required")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    arrays, meta = load_dataset(data_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = arrays["train"], arrays["validation"]
    xraw, prior, target = arrays["features"], arrays["prior_ratio"], arrays["target_ratio"]
    xmean, xscale = xraw[tr].mean(0), np.maximum(xraw[tr].std(0), .02)
    x = torch.as_tensor((xraw-xmean)/xscale, dtype=torch.float32, device=device)
    p = torch.as_tensor(prior, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    trt, vat = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    histories, states = [], []
    for member in range(members):
        torch.manual_seed(seed+member*101)
        model = ResidualMean(x.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
        best, bad, history = float("inf"), 0, []
        for epoch in range(epochs):
            model.train()
            order = trt[torch.randperm(len(trt), device=device)]
            for ids in order.split(1024):
                loss = torch.mean((model(x[ids], p[ids])-y[ids])**2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                score = float(torch.mean((model(x[vat], p[vat])-y[vat])**2))
            history.append(dict(epoch=epoch+1, validation_mse=score))
            if score < best:
                best, bad, best_state = score, 0, deepcopy(model.state_dict())
            else:
                bad += 1
            if bad >= 20:
                break
        states.append({k: v.cpu() for k, v in best_state.items()})
        histories.append(history)
        print(f"Mean {member+1}/{members}: selected validation RMSE={best**.5:.5f} (speed ratio)", flush=True)
    torch.manual_seed(seed+999)
    denoiser = Denoiser(x.shape[-1]).to(device)
    state = dict(schema="physcast-yaw-model-v1", inputs=x.shape[-1], mean_states=states,
                 denoiser_state=denoiser.state_dict(), xmean=xmean.tolist(), xscale=xscale.tolist(),
                 xmin=xraw[tr].min(0).tolist(), xmax=xraw[tr].max(0).tolist(),
                 diffusion_steps=32, residual_scale=1., interval_correction=0.,
                 turbine=meta["turbine"], max_turbines=max(len(meta["farms"][int(g)]["xy_m"]) for g in np.unique(arrays["group"][tr])),
                 seed=seed, data_source=meta["source"])
    predictor = WakePredictor(state, device)
    mean, _ = predictor.ratios(xraw, prior)
    scale = max(float(np.std(target[tr]-mean[tr])), .003)
    residual = torch.as_tensor((target-mean)/scale, dtype=torch.float32, device=device)
    beta, ab = schedule(32, device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=.001, weight_decay=1e-4)
    best, best_state, history = float("inf"), deepcopy(denoiser.state_dict()), []
    vg = torch.Generator(device=device).manual_seed(seed+872)
    vt = torch.randint(32, (len(va),), generator=vg, device=device)
    vn = torch.randn((len(va),), generator=vg, device=device)
    vz = torch.sqrt(ab[vt])*residual[vat] + torch.sqrt(1-ab[vt])*vn
    for step in range(diffusion_updates):
        ids = trt[torch.randint(len(tr), (min(1024, len(tr)),), device=device)]
        t = torch.randint(32, (len(ids),), device=device)
        noise = torch.randn(len(ids), device=device)
        z = torch.sqrt(ab[t])*residual[ids] + torch.sqrt(1-ab[t])*noise
        loss = torch.mean((denoiser(x[ids], z, t/31)-noise)**2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(denoiser.parameters(), 1)
        optimizer.step()
        if (step+1) % 100 == 0 or step == diffusion_updates-1:
            with torch.inference_mode():
                score = float(torch.mean((denoiser(x[vat], vz, vt/31)-vn)**2))
            history.append(dict(update=step+1, validation_noise_mse=score))
            if score < best:
                best, best_state = score, deepcopy(denoiser.state_dict())
    state.update(denoiser_state={k: v.cpu() for k,v in best_state.items()}, residual_scale=scale)
    predictor = WakePredictor(state, device)
    cal = arrays["calibration"]
    low, high = predictor.intervals(xraw[cal], prior[cal], draws=64, seed=seed+123)
    scores = np.maximum.reduce([low-target[cal], target[cal]-high, np.zeros(len(cal))])
    rank = min(len(scores)-1, int(np.ceil((len(scores)+1)*.90))-1)
    state["interval_correction"] = float(np.sort(scores)[rank])
    state["training"] = dict(epochs=epochs, diffusion_updates=diffusion_updates, members=members)
    torch.save(state, out/"model.pt")
    write_json(out/"training.json", dict(mean_history=histories, diffusion_history=history,
                                        interval_correction=state["interval_correction"], data_source=meta["source"]))
    print(f"Model saved to {out/'model.pt'}", flush=True)
    return WakePredictor(state, device)


def evaluate(data_path, checkpoint, out, draws=64):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    arrays, meta = load_dataset(data_path)
    model = WakePredictor(checkpoint)
    ids = arrays["test"]
    x, p = arrays["features"][ids], arrays["prior_ratio"][ids]
    truth, wind = arrays["target_ratio"][ids], arrays["speed_ms"][ids]
    mean, _ = model.ratios(x, p)
    low, high = model.intervals(x, p, draws=draws, seed=773)
    turbine = Turbine(**meta["turbine"])
    yaw = arrays["yaw_deg"][ids]
    result = dict(source=meta["source"], split="held_out_layouts", layout_count=int(len(np.unique(arrays["group"][ids]))),
                  prior_speed_ms=metrics(truth*wind, p*wind), neural_speed_ms=metrics(truth*wind, mean*wind),
                  prior_power_kw=metrics(turbine.power(truth*wind, yaw), turbine.power(p*wind, yaw)),
                  neural_power_kw=metrics(turbine.power(truth*wind, yaw), turbine.power(mean*wind, yaw)),
                  marginal_90_interval=dict(coverage=float(np.mean((truth>=low)&(truth<=high))),
                                            mean_width_ms=float(np.mean((high-low)*wind)), draws=draws,
                                            note="Marginal turbine coverage on correlated layout observations; not joint trajectory coverage"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"metrics.json", result)
    table = np.c_[arrays["group"][ids], wind, truth*wind, p*wind, mean*wind, low*wind, high*wind]
    np.savetxt(out/"predictions.csv", table, delimiter=",", header="layout_id,ambient_ms,truth_ms,prior_ms,ai_ms,lower90_ms,upper90_ms", comments="")
    return result
