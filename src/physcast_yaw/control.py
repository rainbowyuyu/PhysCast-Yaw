"""Receding setpoint optimization with explicit nacelle-heading constraints."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from numbers import Integral

import numpy as np

from .config import Wind, wrap180
from .physics import GaussianWake, TEACHER_PARAMETERS


@dataclass(frozen=True)
class ControllerSettings:
    yaw_limit_deg: float = 30.0
    yaw_rate_deg_s: float = .3
    interval_s: float = 120.0
    direction_std_deg: float = 2.0
    risk_weight: float = .1
    movement_weight: float = .002
    load_weight: float = .002
    candidates: int = 96
    sweeps: int = 3
    seed: int = 42

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Finite controller settings required")
        if not 0 < self.yaw_limit_deg <= 30 or not 0 < self.yaw_rate_deg_s <= 2 or self.interval_s <= 0:
            raise ValueError("Require 0<yaw_limit<=30, 0<rate<=2 and positive interval")
        if not 0 <= self.direction_std_deg <= 10:
            raise ValueError("Direction standard deviation range is 0..10 degrees")
        if min(self.risk_weight, self.movement_weight, self.load_weight) < 0:
            raise ValueError("Penalty weights cannot be negative")
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (self.candidates, self.sweeps, self.seed)):
            raise ValueError("candidates, sweeps and seed must be integers")
        if self.candidates < 8 or self.sweeps < 1 or self.seed < 0:
            raise ValueError("Use >=8 integer candidates and >=1 integer sweeps")


class _OutOfDomain(Exception):
    pass


def optimize_yaw(farm, wind, predictor=None, current_heading_deg=None, settings=None):
    """Return feasible steady setpoints; the teacher is NEVER used for selection.

    All uncertainty scenarios see the SAME nacelle headings. Direction changes
    therefore change actual yaw misalignment; they do not rotate the nacelles.
    """
    settings = settings or ControllerSettings()
    heading = np.full(farm.n, wind.direction_deg) if current_heading_deg is None else np.asarray(current_heading_deg, float)
    if heading.shape != (farm.n,) or not np.isfinite(heading).all():
        raise ValueError("current_heading_deg must be a finite vector of N absolute headings")
    current_yaw = wrap180(wind.direction_deg-heading)
    travel = settings.yaw_rate_deg_s*settings.interval_s
    lo = np.maximum(-settings.yaw_limit_deg, current_yaw-travel)
    hi = np.minimum(settings.yaw_limit_deg, current_yaw+travel)
    if np.any(lo > hi):
        return dict(status="infeasible", reason="angle_and_rate_limits_do_not_intersect", yaw_deg=current_yaw.tolist(),
                    heading_deg=(heading % 360).tolist(), predicted_power_kw=None, objective=None,
                    source="hold_heading", movement_deg=[0.0]*farm.n)
    if wind.speed_ms < farm.turbine.cut_in_ms or wind.speed_ms >= farm.turbine.cut_out_ms:
        return dict(status="parked", reason="outside_generating_wind_range", yaw_deg=current_yaw.tolist(),
                    heading_deg=(heading % 360).tolist(), predicted_power_kw=0., objective=None,
                    source="hold_heading", movement_deg=[0.0]*farm.n)
    base = GaussianWake(farm)
    offsets = np.array([-np.sqrt(3), 0, np.sqrt(3)])*settings.direction_std_deg
    weights = np.array([1/6, 2/3, 1/6])
    normalization = max(float(farm.turbine.power(wind.speed_ms, 0))*farm.n, 1.)
    evaluation_count = 0
    rejected_count = 0

    def objective(candidates, use_ai):
        nonlocal evaluation_count, rejected_count
        yaw = np.atleast_2d(candidates)
        powers, uncertainties = [], []
        valid = np.ones(len(yaw), dtype=bool)
        scenarios = [Wind(wind.speed_ms, wind.direction_deg+o, wind.turbulence_intensity) for o in offsets]
        if use_ai:
            for scenario, offset in zip(scenarios, offsets):
                x, _ = base.features(scenario, yaw+offset)
                valid &= predictor.domain_mask(x, farm)
            rejected_count += int((~valid).sum())
        if not valid.any():
            return np.full(len(yaw), -np.inf), np.full(len(yaw), np.nan)
        for scenario, offset in zip(scenarios, offsets):
            actual_yaw = yaw[valid]+offset
            if use_ai:
                result = predictor.predict(farm, scenario, actual_yaw, fallback=False)
                uncertainties.append(result["power_std_kw"].sum(axis=1))
            else:
                result = base.simulate(scenario, actual_yaw)
                uncertainties.append(np.zeros(valid.sum()))
            powers.append(result["power_kw"].sum(axis=1))
        expected = weights @ np.array(powers)
        disagreement = weights @ np.array(uncertainties)
        movement = np.mean(((yaw[valid]-current_yaw)/settings.yaw_limit_deg)**2, axis=1)
        # Dimensionless side-force proxy, not a fatigue metric or damage-equivalent load.
        side_force_proxy = np.mean(np.sin(np.deg2rad(yaw[valid]))**2, axis=1)
        score = (expected-settings.risk_weight*disagreement)/normalization
        score -= settings.movement_weight*movement + settings.load_weight*side_force_proxy
        evaluation_count += int(valid.sum())*len(offsets)
        all_scores, all_powers = np.full(len(yaw), -np.inf), np.full(len(yaw), np.nan)
        all_scores[valid], all_powers[valid] = score, expected
        return all_scores, all_powers

    def search(use_ai):
        rng = np.random.default_rng(settings.seed)
        initial = np.vstack([np.clip(current_yaw, lo, hi), np.clip(np.zeros(farm.n), lo, hi), lo, hi,
                             rng.uniform(lo, hi, (settings.candidates, farm.n))])
        # Include common wake-steering patterns, but keep zero/hold as incumbents.
        ramps = np.linspace(1, 0, farm.n)
        initial = np.vstack([initial, np.clip(ramps*25, lo, hi), np.clip(-ramps*25, lo, hi)])
        values, _ = objective(initial, use_ai)
        supported = np.flatnonzero(np.isfinite(values))
        if not len(supported):
            raise _OutOfDomain
        seeds = initial[supported[np.argsort(values[supported])[-3:]]]
        winners = []
        for start in seeds:
            best = start.copy()
            for _ in range(settings.sweeps):
                for j in range(farm.n):
                    trials = np.tile(best, (14, 1))
                    trials[:13, j] = np.linspace(lo[j], hi[j], 13)
                    scores, _ = objective(trials, use_ai)
                    best = trials[int(np.argmax(scores))]
            winners.append(best)
        # Compare every refined solution against the original feasible incumbents.
        final = np.vstack([winners, initial[:2]])
        scores, powers = objective(final, use_ai)
        k = int(np.argmax(scores))
        return final[k], float(scores[k]), float(powers[k])

    source, reason = "physics_prior", "requested_physics_baseline"
    if predictor is not None:
        try:
            yaw, value, power = search(True)
            source, reason = "neural_residual", "unsupported_candidates_rejected"
        except _OutOfDomain:
            yaw, value, power = search(False)
            source, reason = "physics_fallback", "no_supported_feasible_seed"
    else:
        yaw, value, power = search(False)
    result_heading = (wind.direction_deg-yaw) % 360
    movement = np.abs(wrap180(result_heading-heading))
    return dict(status="ok", reason=reason, yaw_deg=yaw.tolist(), heading_deg=result_heading.tolist(),
                movement_deg=movement.tolist(), predicted_power_kw=power, objective=value,
                source=source, scenario_evaluations=evaluation_count,
                rejected_ai_candidates=rejected_count,
                max_heading_rate_deg_s=float(movement.max()/settings.interval_s))


def replay(farm, winds, predictor, out, settings=None):
    """Quasi-steady replay. Three independent actuator states, one fixed teacher."""
    settings = settings or ControllerSettings()
    methods = ("aligned", "physics", "ai")
    winds = list(winds)
    if not winds:
        raise ValueError("At least one wind state is required")
    heading = {k: np.full(farm.n, winds[0].direction_deg) for k in methods}
    plant = GaussianWake(farm, TEACHER_PARAMETERS)
    rows = []
    for step, wind in enumerate(winds):
        for method in methods:
            previous = heading[method].copy()
            if method == "aligned":
                delta = np.clip(wrap180(wind.direction_deg-previous),
                                -settings.yaw_rate_deg_s*settings.interval_s,
                                settings.yaw_rate_deg_s*settings.interval_s)
                heading[method] = (previous+delta) % 360
                yaw = wrap180(wind.direction_deg-heading[method])
                result = dict(status="ok", source="rate_limited_alignment")
            else:
                result = optimize_yaw(farm, wind, predictor if method == "ai" else None,
                                      previous, settings)
                yaw = np.array(result["yaw_deg"])
                heading[method] = np.array(result["heading_deg"])
            if np.any(np.abs(yaw) > 60):
                raise ValueError("Replay exceeds the +/-60 degree plant-model domain; use smaller wind increments")
            power = plant.simulate(wind, yaw)["power_kw"][0]
            rows.append(dict(time_s=step*settings.interval_s, method=method, wind_speed_ms=wind.speed_ms,
                             wind_from_deg=wind.direction_deg, power_kw=float(power.sum()),
                             energy_kwh=float(power.sum()*settings.interval_s/3600),
                             movement_deg=float(np.abs(wrap180(heading[method]-previous)).sum()),
                             status=result["status"], source=result["source"],
                             **{f"yaw_T{i+1}_deg": float(v) for i,v in enumerate(yaw)}))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
