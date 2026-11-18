"""Steady Gaussian engineering wakes with rotor quadrature and RSS superposition.

The deflection closure is deliberately simple and documented, not a literal
implementation of the full Bastankhah or FLORIS model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import Farm, Wind


@dataclass(frozen=True)
class WakeParameters:
    expansion_base: float = .038
    expansion_ti: float = .38
    initial_width: float = .27
    deflection_gain: float = .18

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Wake parameters must be positive and finite")


TEACHER_PARAMETERS = WakeParameters(expansion_base=.023, expansion_ti=.30, deflection_gain=.30)


class GaussianWake:
    def __init__(self, farm: Farm, parameters: WakeParameters | None = None):
        self.farm = farm
        self.parameters = parameters or WakeParameters()

    def coordinates(self, wind: Wind):
        a = np.deg2rad(wind.direction_deg)
        flow = np.array([-np.sin(a), -np.cos(a)])
        left = np.array([-flow[1], flow[0]])
        xy = np.asarray(self.farm.xy_m)
        return xy @ flow, xy @ left

    def _deficit(self, dx, dy, dz, source_speed, yaw, ti):
        d = self.farm.turbine.diameter_m
        p = self.parameters
        xd = np.maximum(dx / d, 0)
        sigma = d * (p.initial_width + (p.expansion_base + p.expansion_ti * ti) * xd)
        gamma = np.deg2rad(yaw)
        center = p.deflection_gain * np.sin(gamma) * np.maximum(dx, 0) / (1 + .03 * xd)
        ct = self.farm.turbine.thrust_coefficient(source_speed) * np.cos(gamma)**2
        amplitude = 1 - np.sqrt(np.clip(1 - ct / (8 * (sigma / d)**2), .02, 1))
        deficit = source_speed * amplitude * np.exp(-.5 * (((dy - center) / sigma)**2 + (dz / sigma)**2))
        return np.where(dx > .1 * d, deficit, 0.0)

    def _yaws(self, yaw_deg):
        y = np.atleast_2d(np.asarray(yaw_deg, dtype=float))
        if y.shape[1] != self.farm.n or not np.isfinite(y).all() or (np.abs(y) > 60).any():
            raise ValueError("yaw_deg must have shape (N,) or (batch,N), finite and within +/-60 deg")
        return y

    def simulate(self, wind: Wind, yaw_deg):
        yaw = self._yaws(yaw_deg)
        x, y = self.coordinates(wind)
        order = np.argsort(x, kind="stable")
        speed = np.full(yaw.shape, wind.speed_ms, dtype=float)
        angles = np.arange(8) * np.pi / 4
        # Equal-area-inspired compact quadrature: centre 0.2, ring points 0.1 each.
        ry = np.r_[0, .375 * self.farm.turbine.diameter_m * np.cos(angles)]
        rz = np.r_[0, .375 * self.farm.turbine.diameter_m * np.sin(angles)]
        weights = np.r_[.2, np.full(8, .1)]
        for pos, i in enumerate(order):
            sum_sq = np.zeros((len(yaw), len(weights)))
            for j in order[:pos]:
                deficit = self._deficit(x[i] - x[j], y[i] + ry[None] - y[j], rz[None],
                                        speed[:, j, None], yaw[:, j, None], wind.turbulence_intensity)
                sum_sq += deficit**2
            disk = np.clip(wind.speed_ms - np.sqrt(sum_sq), 0, wind.speed_ms)
            speed[:, i] = np.cbrt(np.sum(disk**3 * weights, axis=1))
        return {"speed_ms": speed, "power_kw": self.farm.turbine.power(speed, yaw)}

    def field(self, wind: Wind, yaw_deg, x_m, y_m, height_m=None):
        yaw = self._yaws(yaw_deg)
        if len(yaw) != 1:
            raise ValueError("Field rendering accepts one yaw vector")
        xg, yg = np.broadcast_arrays(np.asarray(x_m, float), np.asarray(y_m, float))
        a = np.deg2rad(wind.direction_deg)
        xp = -np.sin(a) * xg - np.cos(a) * yg
        yp = np.cos(a) * xg - np.sin(a) * yg
        x, y = self.coordinates(wind)
        speed = self.simulate(wind, yaw)["speed_ms"][0]
        dz = 0.0 if height_m is None else np.asarray(height_m) - self.farm.turbine.hub_height_m
        sum_sq = np.zeros_like(xg)
        for i in range(self.farm.n):
            sum_sq += self._deficit(xp-x[i], yp-y[i], dz, speed[i], yaw[0, i],
                                    wind.turbulence_intensity)**2
        return np.clip(wind.speed_ms - np.sqrt(sum_sq), 0, wind.speed_ms)

    def features(self, wind: Wind, yaw_deg):
        """Shared per-turbine inputs; no absolute location or turbine-ID feature."""
        yaw = self._yaws(yaw_deg)
        ratio = self.simulate(wind, yaw)["speed_ms"] / max(wind.speed_ms, 1e-6)
        x, y = self.coordinates(wind)
        dx = (x[None, :] - x[:, None]) / self.farm.turbine.diameter_m  # source, target
        dy = (y[None, :] - y[:, None]) / self.farm.turbine.diameter_m
        upstream = dx > .1
        relevance = np.where(upstream, np.exp(-.5*(dy/1.5)**2) / (1 + np.maximum(dx, 0)/6), 0)
        g = np.deg2rad(yaw)
        context = [np.ones_like(yaw)*wind.speed_ms/12, np.ones_like(yaw)*wind.turbulence_intensity/.15,
                   np.sin(g), np.cos(g), ratio, 1-ratio]
        for source in (np.ones_like(yaw), np.sin(g), np.cos(g), np.sin(g)**2):
            context.append(source @ relevance)
        for edge in (dx/12, dy/3, dy**2/9, dx**2/144):
            context.append(np.broadcast_to(np.sum(relevance*edge, axis=0), yaw.shape))
        context.extend([np.sin(g) @ (relevance*dy), np.cos(g) @ (relevance*dx/12)])
        return np.stack(context, axis=-1).astype(np.float32), ratio.astype(np.float32)
