"""Public data contracts. Coordinates are metres; powers are kW."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Turbine:
    """Generic engineering curve, not an OEM or NREL reference turbine."""

    diameter_m: float = 126.0
    hub_height_m: float = 90.0
    rated_power_kw: float = 5000.0
    cut_in_ms: float = 3.0
    rated_speed_ms: float = 11.4
    cut_out_ms: float = 25.0
    ct_below_rated: float = 0.8
    yaw_power_exponent: float = 1.88

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Turbine parameters must be finite and positive")
        if not self.cut_in_ms < self.rated_speed_ms < self.cut_out_ms:
            raise ValueError("Require cut_in < rated_speed < cut_out")
        if not 0 < self.ct_below_rated < 1:
            raise ValueError("ct_below_rated must be in (0, 1)")
        if self.hub_height_m <= self.diameter_m / 2:
            raise ValueError("The rotor must clear the ground")

    def power(self, speed_ms, yaw_deg):
        u = np.asarray(speed_ms, dtype=float)
        gamma = np.deg2rad(np.asarray(yaw_deg, dtype=float))
        fraction = (u**3 - self.cut_in_ms**3) / (self.rated_speed_ms**3 - self.cut_in_ms**3)
        p = self.rated_power_kw * np.clip(fraction, 0, 1)
        p *= np.clip(np.cos(gamma), 0, 1) ** self.yaw_power_exponent
        return np.where((u >= self.cut_in_ms) & (u < self.cut_out_ms), p, 0.0)

    def thrust_coefficient(self, speed_ms):
        u = np.asarray(speed_ms, dtype=float)
        ct = self.ct_below_rated * np.minimum(1.0, (self.rated_speed_ms / np.maximum(u, .1))**2)
        return np.where((u >= self.cut_in_ms) & (u < self.cut_out_ms), ct, 0.0)


@dataclass(frozen=True)
class Farm:
    xy_m: list[list[float]]
    turbine: Turbine = field(default_factory=Turbine)

    def __post_init__(self):
        xy = np.asarray(self.xy_m, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not 1 <= len(xy) <= 24 or not np.isfinite(xy).all():
            raise ValueError("xy_m must be finite (N, 2), with 1 <= N <= 24")
        if len(xy) > 1:
            distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
            distances[np.diag_indices(len(xy))] = np.inf
            if distances.min() < 3 * self.turbine.diameter_m:
                raise ValueError("Minimum turbine spacing is 3 rotor diameters")

    @property
    def n(self):
        return len(self.xy_m)


@dataclass(frozen=True)
class Wind:
    speed_ms: float = 8.0
    direction_deg: float = 270.0
    turbulence_intensity: float = .08

    def __post_init__(self):
        values = (self.speed_ms, self.direction_deg, self.turbulence_intensity)
        if not np.isfinite(values).all() or not 0 <= self.speed_ms <= 60:
            raise ValueError("Wind must be finite; speed range is 0..60 m/s")
        if not 0 <= self.turbulence_intensity <= .5:
            raise ValueError("TI is a fraction in [0, .5]")


def wrap180(angle):
    return (np.asarray(angle, dtype=float) + 180) % 360 - 180


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(data) - {"farm", "wind", "controller"}:
        raise ValueError("Unknown top-level config fields")
    f = data["farm"].copy()
    farm = Farm(turbine=Turbine(**f.pop("turbine", {})), **f)
    wind = Wind(**data.get("wind", {}))
    return farm, wind, data.get("controller", {})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
