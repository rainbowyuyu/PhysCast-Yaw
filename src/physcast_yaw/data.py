"""Layout-separated synthetic datasets and an explicit observation importer."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .config import Farm, Turbine, Wind
from .physics import GaussianWake, TEACHER_PARAMETERS

SPLITS = ("train", "validation", "calibration", "test")


def split_layouts(groups, seed=42):
    unique = np.unique(groups)
    if len(unique) < 12:
        raise ValueError("At least 12 distinct layouts are needed for four disjoint splits")
    shuffled = np.random.default_rng(seed).permutation(unique)
    cuts = [int(len(unique)*v) for v in (.60, .75, .85)]
    parts = np.split(shuffled, cuts)
    return {name: np.flatnonzero(np.isin(groups, ids)) for name, ids in zip(SPLITS, parts)}


def generate_dataset(path, layouts=64, samples_per_layout=64, seed=42):
    if layouts < 12 or samples_per_layout < 8:
        raise ValueError("Use at least 12 layouts and 8 samples per layout")
    rng = np.random.default_rng(seed)
    blocks = {k: [] for k in ("features", "prior_ratio", "target_ratio", "speed_ms", "yaw_deg", "group")}
    farms = []
    for group in range(layouts):
        n = int(rng.integers(2, 9))
        spacing = rng.uniform(4.5, 8.5) * 126
        xy = np.c_[np.arange(n)*spacing, rng.uniform(-.65, .65, n)*126]
        # Multiple rows, arbitrary rotation, plus translated locations.
        if group % 3 == 0:
            xy[:, 0] = (np.arange(n) % 3) * spacing
            xy[:, 1] += (np.arange(n) // 3) * rng.uniform(4.5, 7)*126
        rotation = rng.uniform(-np.pi, np.pi)
        rot = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        farm = Farm((xy @ rot.T).tolist())
        farms.append(asdict(farm))
        base, teacher = GaussianWake(farm), GaussianWake(farm, TEACHER_PARAMETERS)
        remaining = samples_per_layout
        while remaining:
            count = min(8, remaining)
            wind = Wind(float(rng.uniform(5.5, 12)), float(270-np.rad2deg(rotation)+rng.uniform(-25,25)),
                        float(rng.uniform(.05, .16)))
            yaw = rng.uniform(-30, 30, (count, n))
            yaw[0] = 0
            features, prior = base.features(wind, yaw)
            truth = teacher.simulate(wind, yaw)["speed_ms"] / wind.speed_ms
            # Stated synthetic measurement noise; not turbulence-resolved dynamics.
            noise = rng.normal(0, .003, (count, 1)) + rng.normal(0, .004, truth.shape)
            target = np.clip(truth + noise, 0, 1)
            values = dict(features=features.reshape(-1, features.shape[-1]), prior_ratio=prior.ravel(),
                          target_ratio=target.ravel(), speed_ms=np.full(count*n, wind.speed_ms),
                          yaw_deg=yaw.ravel(), group=np.full(count*n, group, dtype=int))
            for key, value in values.items():
                blocks[key].append(value)
            remaining -= count
    arrays = {k: np.concatenate(v) for k, v in blocks.items()}
    arrays.update(split_layouts(arrays["group"], seed))
    meta = dict(schema="physcast-yaw-data-v1", source="synthetic_gaussian_teacher", seed=seed,
                layouts=layouts, samples_per_layout=samples_per_layout, farms=farms,
                turbine=asdict(Turbine()), teacher=asdict(TEACHER_PARAMETERS),
                noise="speed-ratio common SD=.003 plus independent SD=.004; clipped to [0,1]",
                split_unit="layout", sample_unit="turbine within a farm scenario")
    save_dataset(path, arrays, meta)
    return arrays, meta


def save_dataset(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, ensure_ascii=False))


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        arrays = {k: data[k].copy() for k in data.files if k != "metadata"}
    if metadata.get("schema") != "physcast-yaw-data-v1":
        raise ValueError("Unsupported dataset schema")
    n = len(arrays["target_ratio"])
    if arrays["features"].shape != (n, 16):
        raise ValueError("Features must have shape (observations, 16)")
    for key in ("features", "prior_ratio", "target_ratio", "speed_ms", "yaw_deg", "group"):
        if len(arrays[key]) != n or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid aligned finite data: {key}")
    used, group_sets = set(), []
    for name in SPLITS:
        ids = arrays[name]
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) == 0:
            raise ValueError(f"Invalid split: {name}")
        if (ids < 0).any() or (ids >= n).any() or len(np.unique(ids)) != len(ids) or used.intersection(ids):
            raise ValueError("Split indices must be unique, disjoint and in bounds")
        used.update(ids.tolist())
        groups = set(arrays["group"][ids].tolist())
        if any(groups.intersection(other) for other in group_sets):
            raise ValueError("Layout leakage across dataset splits")
        group_sets.append(groups)
    if len(used) != n:
        raise ValueError("Every observation must belong to exactly one split")
    if (arrays["speed_ms"] <= 0).any() or ((arrays["target_ratio"] < 0) | (arrays["target_ratio"] > 1)).any():
        raise ValueError("Targets must be effective-speed ratios in [0,1]; ambient speed must be positive")
    return arrays, metadata


def import_observations(csv_path, out, source, seed=42, turbine=None):
    """CSV one row per turbine per scene; scenes of one layout stay in one split.

    Required: layout_id, scene_id, turbine_id, x_m, y_m, wind_speed_ms,
    wind_from_deg, ti, yaw_deg, effective_speed_ms. IDs need not be numeric.
    """
    if not source.strip():
        raise ValueError("Provide a data provenance/source description")
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    scenes = {}
    for row in rows:
        scenes.setdefault((row["layout_id"], row["scene_id"]), []).append(row)
    groups = {key: i for i, key in enumerate(sorted({key[0] for key in scenes}))}
    blocks = {k: [] for k in ("features", "prior_ratio", "target_ratio", "speed_ms", "yaw_deg", "group")}
    layouts = {}
    turbine = turbine or Turbine()
    for (layout_id, _), scene in sorted(scenes.items()):
        scene.sort(key=lambda row: row["turbine_id"])
        if len({r["turbine_id"] for r in scene}) != len(scene):
            raise ValueError("Duplicate turbine in a scene")
        xy = [[float(r[k]) for k in ("x_m", "y_m")] for r in scene]
        fingerprint = [(r["turbine_id"], *p) for r, p in zip(scene, xy)]
        if layout_id in layouts and layouts[layout_id] != fingerprint:
            raise ValueError("One layout_id must keep turbine IDs and coordinates fixed")
        layouts[layout_id] = fingerprint
        farm = Farm(xy, turbine)
        for field in ("wind_speed_ms", "wind_from_deg", "ti"):
            if len({float(r[field]) for r in scene}) != 1:
                raise ValueError(f"Inconsistent scene field: {field}")
        wind = Wind(float(scene[0]["wind_speed_ms"]), float(scene[0]["wind_from_deg"]), float(scene[0]["ti"]))
        if wind.speed_ms <= 0:
            raise ValueError("Training requires positive ambient wind speed")
        yaw = np.array([float(r["yaw_deg"]) for r in scene])
        if not np.isfinite(yaw).all() or np.max(np.abs(yaw)) > 30:
            raise ValueError("Training importer supports +/-30 degree yaw")
        features, prior = GaussianWake(farm).features(wind, yaw)
        target = np.array([float(r["effective_speed_ms"]) for r in scene])/wind.speed_ms
        values = dict(features=features[0], prior_ratio=prior[0], target_ratio=target,
                      speed_ms=np.full(farm.n, wind.speed_ms), yaw_deg=yaw,
                      group=np.full(farm.n, groups[layout_id]))
        for key, value in values.items():
            blocks[key].append(value)
    if not blocks["features"]:
        raise ValueError("No observations found")
    arrays = {k: np.concatenate(v) for k, v in blocks.items()}
    arrays.update(split_layouts(arrays["group"], seed))
    save_dataset(out, arrays, dict(schema="physcast-yaw-data-v1", source=source, seed=seed,
                                   split_unit="layout", turbine=asdict(turbine),
                                   farms=[dict(xy_m=[[p[1], p[2]] for p in layouts[k]]) for k in sorted(layouts)]))
    load_dataset(out)  # enforce the same physical and split contract as generated data
