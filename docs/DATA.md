# Data contract and external observations

[首页](../README.md) · [English](../README.en.md) · [Algorithm](ALGORITHM.md)

The included data generator is a complete synthetic training path. External observations use the same feature and target contract, while retaining an explicit source label.

## Synthetic data

`physcast-yaw generate --out outputs/data.npz` defaults to seed 42, 64 layouts, and 64 farm states per layout. Each layout has 2–8 homogeneous turbines, streamwise spacing 4.5–8.5 rotor diameters, transverse jitter up to 0.65D, and a random rotation. Every third layout uses multiple rows with 4.5–7D row spacing. Turbine clearance is checked.

For each inflow condition, eight yaw states are sampled, beginning with zero yaw. Inflow speed is uniform over 5.5–12 m/s, TI over 0.05–0.16, wind direction within 25° of the layout's row direction, and yaw within ±30°.

Targets are the synthetic teacher's rotor-effective speed ratio plus a scene-common Gaussian term with SD 0.003 and a per-turbine independent term with SD 0.004, then clipped to [0,1]. Thus uncertainty is observation noise, not resolved turbulence. The common term introduces cross-turbine dependence that the marginal DDPM does not explicitly model. Clipping changes the target distribution near free-stream speed.

## Layout-disjoint splits

Layout IDs are shuffled with a fixed seed and allocated approximately 60/15/10/15% to training, validation, calibration, and test sets. All states and turbines of one layout remain in the same split. A minimum of 12 layouts keeps all four groups populated.

| Split | Purpose | Default run layouts | Default run observations |
|---|---|---:|---:|
| Train | Fit networks and feature normalization | 38 | 10,944 |
| Validation | Select mean and diffusion checkpoints | 10 | 3,776 |
| Calibration | Widen marginal intervals | 6 | 1,920 |
| Test | Report final prediction metrics | 10 | 3,840 |

The loader rejects overlapping observations or layouts, invalid indices, uncovered observations, nonfinite inputs, nonpositive ambient speed, and target ratios outside [0,1]. Losses and published metrics pool turbine observations; they are not equally weighted averages over layouts.

## Import CSV observations

One row describes one turbine in one complete scene. Required header:

```csv
layout_id,scene_id,turbine_id,x_m,y_m,wind_speed_ms,wind_from_deg,ti,yaw_deg,effective_speed_ms
```

| Field | Meaning / requirement |
|---|---|
| `layout_id` | Stable geometry ID; never assign different IDs to the same layout merely to populate the split |
| `scene_id` | A condition within the layout, unique together with `layout_id` |
| `turbine_id` | Stable unit ID; each turbine appears once per scene |
| `x_m`, `y_m` | East/north coordinates in metres |
| `wind_speed_ms` | Positive ambient speed, common across the scene |
| `wind_from_deg` | Meteorological wind FROM direction, common across the scene |
| `ti` | Turbulence intensity as a fraction, e.g. `0.08`, common across the scene |
| `yaw_deg` | Nominal misalignment within ±30°, with heading = wind_from − yaw |
| `effective_speed_ms` | Rotor-effective speed label, between zero and ambient speed |

Turbine IDs and coordinates must remain fixed for a layout. A missing turbine changes the scene fingerprint and is rejected. Coordinates must satisfy the spacing contract. The supplied turbine specification is homogeneous across all rows.

```bash
physcast-yaw import-csv --csv my-observations.csv --source "Dataset DOI or simulator/version; processing protocol" --config configs/three_turbines.json --out outputs/observations.npz
physcast-yaw train --data outputs/observations.npz --out outputs/observed-model
physcast-yaw evaluate --data outputs/observations.npz --model outputs/observed-model/model.pt --out outputs/observed-evaluation
```

For import, `--config` supplies only the turbine specification; geometry and inflow come from CSV. Without it, the generic 5 MW turbine is assumed. Use a new output directory for each dataset/model pair. Evaluation uses the dataset's test indices: keep the intended training dataset and checkpoint paired, since the current loader does not enforce a checkpoint–dataset checksum match.

### Preparing labels

Use rotor-equivalent inflow from a documented simulation or a defensible observation process. A single point anemometer is not automatically a rotor average. Power alone is insufficient to identify effective inflow uniquely under curtailment, rated operation, or uncertain yaw losses; the importer does not silently invert power into wind-speed labels.

Keep acquisition timestamps, filtering, unit conversions, averaging conventions, control mode, provenance, and licensing with the source data. The minimal CSV contract does not store timestamps, so experiments requiring lag, transient propagation, or leakage control by campaign require an extended schema and splitter. Repeated campaigns on the same geometry should retain one layout ID.

Fields are defined for flat, wake-deficit cases. Ratios above ambient are rejected rather than silently truncated; terrain speed-up requires a model/schema extension. A changed turbine model requires retraining. The default generation command always uses the bundled generic turbine.

## NPZ representation

Files are read with `allow_pickle=False`.

| Array | Shape | Content |
|---|---|---|
| `features` | `(observations, 16)` | Raw network features |
| `prior_ratio` | `(observations,)` | Prior rotor speed / ambient speed |
| `target_ratio` | `(observations,)` | Label / ambient speed |
| `speed_ms` | `(observations,)` | Ambient wind speed |
| `yaw_deg` | `(observations,)` | Turbine yaw misalignment |
| `group` | `(observations,)` | Integer layout group |
| `train`, `validation`, `calibration`, `test` | 1-D index arrays | Disjoint split membership |
| `metadata` | scalar JSON string | Schema, source, seed, turbine specification, farm geometries |

Checkpoints contain tensors and primitive metadata and are loaded with `torch.load(..., weights_only=True)`. Run code and load artifacts from sources you trust. Data files and checkpoints are ignored by Git; small, explicitly public result tables are under [`benchmark/`](benchmark/).

## Wind trace for replay

```csv
wind_speed_ms,wind_from_deg,ti
8.0,270.0,0.08
8.2,272.0,0.08
8.1,274.0,0.08
```

Each row represents one controller interval, configured by `interval_s` (120 s by default). It is a sequence of steady inflows. All three controllers see the same trace and start aligned to its first direction. The replay stops if any evaluated yaw exceeds the plant model's ±60° range.

## Public example provenance

The committed [benchmark](benchmark/) was generated locally by this repository. It contains no private wind-farm measurements, third-party data, credentials, or trained weights from another project. Its metadata records the data/model hashes and the exact split counts for the reported run.
