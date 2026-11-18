"""Command-line entry point; training and plotting imports are lazy."""
import argparse
import csv
import json

import numpy as np

from .config import Wind, load_config, write_json
from .physics import GaussianWake


def main():
    parser = argparse.ArgumentParser(description="PhysCast-Yaw: wake prediction and yaw optimization")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("simulate", help="Steady engineering wake simulation")
    p.add_argument("--config", required=True)
    p.add_argument("--yaw", nargs="+", type=float)
    p.add_argument("--out", required=True)

    p = sub.add_parser("generate", help="Generate layout-separated synthetic training data")
    p.add_argument("--out", required=True)
    p.add_argument("--layouts", type=int, default=64)
    p.add_argument("--samples-per-layout", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("import-csv", help="Import rotor-effective speed observations")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--config", help="Turbine specification (layout and wind are read from CSV)")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", help="Train the mean ensemble and conditional residual diffusion")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--diffusion-updates", type=int, default=1200)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")

    p = sub.add_parser("evaluate", help="Evaluate the untouched layout test split")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--draws", type=int, default=64)

    for name in ("optimize", "replay"):
        p = sub.add_parser(name, help="Optimize a setpoint" if name == "optimize" else "Replay a quasi-steady wind trace")
        p.add_argument("--config", required=True)
        p.add_argument("--model", help="Omit for the physics-only baseline")
        p.add_argument("--out", required=True)
        if name == "optimize":
            p.add_argument("--heading", type=float, nargs="+", help="Current absolute nacelle headings")
        else:
            p.add_argument("--wind-csv", required=True, help="wind_speed_ms,wind_from_deg,ti; one row per control interval")

    p = sub.add_parser("demo", help="Generate, train, evaluate, control and plot a complete reproducible example")
    p.add_argument("--out", default="outputs/demo")
    p.add_argument("--quick", action="store_true", help="Small smoke run; not the README benchmark")
    p.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.command == "simulate":
        farm, wind, _ = load_config(args.config)
        result = GaussianWake(farm).simulate(wind, args.yaw if args.yaw is not None else np.zeros(farm.n))
        output = {k: v[0].tolist() for k, v in result.items()}
        output.update(source="engineering_prior", farm_power_kw=float(result["power_kw"].sum()))
        write_json(args.out, output)
    elif args.command == "generate":
        from .data import generate_dataset
        generate_dataset(args.out, args.layouts, args.samples_per_layout, args.seed)
    elif args.command == "import-csv":
        from .data import import_observations
        turbine = load_config(args.config)[0].turbine if args.config else None
        import_observations(args.csv, args.out, args.source, args.seed, turbine)
    elif args.command == "train":
        from .learning import train
        train(args.data, args.out, args.epochs, args.diffusion_updates, args.members, args.seed, args.device)
    elif args.command == "evaluate":
        from .learning import evaluate
        print(json.dumps(evaluate(args.data, args.model, args.out, args.draws), indent=2))
    elif args.command in ("optimize", "replay"):
        from .control import ControllerSettings, optimize_yaw, replay
        from .learning import WakePredictor
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        farm, wind, config = load_config(args.config)
        settings = ControllerSettings(**config)
        model = WakePredictor(args.model) if args.model else None
        if args.command == "optimize":
            result = optimize_yaw(farm, wind, model, args.heading, settings)
            write_json(args.out, result)
            print(json.dumps(result, indent=2))
        else:
            with open(args.wind_csv, newline="", encoding="utf-8-sig") as stream:
                winds = [Wind(float(r['wind_speed_ms']), float(r['wind_from_deg']), float(r['ti'])) for r in csv.DictReader(stream)]
            replay(farm, winds, model, args.out, settings)
    elif args.command == "demo":
        from .demo import run_demo
        run_demo(args.out, quick=args.quick, device=args.device)
    print(f"Completed {args.command}: {args.out}")


if __name__ == "__main__":
    main()
