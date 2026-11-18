"""One-command experiment with a frozen synthetic-data protocol."""
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np

from .config import Farm, Wind, write_json
from .control import ControllerSettings, optimize_yaw, replay
from .data import generate_dataset
from .learning import WakePredictor, evaluate, train
from .physics import GaussianWake, TEACHER_PARAMETERS
from .plots import energy_figure, prediction_figure, replay_figure, save, wake_comparison


def finish_demo(out, quick=False):
    """Evaluate already trained demo artifacts, and write plots / exact source tables."""
    out = Path(out)
    result = evaluate(out/'dataset.npz', out/'model/model.pt', out/'evaluation', draws=16 if quick else 64)
    model = WakePredictor(out/'model/model.pt')
    farm, wind = Farm([[0, 0], [756, 0], [1512, 0]]), Wind()
    settings = ControllerSettings(candidates=16 if quick else 96, sweeps=1 if quick else 3)
    teacher = GaussianWake(farm, TEACHER_PARAMETERS)
    commands = {'aligned': dict(yaw_deg=[0., 0., 0.], source='alignment')}
    commands['physics'] = optimize_yaw(farm, wind, settings=settings)
    commands['ai'] = optimize_yaw(farm, wind, model, settings=settings)
    for command in commands.values():
        command['evaluated_power_kw'] = float(teacher.simulate(wind, command['yaw_deg'])['power_kw'].sum())
    write_json(out/'static_comparison.json', commands)
    count = 4 if quick else 24
    time = np.linspace(0, 2*np.pi, count, endpoint=False)
    winds = [Wind(float(8+.5*np.sin(t/2)), float(270+10*np.sin(t)), .08) for t in time]
    np.savetxt(out/'wind.csv', np.array([[w.speed_ms, w.direction_deg, w.turbulence_intensity] for w in winds]),
               delimiter=',', header='wind_speed_ms,wind_from_deg,ti', comments='')
    print(f'Replaying {count} wind states with three independent controllers...', flush=True)
    rows = replay(farm, winds, model, out/'replay.csv', settings)
    energy = {m: sum(r['energy_kwh'] for r in rows if r['method'] == m) for m in ('aligned', 'physics', 'ai')}
    summary = {m: dict(energy_kwh=v, gain_vs_alignment_pct=100*(v/energy['aligned']-1),
                       total_heading_travel_deg=sum(r['movement_deg'] for r in rows if r['method']==m),
                       fallback_steps=sum(r['source']=='physics_fallback' for r in rows if r['method']==m),
                       infeasible_steps=sum(r['status']=='infeasible' for r in rows if r['method']==m)) for m,v in energy.items()}
    write_json(out/'control_summary.json', summary)
    figures = out/'figures'
    save(wake_comparison(farm, wind, commands), figures/'wake-control')
    prediction_figure(out/'evaluation/predictions.csv', result, figures/'prediction')
    replay_figure(rows, figures/'replay')
    energy_figure(summary, figures/'energy')
    write_json(out/'manifest.json', dict(schema='physcast-yaw-demo-v1', seed=42, quick=quick,
                                        source='synthetic_gaussian_teacher', farm=asdict(farm),
                                        controller=asdict(settings), wind_steps=count,
                                        python=platform.python_version(),
                                        dependencies={k: version(k) for k in ('numpy', 'torch', 'matplotlib')},
                                        claim_scope='Synthetic engineering demonstration; no field measurements or transient CFD.'))
    print(f"Test speed RMSE: {result['prior_speed_ms']['rmse']:.4f} -> {result['neural_speed_ms']['rmse']:.4f} m/s", flush=True)
    print(f"AI energy vs alignment: {summary['ai']['gain_vs_alignment_pct']:+.2f}%", flush=True)
    return result, summary


def run_demo(out, quick=False, device='cpu'):
    out = Path(out)
    generate_dataset(out/'dataset.npz', layouts=16 if quick else 64, samples_per_layout=16 if quick else 64, seed=42)
    train(out/'dataset.npz', out/'model', epochs=10 if quick else 100,
          diffusion_updates=100 if quick else 1200, members=2 if quick else 3, seed=42, device=device)
    return finish_demo(out, quick)
