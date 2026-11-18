"""Reproducible figures: numerical results only, with explicit model provenance."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .physics import GaussianWake, TEACHER_PARAMETERS

COLORS = {"aligned": "#8293a5", "physics": "#d88a37", "ai": "#008c87"}
LABELS = {"aligned": "Wind alignment", "physics": "Physics controller", "ai": "AI controller"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.labelcolor": "#283e50", "text.color": "#183246",
                         "axes.edgecolor": "#becbd4", "axes.spines.top": False,
                         "axes.spines.right": False, "grid.alpha": .22,
                         "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
                         "svg.fonttype": "none"})


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def wake_comparison(farm, wind, commands):
    """Common synthetic teacher visualizes actions; no neural full-field claim."""
    style()
    xy, d = np.asarray(farm.xy_m), farm.turbine.diameter_m
    x = np.linspace(xy[:, 0].min()-2*d, xy[:, 0].max()+7*d, 280)
    y = np.linspace(xy[:, 1].min()-4*d, xy[:, 1].max()+4*d, 120)
    xx, yy = np.meshgrid(x, y)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), layout='constrained', sharex=True, sharey=True)
    teacher = GaussianWake(farm, TEACHER_PARAMETERS)
    for ax, method in zip(axs, ('aligned', 'physics', 'ai')):
        yaw = np.array(commands[method]['yaw_deg'])
        u = teacher.field(wind, yaw, xx, yy)/max(wind.speed_ms, 1e-6)
        im = ax.pcolormesh(xx/d, yy/d, u, vmin=.35, vmax=1, cmap='viridis', shading='auto', rasterized=True)
        for i, ((tx, ty), gamma) in enumerate(zip(xy/d, yaw)):
            heading = np.deg2rad(wind.direction_deg-gamma)
            lateral = np.array([np.cos(heading), -np.sin(heading)])*.5
            ax.plot([tx-lateral[0], tx+lateral[0]], [ty-lateral[1], ty+lateral[1]], color='white', lw=3)
            ax.text(tx, ty+1.05, f'T{i+1}  {gamma:+.1f}°', ha='center', color='white', fontsize=9,
                    bbox=dict(facecolor='#173244', edgecolor='none', alpha=.85, pad=2))
        ax.set_title(f"{LABELS[method]}  |  {commands[method]['evaluated_power_kw']/1000:.3f} MW", loc='left')
        ax.set_ylabel('North / D')
        ax.set_aspect('equal')
    axs[-1].set_xlabel('East / rotor diameter D')
    fig.colorbar(im, ax=axs, label='Hub-height speed / ambient speed', fraction=.025, pad=.02)
    fig.suptitle('One wind condition · three sets of yaw commands', fontsize=17, fontweight='bold')
    fig.supxlabel('Synthetic teacher fields at controller setpoints; the neural model predicts rotor speed, not these maps.', fontsize=9)
    return fig


def prediction_figure(csv_path, metrics, out):
    style()
    table = np.genfromtxt(csv_path, delimiter=',', names=True)
    truth, prior, ai = (table[k] for k in ('truth_ms', 'prior_ms', 'ai_ms'))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    bounds = [min(truth.min(), prior.min(), ai.min())-.2, max(truth.max(), prior.max(), ai.max())+.2]
    for ax, pred, key, color, title in zip(axs[:2], (prior, ai), ('prior_speed_ms', 'neural_speed_ms'),
                                         (COLORS['physics'], COLORS['ai']), ('Physics prior', 'Neural residual mean')):
        ax.scatter(truth, pred, s=7, alpha=.23, color=color, rasterized=True)
        ax.plot(bounds, bounds, color='#283e50', lw=1, ls='--')
        ax.set(xlim=bounds, ylim=bounds, xlabel='Synthetic target speed (m/s)', ylabel='Predicted speed (m/s)')
        ax.set_aspect('equal')
        ax.set_title(title, loc='left')
        m = metrics[key]
        ax.text(.04, .96, f"RMSE  {m['rmse']:.3f} m/s\nr  {m['pearson_r']:.4f}", transform=ax.transAxes, va='top')
    ids = np.argsort(truth)[np.linspace(0, len(truth)-1, 64, dtype=int)]
    position = np.arange(len(ids))
    axs[2].fill_between(position, table['lower90_ms'][ids], table['upper90_ms'][ids], color=COLORS['ai'], alpha=.22, label='90% marginal interval')
    axs[2].plot(position, truth[ids], '.', color='#183246', ms=4, label='Target')
    axs[2].plot(position, ai[ids], color=COLORS['ai'], lw=1, label='Mean')
    coverage = metrics['marginal_90_interval']['coverage']
    axs[2].set_title(f'Diffusion intervals · {coverage:.1%} coverage', loc='left')
    axs[2].set(xlabel='64 displayed points, sorted by target speed', ylabel='Rotor-effective speed (m/s)')
    axs[2].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle(f"Unseen-layout prediction · {metrics['layout_count']} test layouts / {len(table):,} turbine observations", fontsize=16, fontweight='bold')
    fig.supxlabel('All reported metrics use the complete test split. Displayed points are correlated within layouts.', fontsize=9)
    save(fig, out)


def replay_figure(rows, out):
    style()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), layout='constrained', sharex=True)
    first = [r for r in rows if r['method'] == 'ai']
    time = np.array([r['time_s'] for r in first])/60
    axs[0].plot(time, [r['wind_from_deg'] for r in first], color='#243e53', lw=2)
    axs[0].set_ylabel('Wind FROM (°)')
    axs[0].set_title('Input wind direction', loc='left')
    for key in (k for k in first[0] if k.startswith('yaw_T')):
        axs[1].plot(time, [r[key] for r in first], marker='.', lw=1.5, label=key.replace('yaw_', '').replace('_deg', ''))
    axs[1].axhline(30, color='#c6cfd6', ls='--', lw=1)
    axs[1].axhline(-30, color='#c6cfd6', ls='--', lw=1)
    axs[1].set_ylabel('Yaw misalignment (°)')
    axs[1].set_title('AI controller commands', loc='left')
    axs[1].legend(frameon=False, ncol=3)
    for method in ('aligned', 'physics', 'ai'):
        current = [r for r in rows if r['method'] == method]
        axs[2].plot(time, [r['power_kw']/1000 for r in current], lw=2, color=COLORS[method], label=LABELS[method])
    axs[2].set(ylabel='Farm power (MW)', xlabel='Replay interval start (min)')
    axs[2].set_title('Power evaluated by the same synthetic teacher', loc='left')
    axs[2].legend(frameon=False, ncol=3)
    for ax in axs:
        ax.grid(axis='y')
    fig.suptitle('Direction changes → constrained yaw → power response', fontsize=17, fontweight='bold')
    fig.supxlabel('Quasi-steady interval replay: propagation delays and power during actuator travel are not resolved.', fontsize=9)
    save(fig, out)


def energy_figure(summary, out):
    style()
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    names = ('aligned', 'physics', 'ai')
    values = [summary[m]['energy_kwh'] for m in names]
    bars = ax.bar(range(3), values, width=.55, color=[COLORS[m] for m in names])
    for bar, method in zip(bars, names):
        item = summary[method]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"{item['energy_kwh']:.1f} kWh\n{item['gain_vs_alignment_pct']:+.2f}%", ha='center', va='bottom', fontsize=11)
    ax.set_xticks(range(3), [LABELS[m] for m in names])
    ax.set_ylim(0, max(values)*1.24)
    ax.set_ylabel('Quasi-steady energy (kWh)')
    ax.set_title('Identical wind trace · identical evaluation model', loc='left', fontsize=15)
    fig.supxlabel('Engineering demonstration; percentages describe this synthetic trace, not field energy yield.', fontsize=9)
    save(fig, out)
