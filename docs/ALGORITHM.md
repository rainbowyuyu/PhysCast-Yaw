# Algorithm and equations

[中文首页](../README.md) · [English](../README.en.md) · [Data](DATA.md) · [Validation](VALIDATION.md)

This document describes the implemented algorithm. Symbols are mapped to the source files so the equations can be checked against the executable model.

## 1. Inputs, coordinates, and output

A scene contains turbine positions, a homogeneous turbine specification, ambient wind speed, wind direction, turbulence intensity, and current nacelle headings. The controller returns nominal yaw misalignments, absolute headings, expected power, movement, model source, and feasibility status.

| Symbol | Meaning | Unit / convention |
|---|---|---|
| $\mathbf{x}_i=(x_i,y_i)$ | Turbine position | metres; east, north |
| $D,h,P_r$ | Rotor diameter, hub height, rated power | m, m, kW |
| $U_\infty,I$ | Ambient speed and turbulence intensity | m/s, fraction |
| $\alpha$ | Meteorological direction **from** which wind arrives | degrees clockwise from north |
| $\psi_i$ | Absolute upwind nacelle heading | same angular convention as $\alpha$ |
| $\gamma_i$ | Yaw misalignment | $\operatorname{wrap}_{180}(\alpha-\psi_i)$ |
| $U_i,P_i$ | Rotor-effective speed and power | m/s, kW |

Positive yaw rotates the upwind nacelle axis counterclockwise relative to the incoming wind, hence $\psi_i=\alpha-\gamma_i$. The streamwise and cross-stream basis vectors are

$$
\mathbf{e}_x=(-\sin\alpha,-\cos\alpha),\qquad
\mathbf{e}_y=(-e_{x,y},e_{x,x}).
$$

Angles passed to trigonometric functions are converted to radians. At $\alpha=270^\circ$, flow travels east. The code supports 1–24 turbines with at least $3D$ spacing. See [`config.py`](../src/physcast_yaw/config.py) and `GaussianWake.coordinates` in [`physics.py`](../src/physcast_yaw/physics.py).

## 2. Engineering wake prior

For a downstream point relative to turbine $j$, define $s=\max(\Delta x/D,0)$. The Gaussian width and lateral centre are

$$
\sigma_j=D\left[0.27+(k_0+k_I I)s\right],\qquad
\delta_j=\frac{k_\delta\sin\gamma_j\max(\Delta x,0)}{1+0.03s}.
$$

The speed deficit is

$$
\Delta U_j=U_j A_j
\exp\left[-\frac{1}{2}\left(\frac{(\Delta y-\delta_j)^2}{\sigma_j^2}
+\frac{\Delta z^2}{\sigma_j^2}\right)\right],
$$

$$
A_j=1-\sqrt{\operatorname{clip}\left(
1-\frac{C_T(U_j)\cos^2\gamma_j}{8(\sigma_j/D)^2},0.02,1\right)}.
$$

Deficits are zero for $\Delta x\leq0.1D$. Turbines are processed from upstream to downstream, so $U_j$ includes upstream wake effects. Multiple wakes combine through root-sum-square deficits:

$$
u(\mathbf{x})=\operatorname{clip}\left(
U_\infty-\sqrt{\sum_j\Delta U_j(\mathbf{x})^2},0,U_\infty\right).
$$

Nine rotor points approximate effective inflow. The centre has weight 0.2 and eight equally spaced ring points at radius $0.375D$ each have weight 0.1:

$$
U_i=\left(\sum_{q=1}^{9}w_q u_{iq}^3\right)^{1/3}.
$$

| Closure | $k_0$ | $k_I$ | $k_\delta$ |
|---|---:|---:|---:|
| Controller prior | 0.038 | 0.38 | 0.18 |
| Synthetic teacher | 0.023 | 0.30 | 0.30 |

These are explicit demonstration parameters. The implementation uses a simplified Gaussian closure, not the full published Bastankhah model or FLORIS. The teacher provides synthetic labels and evaluation; the optimizer does not query it to select actions.

## 3. Power and thrust curves

For the generic turbine, $P_r=5000$ kW, $U_{ci}=3$ m/s, $U_r=11.4$ m/s, $U_{co}=25$ m/s, and $p=1.88$:

$$
P(U,\gamma)=\begin{cases}
P_r\operatorname{clip}\left(\dfrac{U^3-U_{ci}^3}{U_r^3-U_{ci}^3},0,1\right)
\max(\cos\gamma,0)^p,& U_{ci}\leq U<U_{co},\\
0,&\text{otherwise}.
\end{cases}
$$

$$
C_T(U)=\begin{cases}
0.8\min\left(1,\left(\dfrac{U_r}{\max(U,0.1)}\right)^2\right),& U_{ci}\leq U<U_{co},\\
0,&\text{otherwise}.
\end{cases}
$$

This power curve is continuous at rated speed and applies a cosine yaw loss even above rated speed. Real turbines require appropriate controller-dependent power and thrust curves. See `Turbine.power` and `Turbine.thrust_coefficient`.

## 4. Shared conditional mean model

The same network is applied to every turbine. No absolute position or turbine ID enters the network. Features use streamwise geometry in rotor-diameter units and aggregated upstream context:

$$
r_{ji}=\mathbf{1}_{d_{ji}>0.1}
\frac{\exp[-\tfrac12(\ell_{ji}/1.5)^2]}{1+\max(d_{ji},0)/6},
$$

where $d_{ji}=\Delta x_{ji}/D$ and $\ell_{ji}=\Delta y_{ji}/D$. The 16 features are:

1. $U_\infty/12$, $I/0.15$, $\sin\gamma_i$, $\cos\gamma_i$, prior ratio $b_i=U_i^{prior}/U_\infty$, and $1-b_i$.
2. Four upstream sums $\sum_j r_{ji}\{1,\sin\gamma_j,\cos\gamma_j,\sin^2\gamma_j\}$.
3. Four geometry sums $\sum_j r_{ji}\{d_{ji}/12,\ell_{ji}/3,\ell_{ji}^2/9,d_{ji}^2/144\}$.
4. $\sum_j r_{ji}\ell_{ji}\sin\gamma_j$ and $\sum_j r_{ji}(d_{ji}/12)\cos\gamma_j$.

Each feature is standardized using **training observations only**, with a standard-deviation floor of 0.02. Each mean network is an MLP, $16\to64\to64\to1$, with SiLU activations. Its output is

$$
\hat b_i^{(m)}=\operatorname{clip}\left[
b_i+0.8(1-b_i)\tanh f_{\theta_m}(\mathbf{c}_i),0,1\right],
\quad
\bar b_i=\frac{1}{M}\sum_{m=1}^{M}\hat b_i^{(m)}.
$$

The correction is at most 80% of the prior deficit. It vanishes if $b_i=1$: this closure cannot learn a wake completely missed by the prior. Bounds represent the flat, wake-deficit setting; they do not represent terrain-induced speed-up.

The loss is pooled speed-ratio MSE:

$$
\mathcal{L}_{mean}=\frac{1}{N_{obs}}\sum_i(\hat b_i-y_i)^2.
$$

Default training uses three initialization seeds, AdamW at $2\times10^{-3}$, weight decay $10^{-4}$, batches of 1,024, gradient norm clipping at 1, at most 100 epochs, and patience 20. The lowest validation MSE selects each member. Members share the dataset; this is an initialization ensemble, not bootstrap training. Pooling means layouts with more observations carry more weight.

## 5. Conditional residual diffusion

After freezing the selected mean ensemble, normalized residual labels are

$$
z_{0,i}=\frac{y_i-\bar b_i}{s_r},\qquad
s_r=\max\left[\operatorname{Std}_{train}(y-\bar b),0.003\right].
$$

The forward diffusion and denoising loss are

$$
z_t=\sqrt{\bar a_t}z_0+\sqrt{1-\bar a_t}\epsilon,
\quad\epsilon\sim\mathcal{N}(0,1),
$$

$$
\mathcal{L}_{diff}=\mathbb{E}_{i,t,\epsilon}
\left[\left(\epsilon-\epsilon_\phi(z_t,t,\mathbf{c}_i)\right)^2\right].
$$

There are 32 diffusion steps. The cosine cumulative schedule uses offset 0.008; derived $\beta_t$ values are clipped to $[0.0001,0.999]$. The denoiser receives 16 conditions, the residual, and three time features $(t,\sin\pi t,\cos\pi t)$, with two 64-unit SiLU layers.

At reverse sampling, the predicted clean residual is clipped to $[-6,6]$ and inserted into the exact Gaussian DDPM posterior:

$$
\hat z_0=\operatorname{clip}\left(
\frac{z_t-\sqrt{1-\bar a_t}\epsilon_\phi}{\sqrt{\bar a_t}},-6,6\right),
$$

$$
\mu_t=\frac{\beta_t\sqrt{\bar a_{t-1}}}{1-\bar a_t}\hat z_0
+\frac{(1-\bar a_{t-1})\sqrt{1-\beta_t}}{1-\bar a_t}z_t,
\qquad
\tilde\beta_t=\beta_t\frac{1-\bar a_{t-1}}{1-\bar a_t}.
$$

Samples are mapped back by $\tilde b_i=\operatorname{clip}(\bar b_i+s_r\tilde z_{0,i},0,1)$. Default diffusion training uses 1,200 AdamW updates at $10^{-3}$, with validation every 100 updates and a fixed validation noise/time sample. The best validation denoising loss selects the checkpoint.

Samples are independent **marginal rotor-speed residuals**, including the synthetic observation noise. They are not a joint farm realization or a time trajectory. The sampled distribution is separate from ensemble disagreement used by the controller.

### Empirical interval calibration

For each calibration observation, calculate the sample 5th and 95th percentiles $(l_i,h_i)$ and

$$
s_i=\max(l_i-y_i,y_i-h_i,0).
$$

Let $q$ be the ordered score at rank $\min(n,\lceil0.9(n+1)\rceil)$. Report $[\max(0,l_i-q),\min(1,h_i+q)]$. Calibration layouts are separate from training and test layouts. Observations within a layout are dependent, so 90% is an empirical coverage target, not a guaranteed simultaneous farm or trajectory bound.

## 6. Direction-aware yaw optimization

For direction standard deviation $\sigma_\alpha$, approximate a Gaussian expectation with three points:

$$
(\Delta\alpha_s,w_s)=
\left\{(-\sqrt3\sigma_\alpha,\tfrac16),(0,\tfrac23),
(+\sqrt3\sigma_\alpha,\tfrac16)\right\}.
$$

The **same nacelle headings** are used in all scenarios, so
$\gamma_{i,s}=\gamma_i+\Delta\alpha_s$ and $\alpha_s=\alpha+\Delta\alpha_s$.

Let $P_{ref}=\max[N P(U_\infty,0),1\text{ kW}]$. The objective is

$$
J(\boldsymbol\gamma)=
\frac{\sum_s w_s\sum_i\hat P_{i,s}-\lambda_r\sum_s w_s\sum_i d_{i,s}}{P_{ref}}
-\lambda_m\frac1N\sum_i\left(\frac{\gamma_i-\gamma_i^{current}}{\gamma_{max}}\right)^2
-\lambda_l\frac1N\sum_i\sin^2\gamma_i.
$$

Defaults: $\lambda_r=0.1$, $\lambda_m=0.002$, $\lambda_l=0.002$. The disagreement proxy is

$$
d_{i,s}=\frac{P(\operatorname{clip}(\bar b_i+\sigma_{b_i},0,1)U_\infty,\gamma_{i,s})
-P(\operatorname{clip}(\bar b_i-\sigma_{b_i},0,1)U_\infty,\gamma_{i,s})}{2}.
$$

Here $\sigma_{b_i}$ is the population standard deviation across mean-network predictions. Summed power differences are a heuristic disagreement penalty, not farm-power standard deviation. The final term is a side-force proxy, not fatigue damage or a structural load prediction.

### Feasible commands and search

With nominal angle limit $\gamma_{max}=30^\circ$, yaw rate $r_{max}=0.3^\circ/s$, and interval $\Delta t=120$ s, each feasible yaw lies in

$$
[-\gamma_{max},\gamma_{max}]
\cap[\gamma_i^{current}-r_{max}\Delta t,\gamma_i^{current}+r_{max}\Delta t].
$$

The rate limit applies to absolute nacelle motion, including when the wind changes direction. Angle limits apply at the **nominal** direction; scenario misalignments can exceed them. With default direction uncertainty, a $30^\circ$ nominal yaw can be $33.46^\circ$ in a scenario.

1. Construct feasible hold/alignment commands, both angle bounds, 96 uniform random candidates, and positive/negative wake-steering ramps.
2. Check every candidate against the model-support rule in every direction scenario. Reject unsupported candidates.
3. Refine the best three supported starts with three coordinate sweeps. Each turbine uses a 13-point grid plus its current incumbent.
4. Compare refined candidates with feasible hold/alignment incumbents and return the best objective.
5. If no supported seed exists, repeat the whole search using the physics prior.

This is a deterministic seeded search, not a global optimality guarantee. Empty angle/rate intersections return `infeasible` and hold heading. Outside the generating-speed interval, the policy returns `parked` and holds heading, which can retain yaw outside the nominal angle bound.

## 7. Support checks and fallback

The model requires the saved turbine specification and no more turbines than the maximum count in its training layouts. For each feature, its support interval is

$$
[x_{min}-\eta,x_{max}+\eta],\quad
\eta=0.15\max(x_{max}-x_{min},0.01).
$$

This expanded coordinate-wise box allows limited extrapolation and does not detect every unusual combination. Direct prediction flags/falls back per scene, independently of other scenes in the batch. The controller rejects individual unsupported actions before inference. Returned `source`, `reason`, and rejection counts make the path visible.

## 8. Replay and evaluation

Alignment, physics control, and AI control each maintain independent heading states and receive the same wind sequence. A fixed synthetic teacher evaluates all actions. For each interval:

$$
E_k=P_{farm,k}\Delta t/3600\quad[\mathrm{kWh}],\qquad
G=100\left(\frac{\sum_k E_k^{method}}{\sum_k E_k^{alignment}}-1\right).
$$

This is a sequence of steady operating points. The model does not integrate wake advection, actuator travel power, or fatigue. Replay rejects plant misalignment beyond $60^\circ$. Prediction evaluation separately measures RMSE, MAE, bias, Pearson correlation, interval coverage and width. The [validation record](VALIDATION.md) gives all measured results.

## 9. Relationship to physics-informed learning

Physics informs the baseline wake, geometry features, admissible output range, turbine power conversion, and actuator constraints. The implemented learned loss consists of mean MSE and DDPM denoising loss. No differential-equation residual is included, so the code is described as **physics-guided**, rather than a Navier–Stokes PINN.

To introduce a PDE residual, the prediction target must first provide the required spatial and temporal derivatives. Rotor-speed labels alone do not determine those derivatives. A higher-fidelity extension can preserve the control API while replacing the wake model, observations, network, and validation environment.

## Conceptual references

- Bastankhah & Porté-Agel (2014), *A new analytical model for wind-turbine wakes*, [DOI: 10.1016/j.renene.2014.01.002](https://doi.org/10.1016/j.renene.2014.01.002). Gaussian-wake context; this repository's closure is simplified.
- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
- Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic Models*, [arXiv:2102.09672](https://arxiv.org/abs/2102.09672). Cosine diffusion schedule.
- [FLORIS](https://github.com/NREL/floris): a separate open-source wake-model and control tool; not a dependency or a claimed benchmark in this project.
