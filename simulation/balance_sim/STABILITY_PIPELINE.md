# Delay margins with pipeline sensing (`stability_pipeline.py`)

**Tool:** `python simulation/stability_pipeline_table.py` (~10 s). Model: `balance_sim/stability_pipeline.py`. Tests: `simulation/tests/test_stability_pipeline.py` (LEGO config only).
**Sources:** `reports/delay_margin_check/REPORT.md` and its adversarial review `review/REVIEW.md` (outside the repo).

## What changed and why

The review's **12.25 ms (L `l_e2_s1`) vs 25.25 ms (MB PID)** cruise margins come from `stability.py`. That model uses **ideal-state sensing**: the controller sees the true v, θ and θ̇. The LEGO policies are trained, evaluated and run with a different loop:

- a complementary filter;
- a delayed (5 ms), differenced (10 ms) encoder;
- back-EMF feedforward;
- the Build HAT's 10 ms PDM.

`stability_pipeline.py` linearizes that loop exactly as a 10 ms periodic system. It reproduces `stability.py` to round-off when every stage is off. It matches small-signal nonlinear `simulate()` runs to 1e-3 at standstill and 2e-3 at cruise (feedforward on). Its hardware-timing knobs match the reviewer's independent nonlinear loop (`nl_bisect.mysim`) to ≤ 6.5e-4 of the peak, at standstill and at cruise. Linearize at the speed L actually cruises at: ~0.38 m/s, because its leaky integrators leave it ~5 % slow. The margins are insensitive to that (≤ 1 ms, best phase only). The knobs, all off by default:

| Knob | What it models | Default `hardware_like` |
|---|---|---|
| `imu_delay_sub` | IMU-only delay (the ~3 ms ICM DLPF) on the gyro and accelerometer, not on the encoder path. The sim lumps the DLPF into `latency`. | 3 ms |
| `right_lag_sub` | The right wheel acts late (command-string order on the HAT) | 2.5 ms |
| `right_enc_age_sub` | The right encoder runs on its own 10 ms grid. Its per-run phase is arbitrary on the robot. | = left (5 ms) |

## Margins (linear; ms of total sample→HAT delay; worst–best over the free PDM phase in 1 ms steps; cruise 0.40 m/s)

| Controller | Condition | Ideal (`stability.py`) | Sim pipeline | Hardware-like (DLPF split + right lag) |
|---|---|---|---|---|
| L | cruise nominal | 12.25 | 29–35 | 23–26.5 |
| L | cruise corner | 10 | 22–26 | 18–20 |
| MB | cruise nominal | 25.25 | 26–31 | 24–26.5 |
| MB | cruise corner | 21.75 | 20–24 | 19–21 |
| L | standstill | 80+ | 80+ | 80+ |
| MB | standstill nominal / corner | 65.5 / 51 | unstable* | unstable* |

\*A bounded ~0.5 Hz limit cycle from the CF's accelerometer coupling (REPORT §3). It is not a fall risk; the real MB robot balances.

DLPF sweep, linear, no PDM, cruise nominal (0 / 1.5 / 3 / 4.5 ms):
- L: 37.0 / 33.75 / 31.0 / 27.75 ms;
- MB: 33.0 / 32.25 / 31.5 / 30.75 ms.

L loses ~2 ms per ms of DLPF; MB loses ~0.5 ms per ms. L's gains are tuned to the *relative* lag between encoder and IMU, and the hardware changes that lag. A right encoder 2.5 ms younger or older moves L by −2.5 / +3 ms and MB by −0.75 / +0.5 ms.

## What to say

- The review's **12.25 / 25.25 ms** were computed with ideal-state sensing. "L's margin is half the PID's" is **wrong for the LEGO loop** and must not appear.
- With pipeline sensing, the reviewer's nonlinear bisection gives:
  - L **≈ 21–31 ms hardware-like** (23–35 under the sim's own lumping);
  - MB **26–33 ms**.

  This linear tool gives somewhat lower hardware-like values for both:
  - L 18–26.5 ms;
  - MB 19–26.5 ms.

  The difference is that the tool scores linear instability onset over 10 PDM phases. The review's PDM bisection counted only falls or fast growth over 4 phases. Either way the two controllers are **comparable**: tied within a few ms, with MB slightly ahead under hardware-like timing. Write **"comparable delay tolerance"**, not "L ≥ PID".
- **Backlash** is not in the linear model. In the nonlinear sim it drives a **~4 Hz cruise limit cycle for L** (1.5–2° RMS at 10–15 ms). A 4 Hz peak on hardware is therefore not evidence of the ideal-sensing 3.8 Hz mode. Onset is discriminated by *when* growth starts, in the 1.3–3.3 Hz band.
- The robot's **base delay is 9.5 ± 2.5 ms**. Only ~3.4 ms of it is measured; the DLPF ~3 ms and HAT apply ~3 ms are estimates. Injected-delay onset is expected near (margin − 9.5) ms, minus up to ~5 ms for noise and robust effects.
- These margins apply to the LEGO robot only. Pololu sensing is ideal state + noise, so `stability.py` remains the right model there.
