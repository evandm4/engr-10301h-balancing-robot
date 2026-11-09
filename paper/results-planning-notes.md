# Results & Conclusions — Planning Notes

Nov 9, 2026 · @Evan Moore

Working notes for drafting the Results and Conclusions of the balancing-robot paper. Numbers come from the 20 s scenario runs of 2026-11-08 (`results/lqr_comparison/20261108-220417`).

## Central thesis

ML tuning's main gain over a competent model-based design is robustness, not typical-case performance. On a typical run, the textbook PID designed from the linear model is only 36% worse than the ML-tuned PID (median per scenario). But it falls on 2 of 300 unseen robots and tolerates 23.75 ms of latency, against no falls and 37.25 ms for the ML-tuned PID. That is a stronger and more honest claim than "ML beats a hand-tuned guess", and the conclusions should be framed around it.

## Evidence

Validation on 300 unseen robots and noise seeds, 60 s runs, sorted by median cost relative to the ML-tuned PID. Mean costs are dominated by falls (each adds 1000–2000), so median ratio and fall count are the fair comparison.

| Controller | Median cost vs ML PID | Mean cost | Falls (of 300) | Linear stable latency | First falls in latency sweep |
| --- | --- | --- | --- | --- | --- |
| PID, ML-tuned | 1.00× | 2.26 | 0 | 0–37.25 ms | 40 ms (3 of 60) |
| LQR, ML-tuned gains | 1.35× | 3.05 | 0 | 0–35.00 ms | 30 ms (1) |
| PID, model-based | 1.36× | 15.34 | 2 | 0–23.75 ms | 20 ms (2) |
| LQR, ML-tuned weights | 1.66× | 48.92 | 7 | 0–42.25 ms | 25 ms (1) |
| LQR, latency ignored | 3.15× | 7.08 | 0 | 0–36.50 ms | none up to 40 ms |
| LQR, cost-matched | 4.35× | 22.32 | 2 | 0–44.25 ms | 20 ms (1) |
| PID, hand-tuned | 6.18× | 30.02 | 2 | 0–15.75 ms | 20 ms (2) |

## Supporting findings

- **The tuned gains are reproducible.** Two independent searches (the batch tuner and the LQR comparison) both land on k\_p 1.635, k\_i 1.77, k\_d 0.299, k\_v 0.620.
- **Why the tuner moves where it does.** Bound (7) caps the outer gain at (c/e)(k\_d/k\_p). The tuned PID roughly doubles both k\_d and k\_v relative to the model-based design, which buys delay margin (23.75 → 37.25 ms).
- **Standard design rules fail here.** An outer bandwidth of z/2 is unstable even without delay, and an underdamped inner loop is unstable at 10 ms. The coupling through the RHP zero is the reason.
- **Pushes are what separate controllers.** Both hand-tuned falls come from the forward push at top speed and disappear when the pushes are removed. The other controllers' falls haven't been traced yet.
- **Delay tolerance differs from typical cost.** The latency-ignoring LQR is mediocre on typical runs (3.15×) but the most delay-tolerant (no falls up to 40 ms).
- **The LQR weight search can't avoid falls** with only its integral weight free (7 falls). Re-opening the other weights is untested.

## Caveats to state

- **Simulation only.** No physical robot; constants are datasheet values and estimates (friction, rotor inertia, latency).
- **Motor driver.** Results assume a current-controlled driver. With a voltage-mode driver capped at the torque limit, every gain set falls.
- **Idealized pitch estimate.** No filter is simulated; the controller sees true pitch plus noise and an offset.
- **LQR gain search has no integral term**, which friction penalizes; the PID-vs-LQR margin partly reflects that.
- **One scenario family.** Step commands, slopes and rough ground are untested.
- **Fall penalty bound covers 20 s training runs only.** The 60 s validation runs can exceed it, so validation reports falls as a separate count.

## Open items before drafting

- [ ] Trace which scenarios make the model-based PID and the LQRs fall (push, latency draw, or start tilt)
- [ ] Paired statistics: confidence intervals on the median ratios over the 300 shared scenarios
- [ ] Figures: per-scenario cost distributions, latency sweep with falls marked, one example run with the pushes
- [ ] Decide whether the hand-tuned PID appears in Results at all, or only in the repo
- [ ] Optional re-run: LQR weight search with all weights free, and an LQR gain search with a guarded integral term
- [ ] Decide whether to re-run the pivot side project on the new scenario or leave it out of the paper
