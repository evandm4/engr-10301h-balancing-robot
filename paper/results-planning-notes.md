# Results & Conclusions — Planning Notes

Nov 9, 2026 · @Evan Moore

Working notes for drafting the Results and Conclusions of the balancing-robot paper. Numbers come from `results/lqr_comparison/20261109-000021` (latency drawn from 2.5–20 ms, LQR gain search with a guarded integral gain), validated on a third held-out set, seeds 3000–3099. Two earlier sets were retired: one informed the scenario design, and results on the other were seen before those two changes.

## Central thesis

ML tuning's main gain over a competent model-based design is tolerance of delay and disturbance, not typical-case performance. On a typical run the textbook PID designed from the linear model is only 34% worse than the ML-tuned PID (median per scenario). But its loop is stable only up to 23.75 ms of latency, against 37.5 ms, and in the latency sweep it starts falling at 20 ms where the ML-tuned PID holds to 30 ms. At normal latencies the evidence is thinner: across three held-out sets the model-based PID fell 3 times in 900 runs (2, 1 and 0), and the ML-tuned PID never fell. That is a stronger and more honest claim than "ML beats a hand-tuned guess", and the conclusions should be framed around it.

## Evidence

Validation on 300 unseen robots and noise seeds, 60 s runs, sorted by median cost relative to the ML-tuned PID. Mean costs are dominated by falls (each adds 1000–2000), so median ratio and fall count are the fair comparison.

| Controller | Median cost vs ML PID | Mean cost | Falls (of 300) | Linear stable latency | First falls in latency sweep |
| --- | --- | --- | --- | --- | --- |
| PID, ML-tuned | 1.00× | 2.32 | 0 | 0–37.5 ms | 40 ms (3 of 60) |
| LQR, ML-tuned gains (with integral) | 1.09× | 2.54 | 0 | 0–35.0 ms | 40 ms (26) |
| PID, model-based | 1.34× | 3.15 | 0 | 0–23.75 ms | 20 ms (1) |
| LQR, ML-tuned weights | 1.54× | 23.41 | 3 | 0–39.0 ms | 20 ms (4) |
| LQR, latency ignored | 3.14× | 7.28 | 0 | 0–36.5 ms | 40 ms (1) |
| LQR, cost-matched | 4.35× | 16.35 | 1 | 0–44.25 ms | 20 ms (1) |
| PID, hand-tuned | 5.80× | 23.23 | 1 | 0–15.75 ms | 20 ms (1) |

## Supporting findings

- **The tuned gains are reproducible.** Two independent searches (the batch tuner and the LQR comparison) both land on k\_p 1.615, k\_i 1.75, k\_d 0.298, k\_v 0.626 (and moved by at most 1.2% when the latency range was widened).
- **Why the tuner moves where it does.** Bound (7) caps the outer gain at (c/e)(k\_d/k\_p). The tuned PID roughly doubles both k\_d and k\_v relative to the model-based design, which buys delay margin (23.75 → 37.5 ms).
- **Standard design rules fail here.** An outer bandwidth of z/2 is unstable even without delay, and an underdamped inner loop is unstable at 10 ms. The coupling through the RHP zero is the reason.
- **Pushes are what separate controllers.** In the batch tuning run, both hand-tuned falls came from the forward push at top speed and disappeared when the pushes were removed. The falls on the current held-out set (3 weight-tuned LQR, 1 hand-tuned, 1 cost-matched LQR) haven't been traced yet.
- **Delay tolerance differs from typical cost.** The latency-ignoring LQR is mediocre on typical runs (3.14×) but the most delay-tolerant (one fall at 40 ms). Integral action matters for the LQR: with a guarded integral gain the directly tuned LQR went from 34% to 9% behind the ML-tuned PID.
- **The LQR weight search can't avoid falls** with only its integral weight free (3 falls on the held-out set). Re-opening the other weights is untested.
- **The stability guard binds, which shows what it is protecting against.** The best directly tuned LQR sits exactly on the 0.999 spectral-radius limit: the optimizer pushes toward slower, near-unstable integral action because it scores better within a 20 s run. An unguarded search could likely close the remaining 9% gap, but an earlier unguarded search found exactly that kind of gain (an eigenvalue of 1.0009, a slow runaway the training runs never exposed). The guard trades a little measured performance for a stability guarantee the score alone cannot provide: a concrete case of an optimizer exploiting the limits of its own test, and of why a black-box tuner needs constraints from analysis.

## Caveats to state

- **Simulation only.** No physical robot; constants are datasheet values and estimates (friction, rotor inertia, latency).
- **Motor driver.** Results assume a current-controlled driver. With a voltage-mode driver capped at the torque limit, every gain set falls.
- **Idealized pitch estimate.** No filter is simulated; the controller sees true pitch plus noise and an offset.
- **One scenario family.** Step commands, slopes and rough ground are untested.
- **Fall penalty bound covers 20 s training runs only.** The 60 s validation runs can exceed it, so validation reports falls as a separate count.

## Open items before drafting

- [ ] Trace which scenarios make the model-based PID and the LQRs fall (push, latency draw, or start tilt)
- [ ] Paired statistics: confidence intervals on the median ratios over the 300 shared scenarios
- [ ] Figures: per-scenario cost distributions, latency sweep with falls marked, one example run with the pushes
- [ ] Decide whether the hand-tuned PID appears in Results at all, or only in the repo
- [ ] Optional re-run: LQR weight search with all weights free (the guarded integral gain is done)
- [ ] Decide whether to re-run the pivot side project on the new scenario or leave it out of the paper
