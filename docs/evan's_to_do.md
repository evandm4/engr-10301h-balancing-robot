1. Explore literature on the performance difference between PID and LQR systems for machine learning models, as well as real world application.
    a. consider adding an LQR controller
2. Institute continuous rather than stepped goal profiles, so that theoretical cost can be zero within the bounds of the model. 
    a. This could be modeled any number of ways, but the best is probably a basic sinusoidal curve such that the endpoints' derivatives (acceleration values) are zero.
    b. Other methods could include polynomial or piecewise linear equation fitting.
3. Reevaluate the ML cost weighting. See PHYSICS.md 10.5.
    a. Done 2026-10-24: Bryson-rule weights, set by three limits in config.py's CostConfig (pitch 0.3 rad, velocity 0.2 m/s, torque 0.1 N m).
    b. Done 2026-10-24: kv_p bound raised (0.4 to 0.9; a run at 2.0 found the optimum at kv_p about 0.82); run_batch_tuning.py and run_lqr_comparison.py re-run (results/batch_tuning/20261024-133625 and -135453, results/lqr_comparison/20261024-140819).
    c. Done 2026-10-24: pitch-rate weight and the commands-in-flight gains held at off (LQRConfig); the gain search no longer ends on a boundary and its cost is unchanged (results/lqr_comparison/20261024-142418).
    d. Open: the LQR weight search still ends in a corner (w_velocity at its floor, w_torque at its ceiling, cost 0.1857 vs 0.1304 for the PID). Not tested: widening further, or a different weight parameterization. Other results (pivot, PHYSICS.md 8.2) still use the old cost.
    e. Note the cost can't reach zero even with a smooth goal: the start tilt has to be recovered (about 0.027 of pitch cost).
4. 