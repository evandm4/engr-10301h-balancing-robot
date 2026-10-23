1. Explore literature on the performance difference between PID and LQR systems for machine learning models, as well as real world application.
    a. consider adding an LQR controller
2. Institute continuous rather than stepped goal profiles, so that theoretical cost can be zero within the bounds of the model. 
    a. This could be modeled any number of ways, but the best is probably a basic sinusoidal curve such that the endpoints' derivatives (acceleration values) are zero.
    b. Other methods could include polynomial or piecewise linear equation fitting.
3. Reevaluate the ML cost weighting. See PHYSICS.md 10.5.
4. 