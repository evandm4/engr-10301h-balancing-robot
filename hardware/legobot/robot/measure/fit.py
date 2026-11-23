"""Fitting functions for the bench measurements (SOFTWARE_PLAN C3, DATASHEET section 8).

numpy and the standard library only, so the same file re-fits a logged CSV on a Windows laptop.
Every function is pure: arrays in, numbers out. Times are seconds, angles radians, speeds rad/s.

Motor model used throughout (at the output shaft, one motor): drive voltage V = g(duty), motor torque
a (V - k w) with a = eta k / R, friction tau_c sgn(w) + b w, inertia J. The HAT has no current sensor, so
only k, a, J, tau_c, b and g are identifiable without a multimeter. Encoder: 1 count = 1 degree.

Standard errors from the least-squares fits assume independent residuals. Encoder quantization makes
residuals correlated, so these SEs are somewhat optimistic; the scripts also report the spread across
repeated trials, which is the more honest number.
"""
import math
from collections import namedtuple

import numpy as np


# ---------------------------------------------------------------------------------------------
# 1. Linear least squares
# ---------------------------------------------------------------------------------------------
def linfit_cov(X, y):
    """Ordinary least squares y = X coef. Returns (coef, cov, resid_sd).

    cov = s2 inv(X'X) with s2 = SSE / (n - p). X may be a 1-D array (one column, no intercept)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n <= p:
        raise ValueError(f"need more than {p} points, got {n}")
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ coef
    s2 = float(r @ r) / (n - p)
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef, cov, math.sqrt(s2)


def linfit(X, y):
    """Least squares y = X coef. Returns (coef, se, resid_sd), se = sqrt(diag(s2 inv(X'X)))."""
    coef, cov, sd = linfit_cov(X, y)
    return coef, np.sqrt(np.diag(cov)), sd


# ---------------------------------------------------------------------------------------------
# 2. Steady speed
# ---------------------------------------------------------------------------------------------
def steady_speed(t, pos_rad, t0, t1):
    """Constant-speed model pos = p0 + w (t - t_mid) on t0 <= t <= t1. Returns (w, se_w) in rad/s.

    (nan, nan) if fewer than 4 samples fall in the window."""
    t = np.asarray(t, float)
    pos = np.asarray(pos_rad, float)
    m = (t >= t0) & (t <= t1)
    if m.sum() < 4:
        return float("nan"), float("nan")
    tt = t[m] - t[m].mean()
    coef, se, _ = linfit(np.column_stack([np.ones(tt.size), tt]), pos[m])
    return float(coef[1]), float(se[1])


# ---------------------------------------------------------------------------------------------
# 3. Spin-up from a duty step
# ---------------------------------------------------------------------------------------------
SpinFit = namedtuple("SpinFit", "omega tau t_d rms theta0")
TAU_GRID = np.geomspace(0.005, 0.300, 120)       # s
TD_GRID = np.arange(0.0, 0.0405, 0.0005)         # s


def _spin_basis(s, tau):
    """Position basis for a first-order speed rise starting at s = 0: u - tau (1 - exp(-u/tau)), u = max(s, 0)."""
    u = np.maximum(s, 0.0)
    return u - tau * (1.0 - np.exp(-u / tau))


def fit_spinup(t, pos_rad, t_cmd, t_end=0.6):
    """Fit a delayed first-order speed rise to POSITION (not to differenced speed).

    Model: for s = t - (t_cmd + t_d) > 0,  theta = theta0 + w_inf [ s - tau (1 - exp(-s/tau)) ],
    and theta = theta0 before. That is speed w(s) = w_inf (1 - exp(-s/tau)) after a pure delay t_d.
    Uses data from t_cmd - 0.05 s to t_cmd + t_end. Grid search over tau (5-300 ms, 120 log points) and
    t_d (0-40 ms, 0.5 ms steps), then a finer local grid; for each pair theta0 and w_inf follow from linear
    least squares, and the pair with the smallest SSE wins.

    Returns SpinFit(omega=w_inf, tau, t_d, rms=rms residual, theta0). w_inf is signed. If the motor did
    not move, tau and t_d are meaningless: the caller should check |omega| first. A t_d equal to 0 or 40 ms
    means the search hit its edge."""
    t = np.asarray(t, float)
    y = np.asarray(pos_rad, float)
    m = (t >= t_cmd - 0.05) & (t <= t_cmd + t_end)
    t, y = t[m] - t_cmd, y[m]
    if t.size < 10:
        raise ValueError("fewer than 10 samples around the step")
    ym = y.mean()
    yc = y - ym
    syy = float(yc @ yc)

    def scan(taus, tds):
        best = (np.inf, None, None, None, None)
        for td in tds:
            f = _spin_basis((t - td)[None, :], taus[:, None])
            fm = f.mean(axis=1, keepdims=True)
            fc = f - fm
            sff = (fc * fc).sum(axis=1)
            sfy = fc @ yc
            ok = sff > 1e-12
            w = np.where(ok, sfy / np.where(ok, sff, 1.0), 0.0)
            sse = np.where(ok, syy - w * sfy, syy)
            i = int(np.argmin(sse))
            if sse[i] < best[0]:
                best = (float(sse[i]), float(taus[i]), float(td), float(w[i]), float(ym - w[i] * fm[i, 0]))
        return best

    sse, tau, td, w, th0 = scan(TAU_GRID, TD_GRID)
    # local refinement: tau +-10 % in 0.5 % steps, t_d +-1 ms in 0.1 ms steps
    sse, tau, td, w, th0 = scan(tau * np.linspace(0.9, 1.1, 41), np.maximum(td + np.arange(-0.001, 0.00101, 0.0001), 0.0))
    return SpinFit(w, tau, td, math.sqrt(sse / t.size), th0)


# ---------------------------------------------------------------------------------------------
# 4. Coast-down
# ---------------------------------------------------------------------------------------------
CoastFit = namedtuple("CoastFit", "alpha beta se_alpha se_beta rms n")


def window_slopes(t, y, centers, width):
    """Least-squares slope of y against t in a window of the given width around each center (nan if
    fewer than 4 samples)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    out = np.full(len(centers), np.nan)
    for i, c in enumerate(centers):
        lo = np.searchsorted(t, c - width / 2)
        hi = np.searchsorted(t, c + width / 2, side="right")
        if hi - lo >= 4:
            tt = t[lo:hi] - t[lo:hi].mean()
            yy = y[lo:hi]
            out[i] = (tt @ (yy - yy.mean())) / (tt @ tt)
    return out


def fit_coastdown(t, pos_rad, t_coast, width=0.05, step=0.01, min_speed=0.5, skip=0.02):
    """Coast-down model w_dot = -alpha sgn(w) - beta w, alpha = tau_c / J, beta = b / J.

    Speed comes from position by local linear fits in sliding 50 ms windows every 10 ms, starting `skip`
    seconds after t_coast (so the drive's electrical decay is excluded). Acceleration is the central
    difference of three consecutive window speeds. Samples whose three speeds all exceed min_speed
    (0.5 rad/s) in magnitude with the same sign are kept, then
    w_dot = -alpha sgn(w) - beta w is fitted by least squares. Returns CoastFit(alpha, beta, se_alpha,
    se_beta, rms, n); the SEs are optimistic because neighbouring windows overlap.

    CAVEAT: the HAT's `coast` is assumed to be open-circuit. If it is actually a brake (windings
    shorted), beta includes electrical braking, a k^2 / R term, and is not the mechanical b / J."""
    t = np.asarray(t, float)
    pos = np.asarray(pos_rad, float)
    m = t >= t_coast + skip
    t, pos = t[m], pos[m]
    if t.size < 10:
        raise ValueError("too few samples after the coast command")
    centers = np.arange(t[0] + width / 2, t[-1] - width / 2, step)
    w = window_slopes(t, pos, centers, width)
    wc = w[1:-1]
    wdot = (w[2:] - w[:-2]) / (2 * step)
    ok = (np.isfinite(w[:-2]) & np.isfinite(w[1:-1]) & np.isfinite(w[2:])
          & (np.abs(w[:-2]) > min_speed) & (np.abs(wc) > min_speed) & (np.abs(w[2:]) > min_speed)
          & (np.sign(w[:-2]) == np.sign(wc)) & (np.sign(w[2:]) == np.sign(wc)))
    if ok.sum() < 10:
        raise ValueError("fewer than 10 usable coast-down samples")
    wc, wdot = wc[ok], wdot[ok]
    X = np.column_stack([-np.sign(wc), -wc])
    coef, se, sd = linfit(X, wdot)
    return CoastFit(float(coef[0]), float(coef[1]), float(se[0]), float(se[1]), sd, int(ok.sum()))


# ---------------------------------------------------------------------------------------------
# 5. Load drop (string and mass)
# ---------------------------------------------------------------------------------------------
def fit_load_drop(rows):
    """rows: (duty, tau_L, w_loaded, w_noload_same_duty), speeds in the lifting direction (rad/s).

    Model tau_L = c (w_nl - w) with c = a k + b. Both runs have the same drive voltage V = g(duty), so
    g cancels and the model holds even if g(duty) is nonlinear (so does the Coulomb friction, which acts
    the same way in both runs). Fitted through the origin: c = sum(x y) / sum(x^2), x = w_nl - w,
    y = tau_L, se = sqrt(s2 / sum(x^2)) with s2 = SSE / (n - 1).

    Returns dict: c, se, n, resid (N m, one per row), per_mass {tau_L: mean residual}. A residual that
    grows with the mass or changes sign with it means the model is not linear in load."""
    r = np.asarray(rows, float)
    if r.ndim != 2 or r.shape[1] != 4 or len(r) < 2:
        raise ValueError("need at least two rows of (duty, tau_L, w_loaded, w_noload)")
    x = r[:, 3] - r[:, 2]
    y = r[:, 1]
    sxx = float(x @ x)
    if sxx < 1e-9:
        raise ValueError("loaded and no-load speeds are identical: no load effect to fit")
    c = float(x @ y) / sxx
    resid = y - c * x
    s2 = float(resid @ resid) / (len(r) - 1)
    per = {}
    for m in sorted(set(np.round(r[:, 1], 9))):
        per[float(m)] = float(resid[np.isclose(r[:, 1], m)].mean())
    return {"c": c, "se": math.sqrt(s2 / sxx), "n": len(r), "resid": resid, "per_mass": per}


# ---------------------------------------------------------------------------------------------
# 6. Balance point from free falls
# ---------------------------------------------------------------------------------------------
def local_slope(t, y, n=5):
    """Derivative of y by a least-squares line through n consecutive samples (n odd), using the real
    timestamps. For evenly spaced samples this is the n-point Savitzky-Golay first derivative
    (quadratic, window n): (-2, -1, 0, 1, 2) / (10 dt) for n = 5. Edges (n // 2 samples each end) are nan."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    h = n // 2
    out = np.full(len(t), np.nan)
    for i in range(h, len(t) - h):
        tt = t[i - h:i + h + 1]
        tt = tt - tt.mean()
        yy = y[i - h:i + h + 1]
        out[i] = (tt @ (yy - yy.mean())) / (tt @ tt)
    return out


def balance_segment(t, rate, theta_start, n=5):
    """From one free fall: gyro rate (bias already removed) at times t, the angle at t[0] taken from the
    accelerometer. Returns (theta, theta_dot, theta_ddot): theta = theta_start + trapezoid integral of the
    rate, theta_dot = rate, theta_ddot = local_slope(rate) over n samples (5 samples = 20 ms at 200 Hz,
    close to the ~25 ms in the plan). The first and last n // 2 theta_ddot values are nan."""
    t = np.asarray(t, float)
    rate = np.asarray(rate, float)
    theta = theta_start + np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    return theta, rate, local_slope(t, rate, n)


def _balance_ls(theta, theta_dot, theta_ddot):
    X = np.column_stack([theta, theta_dot, np.ones(len(theta))])
    coef, cov, sd = linfit_cov(X, theta_ddot)
    return X, coef, cov, sd


def fit_balance(theta, theta_dot, theta_ddot):
    """Free-fall model of an inverted pendulum near upright: theta_ddot = c theta + e theta_dot + h.

    With theta0 the angle where the acceleration is zero (the balance point), h = -c theta0, so
    theta0 = -h / c. Its SE is the delta method with the coefficient covariance matrix:
    var = g' Cov g, g = (h / c^2, 0, -1 / c) for (c, e, h). One trim pass: rows with |resid| above 3
    scaled MAD (1.4826 MAD) are dropped and the fit repeated. Rows with a nan are ignored.

    Returns dict: c, e, h, theta0, se_theta0, se_c, resid_sd, n, n_dropped."""
    th = np.asarray(theta, float)
    td = np.asarray(theta_dot, float)
    tdd = np.asarray(theta_ddot, float)
    ok = np.isfinite(th) & np.isfinite(td) & np.isfinite(tdd)
    th, td, tdd = th[ok], td[ok], tdd[ok]
    X, coef, cov, sd = _balance_ls(th, td, tdd)
    r = tdd - X @ coef
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    keep = np.abs(r) <= 3 * mad if mad > 0 else np.ones(len(r), bool)
    dropped = int((~keep).sum())
    if dropped:
        X, coef, cov, sd = _balance_ls(th[keep], td[keep], tdd[keep])
    c, e, h = coef
    g = np.array([h / c**2, 0.0, -1.0 / c])
    return {"c": float(c), "e": float(e), "h": float(h), "theta0": float(-h / c),
            "se_theta0": float(math.sqrt(g @ cov @ g)), "se_c": float(math.sqrt(cov[0, 0])),
            "resid_sd": sd, "n": int(keep.sum()), "n_dropped": dropped}


# ---------------------------------------------------------------------------------------------
# 7. Allan deviation
# ---------------------------------------------------------------------------------------------
def allan_dev(x, fs, taus):
    """Overlapping Allan deviation of a rate signal x sampled at fs (Hz) at averaging times taus (s).

    With phase p = cumulative sum of x / fs and m = round(tau fs):
    avar(tau) = mean over k of (p[k+2m] - 2 p[k+m] + p[k])^2 / (2 tau^2).
    Returns (taus_actual = m / fs, adev), adev = nan where 2 m exceeds the record. For white noise of
    standard deviation s, adev = s / sqrt(m)."""
    x = np.asarray(x, float)
    p = np.concatenate([[0.0], np.cumsum(x) / fs])
    ta, ad = [], []
    for tau in taus:
        m = max(1, int(round(tau * fs)))
        ta.append(m / fs)
        if 2 * m >= len(p):
            ad.append(float("nan"))
            continue
        d = p[2 * m:] - 2 * p[m:-m] + p[:-2 * m]
        ad.append(math.sqrt(float(d @ d) / (2 * (m / fs) ** 2 * len(d))))
    return np.array(ta), np.array(ad)
