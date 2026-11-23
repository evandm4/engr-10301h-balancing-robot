"""Derived numbers for hardware/legobot/DATASHEET.md (the LEGO SPIKE Prime build).

Run:  python hardware/legobot/legobot_numbers.py

Every input is a datasheet value or an estimate quoted in DATASHEET.md, with its
source in the comment. Nothing here touches config.py; it only prints numbers.
"""

import math

G = 9.81
RPM = 2 * math.pi / 60


def heading(text):
    print(f"\n=== {text} ===")


# --------------------------------------------------------------- motors ---
# Pybricks motor_data.py ("Data from experiments by Pybricks authors"), SI units:
# test voltage V, a loaded point (tau_x, i_x, w_x), the no-load point (tau_0, i_0, w_0),
# and an estimated no-load acceleration a. Constants are output-shaft quantities.
PYBRICKS = {
    "Large (45602)": dict(V=7.2, tau_x=1018.64 / 1000 * 9.81 / 100, i_x=0.53, w_x=12.7,
                          tau_0=0.0, i_0=0.10, w_0=16.6, a=math.radians(800 / 0.04), L=0.0004 * 30),
    "Medium (45603)": dict(V=7.2, tau_x=1018.64 / 1000 * 9.81 / 100, i_x=0.51, w_x=4.6,
                           tau_0=0.0, i_0=0.09, w_0=16.6, a=math.radians(920 / 0.035), L=0.0008 * 30),
}

# LEGO Education technical fact sheets, all at 7.2 V: (no-load rpm, no-load A,
# max-eff N m, max-eff rpm, max-eff A, stall N m, stall A). Tolerance +/-15%.
LEGO = {
    "Large (45602)": (175, 0.135, 0.08, 135, 0.43, 0.25, 1.40),
    "Medium (45603)": (185, 0.110, 0.035, 135, 0.28, 0.18, 0.80),
}


def pybricks_model(V, tau_x, i_x, w_x, tau_0, i_0, w_0, a, L):
    """Same formulas as Pybricks motor_model.make_model."""
    kt = (tau_x - tau_0) / (i_x - i_0)            # output torque per amp, losses included
    tau_s = kt * i_0 - tau_0                       # Coulomb friction torque (per motor)
    r_over_ke = kt * (w_x - w_0) / (tau_0 - tau_x)
    ke = V / (w_0 + r_over_ke / kt * (tau_0 + tau_s))
    R = r_over_ke * ke
    J = (kt * V / R - tau_s) / a                   # output-side inertia (rotor reflected + output)
    return dict(kt=kt, ke=ke, R=R, tau_s=tau_s, J=J, L=L)


def lego_model(rpm0, i0, tau_me, rpm_me, i_me, tau_st, i_st, V=7.2):
    """Linear motor through the no-load and stall points of the fact sheet."""
    w0 = rpm0 * RPM
    R = V / i_st
    ke = (V - i0 * R) / w0
    kt = tau_st / i_st
    # Check the linear model against the published max-efficiency point.
    w_me_lin = w0 * (1 - tau_me / tau_st)
    i_me_lin = i0 + (i_st - i0) * tau_me / tau_st
    eff = tau_me * rpm_me * RPM / (V * i_me)
    return dict(w0=w0, R=R, ke=ke, kt=kt, w_me_lin=w_me_lin, i_me_lin=i_me_lin, w_me=rpm_me * RPM,
                i_me=i_me, eff=eff)


heading("Motor constants, output shaft, per motor (DATASHEET 5.1)")
models = {}
for name in PYBRICKS:
    p = pybricks_model(**PYBRICKS[name])
    l = lego_model(*LEGO[name])
    models[name] = p
    print(f"{name}")
    print(f"  Pybricks: R = {p['R']:.2f} ohm, k_e = {p['ke']:.4f} V s/rad, k_t (net) = {p['kt']:.4f} N m/A, "
          f"eta = k_t/k_e = {p['kt'] / p['ke']:.3f}, tau_coulomb = {p['tau_s']:.4f} N m, "
          f"J_out = {p['J']:.3e} kg m^2, L/R = {1000 * p['L'] / p['R']:.1f} ms")
    st_i = 7.2 / p["R"]
    print(f"            stall at 7.2 V: {st_i:.2f} A, {p['kt'] * st_i - p['tau_s']:.3f} N m;  "
          f"mechanical time constant J R/(k_t k_e) = {1000 * p['J'] * p['R'] / (p['kt'] * p['ke']):.0f} ms")
    print(f"  LEGO sheet: R = {l['R']:.2f} ohm, k_e = {l['ke']:.4f} V s/rad, k_t = {l['kt']:.4f} N m/A, "
          f"eta = {l['kt'] / l['ke']:.3f}, no-load {l['w0']:.2f} rad/s")
    print(f"            linear model at the max-efficiency torque: {l['w_me_lin']:.2f} rad/s, {l['i_me_lin']:.3f} A "
          f"(sheet: {l['w_me']:.2f} rad/s, {l['i_me']:.3f} A); motor+gearbox efficiency there {100 * l['eff']:.0f}%")
    print(f"            no-load drag k_t i_0 = {l['kt'] * LEGO[name][1]:.4f} N m (Pybricks k_t: "
          f"{p['kt'] * LEGO[name][1]:.4f})")

# ------------------------------------------------------ wheel-side view ---
heading("Wheel side, both motors (DATASHEET 5.2-5.3)")
WHEELS = {"56 x 14 mm (39367)": 0.028, "69 mm (planned)": 0.0345, "88 x 14.34 mm (49295)": 0.044}
for name, p in models.items():
    for wname, r in WHEELS.items():
        J2 = 2 * p["J"]
        print(f"{name}, wheel {wname}: rotor+gear inertia 2 J = {J2:.2e} kg m^2 = {J2 / r**2:.2f} kg "
              f"of equivalent translating mass")
        for V in (6.8, 7.2, 7.6, 8.4):
            w_free = (V - p["R"] * p["tau_s"] / p["kt"]) / p["ke"]   # no-load: k_t i = tau_s
            stall = 2 * (p["kt"] * V / p["R"] - p["tau_s"])
            print(f"   V = {V:.1f}: no-load {w_free:5.2f} rad/s = {w_free * r:.2f} m/s; stall (both) {stall:.3f} N m "
                  f"= {stall / r:5.1f} N at the ground; voltage-mode drag c_m = n k_t k_e/R = "
                  f"{2 * p['kt'] * p['ke'] / p['R']:.4f} N m s/rad")

# ----------------------------------------------------------- wheel size ---
heading("Wheel size: axle acceleration and torque left at speed, Large motors (DATASHEET 5.6)")
# Axle acceleration per N m of wheel torque, ignoring the pendulum coupling:
# a = tau / (r M + J / r), M = body + wheels, J = both motors' output inertia + wheels.
# Best radius r* = sqrt(J / M). Body 0.55 kg (Est.) and wheels 0.078 kg / 6.0e-5 kg m^2 are the values of section 2.
p = models["Large (45602)"]
M = 0.55 + 0.078                # body (Est.) + two 39 g wheels with axles (weighed)
for scale in (0.5, 1.0, 1.5):
    J = 2 * p["J"] * scale + 6.0e-5
    cells = ", ".join(f"{name.split(' ')[0]} mm {1 / (r * M + J / r):.1f}" for name, r in WHEELS.items())
    print(f"motor inertia x{scale}: best radius {1000 * math.sqrt(J / M):.0f} mm (diameter "
          f"{2000 * math.sqrt(J / M):.0f}); m/s^2 per N m: {cells}")
w0 = (7.2 - p["R"] * p["tau_s"] / p["kt"]) / p["ke"]
for name, r in WHEELS.items():
    left = ", ".join(f"{v} m/s {max(0.0, 1 - v / r / w0):.0%}" for v in (0.2, 0.3, 0.4))
    print(f"{name}: share of stall torque left at 7.2 V: {left}")

# --------------------------------------------------------------- sensors ---
heading("Sensor noise at the control rate (DATASHEET 5.4)")
GYRO_ND = 0.015            # dps/sqrt(Hz), ICM-20948 Table 1
ACC_ND = 230e-6            # g/sqrt(Hz), ICM-20948 Table 2
GYRO_NBW = {"DLPFCFG=2 (119.5 Hz)": 154.3, "DLPFCFG=3 (51.2 Hz)": 73.3, "DLPFCFG=4 (23.9 Hz)": 35.9}
for k, nbw in GYRO_NBW.items():
    s = GYRO_ND * math.sqrt(nbw)
    print(f"gyro {k}: sigma = {s:.3f} dps = {math.radians(s):.2e} rad/s")
acc = ACC_ND * math.sqrt(34.4)     # ACCEL_DLPFCFG=3, NBW 34.4 Hz
print(f"accelerometer tilt, raw, ACCEL_DLPFCFG=3 (NBW 34.4 Hz): {acc:.2e} rad")
for tau in (0.5, 1.0):
    enbw = math.pi / 2 / (2 * math.pi * tau)
    print(f"complementary filter tau = {tau} s: accelerometer share of pitch noise {ACC_ND * math.sqrt(enbw):.2e} rad")
print(f"board-level zero-g offset 50 mg uncalibrated: {math.atan(0.050):.3f} rad of tilt; "
      f"gyro ZRO +/-5 dps = {math.radians(5):.3f} rad/s")
for wname, r in WHEELS.items():
    q = 2 * math.pi * r / 360
    print(f"encoder, wheel {wname}: 1 count = {1000 * q:.3f} mm, sigma_x = {1000 * q / math.sqrt(12):.3f} mm; "
          f"+/-3 deg accuracy (uniform) sigma = {1000 * math.radians(3) / math.sqrt(3) * r:.2f} mm; "
          f"velocity from counts 10 ms apart sigma = {q / 0.010 / math.sqrt(6):.3f} m/s; "
          f"(5 ms apart: {q / 0.005 / math.sqrt(6):.3f})")

# --------------------------------------------------------------- latency ---
heading("Serial link (DATASHEET 6)")
BAUD = 115200
for cmd in ("port 0 ; set 0.421 ; port 1 ; set -0.421", "po 0;set .421;po 1;set -.421"):
    n = len(cmd) + 1   # + carriage return
    print(f"{cmd!r}: {n} chars = {1000 * n * 10 / BAUD:.2f} ms at 115200 8N1")
line = "P0C0: +123 -4567 -170"
print(f"one combi data line ~{len(line) + 2} chars = {1000 * (len(line) + 2) * 10 / BAUD:.2f} ms; "
      f"two motors at 100 Hz use {100 * 2 * (len(line) + 2) * 10 / BAUD:.0%} of the return direction")
