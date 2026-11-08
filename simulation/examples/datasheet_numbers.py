"""Reproduces every derived number in DATASHEET.md (motor constants, friction, rotor and wheel
inertia, the reference body builds, sensor noise at the control rate, local gravity).

Inputs are the datasheet values and labelled estimates quoted in DATASHEET.md; nothing here is
read from config.py. Run: python simulation/examples/datasheet_numbers.py
"""
import math

KGCM = 0.0980665
RPM = 2 * math.pi / 60

print("=== Pololu 25D bare motors (rated V) ===")
motors = {  # V, free rpm (bare), I_free (gearmotor, no enc), I_stall, stall kg.cm (bare), stall kg.cm @20.4
    "6V HP": (6.0, 10000, 0.42, 6.0, 0.3, 4.8),
    "6V LP": (6.0, 6200, 0.10, 2.0, 0.15, 2.5),
    "12V HP": (12.0, 10000, 0.25, 5.0, 0.4, 7.4),
    "12V MP": (12.0, 8200, 0.08, 1.8, 0.17, 3.2),
}
N = 20.4
for name, (V, rpm, i0, istall, tb, t20) in motors.items():
    R = V / istall
    ke = (V - i0 * R) / (rpm * RPM)
    kt = tb * KGCM / istall
    eta = t20 / (tb * N)
    for Vs in (7.6, 8.4, 6.8):
        w_nl = (Vs - i0 * R) / ke / N
    w_nl76 = (7.6 - i0 * (R + 0.25)) / ke / N
    tq_per_A_out = t20 * KGCM / istall   # output torque per amp incl. gearbox
    tau_f_motor = kt * i0                # friction torque at motor shaft at free run
    tau_f_out = tau_f_motor * N         # per motor, output side (gearbox loss at no load included)
    print(f"{name}: R={R:.2f} ohm, ke={ke*1e3:.2f} mV s/rad, kt(stall)={kt*1e3:.2f} mN m/A, "
          f"eta_stall={eta:.2f}, no-load out @7.6V(+0.25ohm)={w_nl76:.1f} rad/s "
          f"({w_nl76*0.035:.2f} m/s @r=35mm), out torque/A={tq_per_A_out*1e3:.1f} mN m/A, "
          f"free-run friction/motor out={tau_f_out*1e3:.1f} mN m at {rpm/N*RPM:.0f} rad/s")

print("\n=== 6V HP 20.4:1 on 2S, chosen design point ===")
V, rpm, i0, istall, tb, t20 = motors["6V HP"]
R_m = 1.0
ke = (V - i0 * R_m) / (rpm * RPM)
kt_stall = tb * KGCM / istall
eta = t20 / (tb * N)
R_drv = 0.2   # estimate: driver HS+LS (see datasheet text)
R_wire = 0.05
R_tot = R_m + R_drv + R_wire
for Vs in (6.8, 7.6, 8.4):
    print(f"Vs={Vs}: stall current (no limit) {Vs/R_tot:.1f} A, no-load out {(Vs - i0*R_tot)/ke/N:.1f} rad/s")
for I in (1.5, 2.0, 2.5, 3.0):
    print(f"I={I} A/motor: total out torque ideal (kt*N*I*2) {2*ke*N*I:.3f}, with eta {2*ke*N*I*eta:.3f} N m")
# Friction from free-run current: split into Coulomb + viscous at the free-run output speed
w0_out = rpm / N * RPM  # rad/s at 6 V
T0_total = 2 * ke * i0 * N   # both motors, output-referred, N m (ke as torque constant)
print(f"free-run output speed {w0_out:.1f} rad/s; total free-run friction (2 motors, output) {T0_total:.4f} N m")
for alpha in (0.3, 0.5, 0.8):
    print(f"  coulomb fraction {alpha}: tau_c={alpha*T0_total:.4f} N m, b={(1-alpha)*T0_total/w0_out:.5f} N m s/rad")
print(f"  if all viscous: b={T0_total/w0_out:.5f}; if all Coulomb: tau_c={T0_total:.4f}")
c_m = 2 * N**2 * ke**2 / R_tot
print(f"voltage-mode back-EMF drag c_m = n N^2 k^2/R = {c_m:.4f} N m s/rad (sim value with k=0.005,R=3: {2*20**2*0.005**2/3:.4f})")

# Rotor inertia estimate: armature lamination stack + winding as a solid cylinder
for m_arm, r_arm in ((0.010, 0.0080), (0.015, 0.0085), (0.020, 0.0090)):
    Jr = 0.5 * m_arm * r_arm**2
    print(f"armature {m_arm*1e3:.0f} g r={r_arm*1e3:.1f} mm -> Jr={Jr:.2e} kg m^2, "
          f"tau_m = Jr R/(k^2) = {Jr*R_m/ke/kt_stall*1e3:.1f} ms, reflected n N^2 Jr = {2*N**2*Jr:.2e}")

print("\n=== Wheels ===")
for name, m, r in (("Pololu 70x8 (1425)", 0.014, 0.035), ("Pololu 80x10 multi-hub (3690)", 0.023, 0.040)):
    disk = 0.5 * m * r**2
    ring = m * r**2
    est = 0.65 * m * r**2
    print(f"{name}: per wheel disk {disk:.2e}, ring {ring:.2e}, est(0.65 m r^2) {est:.2e}; pair est {2*est:.2e}")
hub = 0.0032
print(f"hub 1081 (3.2 g, r 9.5 mm) inertia each {0.5*hub*0.0095**2:.1e}")

print("\n=== Reference body build ===")
# (name, mass kg, x m (fore-aft from axle), z m (above axle), Ixx_own about pitch axis through own COM)
def box(m, dx, dz):
    return m * (dx**2 + dz**2) / 12
def rod(m, L):
    return m * L**2 / 12
parts = [
    ("2x 25D HP 6V 20.4:1 + encoder (98 g each)", 0.196, 0.0, 0.0, 2 * 0.5 * 0.098 * 0.0125**2),
    ("2x motor bracket + screws (est.)", 0.020, 0.0, 0.0, 0.0),
    ("Bottom deck, 3 mm acrylic 120x70 mm", 0.030, 0.0, 0.020, box(0.030, 0.070, 0.003)),
    ("2x motor driver carrier + wiring (est.)", 0.012, 0.0, 0.025, 0.0),
    ("Middle deck, 3 mm acrylic 120x70 mm", 0.030, 0.0, 0.100, box(0.030, 0.070, 0.003)),
    ("MCU + IMU + 5 V regulator + perfboard (est.)", 0.035, 0.0, 0.108, box(0.035, 0.05, 0.01)),
    ("Top deck, 3 mm acrylic 120x70 mm", 0.030, 0.0, 0.180, box(0.030, 0.070, 0.003)),
    ("2S 1300 mAh LiPo (79 g, 69x35x17 mm)", 0.079, 0.0, 0.1895, box(0.079, 0.035, 0.017)),
    ("4x M4 steel rod/standoff 190 mm, at x=+-0.028", 0.056, 0.0, 0.085, rod(0.056, 0.19) + 0.056 * 0.028**2),
    ("Wiring, switch, connectors, fasteners (est.)", 0.040, 0.0, 0.080, box(0.040, 0.06, 0.15)),
]
M = sum(p[1] for p in parts)
zc = sum(p[1] * p[3] for p in parts) / M
xc = sum(p[1] * p[2] for p in parts) / M
I = sum(p[4] + p[1] * ((p[3] - zc)**2 + (p[2] - xc)**2) for p in parts)
for p in parts:
    print(f"  {p[0]:50s} {p[1]*1e3:6.0f} g  z={p[3]*1e3:6.1f} mm")
print(f"  body mass {M:.3f} kg, COM height {zc:.4f} m, I_com {I:.5f} kg m^2, radius of gyration {math.sqrt(I/M):.4f} m")

# Tall build: decks at 20/130/240 mm, 250 mm rods, 2S 2200 mAh (135 g) on top, 60 g printed side panels (est.)
tall = [
    ("2x 25D HP 6V 20.4:1 + encoder", 0.196, 0.0, 0.0, 2 * 0.5 * 0.098 * 0.0125**2),
    ("2x motor bracket + screws (est.)", 0.020, 0.0, 0.0, 0.0),
    ("Bottom deck", 0.030, 0.0, 0.020, box(0.030, 0.070, 0.003)),
    ("2x motor driver carrier + wiring (est.)", 0.012, 0.0, 0.025, 0.0),
    ("Middle deck", 0.030, 0.0, 0.130, box(0.030, 0.070, 0.003)),
    ("MCU + IMU + regulator + perfboard (est.)", 0.035, 0.0, 0.138, box(0.035, 0.05, 0.01)),
    ("Top deck", 0.030, 0.0, 0.240, box(0.030, 0.070, 0.003)),
    ("2S 2200 mAh LiPo (135 g, 103x34x19 mm)", 0.135, 0.0, 0.2505, box(0.135, 0.034, 0.019)),
    ("4x M4 steel rod 250 mm, at x=+-0.028", 0.074, 0.0, 0.115, rod(0.074, 0.25) + 0.074 * 0.028**2),
    ("Wiring, switch, connectors, fasteners (est.)", 0.050, 0.0, 0.100, box(0.050, 0.06, 0.20)),
    ("Printed side panels (est.)", 0.060, 0.0, 0.120, box(0.060, 0.07, 0.24)),
]
Mt = sum(p[1] for p in tall)
zt = sum(p[1] * p[3] for p in tall) / Mt
It = sum(p[4] + p[1] * ((p[3] - zt)**2 + p[2]**2) for p in tall)
print(f"  tall build: body mass {Mt:.3f} kg, COM height {zt:.4f} m, I_com {It:.5f} kg m^2, radius of gyration {math.sqrt(It/Mt):.4f} m")
print(f"pre-2026-11-08 sim: radius of gyration sqrt(0.0033/0.8)={math.sqrt(0.0033/0.8):.4f} m")

print("\n=== Sensors ===")
fs = 200.0
# gyro
for name, nd_dps in (("MPU-6050", 0.005), ("ICM-42688-P", 0.0028)):
    for bw in (42, 98, 188):
        enbw = bw * math.pi / 2
        s = nd_dps * math.sqrt(enbw)
        print(f"{name} gyro BW {bw} Hz: {s:.3f} deg/s = {math.radians(s):.2e} rad/s")
# accel tilt noise, raw and after complementary filter (accel through 1st-order LPF of tau)
for name, nd_ug in (("MPU-6050", 400), ("ICM-42688-P", 70)):
    for bw in (44, 98):
        s = nd_ug * 1e-6 * math.sqrt(bw * math.pi / 2)
        print(f"{name} accel BW {bw} Hz raw tilt noise {s:.2e} rad")
    for tau in (0.5, 1.0):
        fc = 1 / (2 * math.pi * tau)
        s_acc = nd_ug * 1e-6 * math.sqrt(fc * math.pi / 2)
        print(f"  complementary tau {tau}s: accel part {s_acc:.1e} rad")
for name, nd_dps in (("MPU-6050", 0.005), ("ICM-42688-P", 0.0028)):
    nd = math.radians(nd_dps)
    for tau in (0.5, 1.0):
        # gyro white rate noise integrated, high-passed by complementary filter: var = nd^2 * tau / 2 (one-sided density)
        print(f"{name} tau {tau}: gyro ARW part {nd*math.sqrt(tau/2):.1e} rad; bias 0.01 rad/s -> pitch offset {0.01*tau:.3f} rad")
# encoder
cpr_out = 979.62
for r in (0.035, 0.040):
    q = 2 * math.pi * r / cpr_out
    print(f"r={r}: count {q*1e3:.3f} mm, pos std q/sqrt12 {q/math.sqrt(12)*1e3:.3f} mm, "
          f"velocity diff at {fs:.0f} Hz: step {q*fs:.3f} m/s, std {q*fs/math.sqrt(6):.4f} m/s")
for bl_deg in (0.5, 1, 2):
    print(f"backlash {bl_deg} deg at output -> {math.radians(bl_deg)*0.035*1e3:.2f} mm at r=35 mm")

print("\n=== Gravity, West Lafayette IN (40.4237 N, ~187 m) ===")
phi = math.radians(40.4237)
g = 9.780327 * (1 + 0.0053024 * math.sin(phi)**2 - 0.0000058 * math.sin(2 * phi)**2) - 3.086e-6 * 187
print(f"g = {g:.4f} m/s^2")

print("\n=== Servo MG996R ===")
print(f"11 kg cm = {11*KGCM:.3f} N m; 9.4 kg cm = {9.4*KGCM:.3f} N m; 0.15 s/60 -> {math.radians(60)/0.15:.2f} rad/s")
print(f"AS5600 12-bit: {360/4096:.4f} deg/count, std {360/4096/math.sqrt(12):.4f} deg")
