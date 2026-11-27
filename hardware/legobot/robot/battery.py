"""Report the battery voltage the Build HAT sees (`vin`). Never moves a motor.

    cd ~/legobot && ~/legobot-venv/bin/python -m robot.battery            # one reading
    cd ~/legobot && ~/legobot-venv/bin/python -m robot.battery --watch    # every 2 s, Ctrl-C to stop
    cd ~/legobot && ~/legobot-venv/bin/python -m robot.battery -n 5 --period 1

Stop anything else using the HAT first (balance loop, measure scripts): only one process can own /dev/serial0.
Under load the reading sags; read it with the motors idle for the resting voltage. 2S LiPo: 8.4 V full, 7.4-7.6 V
nominal, the robot's cutoff is 6.8 V (loop_config.VIN_CUTOFF). Also prints the Pi's under-voltage flag if `vcgencmd` exists.
"""
import argparse
import subprocess
import time

from . import loop_config as lc
from .buildhat_serial import BuildHat

# The cutoff is the balancing loop's own (it refuses to start below it and trips under it), so this script's
# "BELOW CUTOFF" matches what balance.py will do. It was a separate 6.6 V that missed the G7 change to 6.8 V.
FULL_V, LOW_V, CUTOFF_V = 8.4, 7.0, lc.VIN_CUTOFF


def read_vin(hat, tries=3):
    for _ in range(tries):
        v = hat.vin()
        if v is not None:
            return v
    return None


def pi_throttled():
    """Text for `vcgencmd get_throttled`, or '' if unavailable."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        flags = int(out.split("=")[1].strip(), 16)
        return f"  Pi throttled={flags:#x}" + ("  (UNDER-VOLTAGE NOW)" if flags & 0x1 else "")
    except Exception:
        return ""


def describe(v):
    if v < CUTOFF_V:
        return "BELOW CUTOFF: charge now"
    if v < LOW_V:
        return "low: charge soon"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--watch", action="store_true", help="keep reading until Ctrl-C")
    ap.add_argument("-n", type=int, default=1, help="number of readings (default 1)")
    ap.add_argument("--period", type=float, default=2.0, help="seconds between readings (default 2)")
    a = ap.parse_args()

    hat = BuildHat()
    try:
        k = 0
        while True:
            v = read_vin(hat)
            if v is None:
                print("no reply to `vin`")
            else:
                print(f"{time.strftime('%H:%M:%S')}  vin {v:.2f} V  {describe(v)}{pi_throttled()}")
            k += 1
            if not a.watch and k >= a.n:
                break
            time.sleep(a.period)
    except KeyboardInterrupt:
        pass
    finally:
        hat.close()


if __name__ == "__main__":
    main()
