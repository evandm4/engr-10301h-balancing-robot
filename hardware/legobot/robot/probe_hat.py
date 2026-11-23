"""Probe the Build HAT firmware's commands without moving a motor (C1; DATASHEET 3.3).

    cd ~/legobot && ~/legobot-venv/bin/python -m robot.probe_hat
Every duty sent here is 0, so nothing turns.
"""
import time

from .buildhat_serial import BuildHat


def show(hat, cmd, wait=0.4):
    lines = hat.command(cmd, wait)
    print(f">>> {cmd}")
    for line in lines:
        print(f"    {line}")


def main():
    t0 = time.perf_counter()
    hat = BuildHat(log_lines=True)
    print(f"ready in {time.perf_counter() - t0:.2f} s: {hat.version}")
    show(hat, "help", 1.5)
    for cmd in ("vin", "list", "echo 0", "port 0 ; port_plimit 0.3", "port 0 ; pwmparams 0.65 0.01",
                "plimit 0.3", "bias 0.1",
                # abbreviation candidates, all at zero duty
                "po 0 ; pwm ; set 0", "port 0;pwm;set 0", "po 0;pwm;set 0", "p 0;pwm;set 0",
                "po 0;set 0", "po 0;pw;set 0", "po 0;s 0", "port 0 ; coast",
                # never send `on`: the firmware treats it as "pwm ; set 1" (full power)
                "port 0 ; combi 0 1 0 2 0 3 0 ; select 0 ; selrate 10"):
        show(hat, cmd)
    time.sleep(1.0)
    print("data lines in the last second:", hat.counts())
    print("sample data lines:", hat.data_log[-4:])
    hat.close()


if __name__ == "__main__":
    main()
