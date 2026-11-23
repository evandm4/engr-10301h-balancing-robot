"""Build HAT driver over raw serial (SOFTWARE_PLAN C1, DATASHEET 3.3 and 3.5).

The `buildhat` library is used only to load the firmware, in a short separate process. After that
this module talks to /dev/serial0 directly so the timing of every command is ours:

    hat = BuildHat()                 # opens, checks `version`, loads firmware if needed
    hat.configure(plimit=0.3)        # echo off, port_plimit, pwmparams, combi 1 2 3 at 100 Hz
    hat.set_pwm(0.2, 0.2)            # forward duty for left and right (signs applied here)
    m = hat.latest(LEFT_PORT)        # MotorReading(t, speed, pos, apos), forward-positive
    hat.close()                      # coasts both motors

A background thread timestamps every line with time.perf_counter at the read that completed it.
Coasting on exit, exception, SIGINT and SIGTERM is installed by `install_safety`; run_safe.sh
covers a process that dies without running Python's cleanup (kill -9).
"""
import atexit
import fcntl
import signal
import subprocess
import sys
import threading
import time
from collections import deque, namedtuple

import serial

from . import hw_config as cfg

MotorReading = namedtuple("MotorReading", "t speed pos apos")   # s (perf_counter), %, deg, deg
MOTOR_PORTS = (cfg.LEFT_PORT, cfg.RIGHT_PORT)
TIOCSERGETLSR = 0x5459          # Linux ioctl: line status; bit 0 = transmitter empty


class HatNotReady(RuntimeError):
    pass


def load_firmware(timeout=60.0):
    """Load the firmware with the buildhat library in its own process (DATASHEET 3.5)."""
    subprocess.run([sys.executable, "-c", "from buildhat import Hat; Hat()"],
                   check=True, timeout=timeout)


def _fmt(x):
    """Shortest decimal for a duty in [-1, 1]: 0.421 -> '.421', -0.4 -> '-.4', 0 -> '0'."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    if s in ("", "-", "-0"):
        return "0"
    return s.replace("0.", ".", 1) if s.lstrip("-").startswith("0.") else s


class BuildHat:
    def __init__(self, port=cfg.SERIAL_PORT, load_if_needed=True, log_lines=False):
        self._port_name = port
        self.ser = None
        self._lock = threading.Lock()          # serialises writes
        self._latest = {p: None for p in MOTOR_PORTS}
        self._pair = {p: (None, None) for p in MOTOR_PORTS}   # (previous, latest), replaced in one assignment
        self._line_times = {p: deque(maxlen=512) for p in MOTOR_PORTS}   # arrival times, for control.LineClock
        self._counts = {p: 0 for p in MOTOR_PORTS}
        self.messages = deque(maxlen=200)      # (t, text) for every non-data line
        self.data_log = [] if log_lines else None   # (t, port, speed, pos, apos) raw HAT signs
        self.write_log = []                    # (t0, t_written, t_drained, nbytes, age_l, age_r)
        self.fault = None                      # set on "Motor power fault"
        self.vin_reading = None                # (t, volts) of the newest reply to `vin`
        self._stop = threading.Event()
        self._reader = None

        self._open()
        reply = self._version()
        if not reply.startswith("Firmware version:"):
            if not load_if_needed:
                raise HatNotReady(f"HAT replied {reply!r}")
            self.ser.close()
            load_firmware()
            self._open()
            reply = self._version()
            if not reply.startswith("Firmware version:"):
                raise HatNotReady(f"HAT still not running firmware after init: {reply!r}")
        self.version = reply
        self._reader = threading.Thread(target=self._read_loop, name="hat-reader", daemon=True)
        self._reader.start()

    # ---- setup ----
    def _open(self):
        self.ser = serial.Serial(self._port_name, cfg.BAUD, timeout=0.1, write_timeout=0.5)
        self.ser.reset_input_buffer()

    def _version(self):
        """Stop any streams, then send `version` and return the first line that answers it."""
        self.ser.write(b"port 0 ; select ; port 1 ; select ; port 2 ; select ; port 3 ; select\r")
        time.sleep(0.05)
        self.ser.reset_input_buffer()
        self.ser.write(b"version\r")
        end = time.perf_counter() + 1.0
        while time.perf_counter() < end:
            line = self.ser.readline().decode("ascii", "replace").strip()
            if "version" in line.lower():
                return line
        return ""

    def command(self, text, wait=0.3):
        """Send one command and return the non-data lines that arrive within `wait` s (for probing)."""
        t0 = time.perf_counter()
        self._write((text + "\r").encode("ascii"))
        time.sleep(wait)
        return [m for t, m in list(self.messages) if t >= t0]

    def configure(self, plimit=cfg.BENCH_PLIMIT, pwmparams=cfg.PWMPARAMS, selrate=cfg.SELRATE_MS):
        """Echo off, power limit, PWM parameters, and a 100 Hz speed/pos/apos stream on both ports."""
        self._write(b"echo 0\r")
        for p in MOTOR_PORTS:
            self._write(f"port {p} ; port_plimit {plimit} ; pwmparams {pwmparams[0]} {pwmparams[1]}"
                        f" ; pwm ; coast\r".encode())
            self._write(f"port {p} ; combi 0 1 0 2 0 3 0 ; select 0 ; selrate {selrate}\r".encode())
        self.plimit = plimit

    # ---- reader thread ----
    def _read_loop(self):
        buf = b""
        ser = self.ser
        while not self._stop.is_set():
            try:
                chunk = ser.read(max(1, ser.in_waiting))
            except (serial.SerialException, OSError, TypeError):
                if self._stop.is_set():
                    return
                raise
            if not chunk:
                continue
            t = time.perf_counter()
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                self._handle(t, raw.decode("ascii", "replace").strip())

    def _handle(self, t, line):
        # Combi data line: "P0C0: <speed> <pos> <apos>"
        if len(line) > 5 and line[0] == "P" and line[2:4] == "C0" and line[1].isdigit():
            p = int(line[1])
            try:
                speed, pos, apos = (int(v) for v in line[5:].split()[:3])
            except ValueError:
                self.messages.append((t, line))
                return
            if p in self._latest:
                s = cfg.PORT_SIGN[p]
                reading = MotorReading(t, s * speed, s * pos, s * apos)
                self._pair[p] = (self._latest[p], reading)
                self._line_times[p].append(t)
                self._latest[p] = reading
                self._counts[p] += 1
                if self.data_log is not None:
                    self.data_log.append((t, p, speed, pos, apos))
            return
        if line:
            self.messages.append((t, line))
            if line.endswith(" V"):            # reply to `vin`, e.g. "7.70 V"
                try:
                    self.vin_reading = (t, float(line.split()[0]))
                except ValueError:
                    pass
            if "fault" in line.lower():
                self.fault = (t, line)

    # ---- I/O used by the control loop ----
    def latest(self, port):
        return self._latest[port]

    def line_times(self, port):
        """Arrival times (perf_counter) of the port's last 512 data lines, oldest first (a copy)."""
        return list(self._line_times[port])

    def pair(self, port):
        """(previous, latest) readings of a port, consistent with each other (either may be None)."""
        return self._pair[port]

    def counts(self):
        return dict(self._counts)

    def reader_alive(self):
        return self._reader is not None and self._reader.is_alive()

    def request_vin(self):
        """Ask for the input voltage without waiting; the reply lands in vin_reading (control loop)."""
        self._write(b"vin\r", drain=False)

    def _write(self, data, drain=True):
        """drain: True = tcdrain (blocks ~10 ms longer than the wire time on this Pi: don't use it
        in the loop); "lsr" = busy-poll the UART's transmitter-empty bit, which timestamps when the
        last byte left (for measuring only); False = return as soon as the kernel has the bytes."""
        with self._lock:
            t0 = time.perf_counter()
            self.ser.write(data)
            t1 = time.perf_counter()
            if drain == "lsr":
                fd = self.ser.fileno()
                buf = bytearray(4)
                while True:
                    fcntl.ioctl(fd, TIOCSERGETLSR, buf)
                    if buf[0] & 1:             # TIOCSER_TEMT
                        break
            elif drain:
                self.ser.flush()
            return t0, t1, time.perf_counter()

    def set_pwm(self, left, right, drain=False):
        """Forward duty for each wheel, clamped to DUTY_CAP; one short command for both motors.

        Logs (t0, t_written, t_done, bytes, age of newest left/right reading at t0); t_done is
        when the last byte left the UART if drain="lsr", else when write() returned."""
        cap = cfg.DUTY_CAP
        dl = max(-cap, min(cap, left)) * cfg.PORT_SIGN[cfg.LEFT_PORT]
        dr = max(-cap, min(cap, right)) * cfg.PORT_SIGN[cfg.RIGHT_PORT]
        msg = (self.PWM_CMD.format(l=cfg.LEFT_PORT, r=cfg.RIGHT_PORT, a=_fmt(dl), b=_fmt(dr))
               + "\r").encode()
        ages = [None if self._latest[p] is None else time.perf_counter() - self._latest[p].t
                for p in MOTOR_PORTS]
        t0, t1, t2 = self._write(msg, drain)
        self.write_log.append((t0, t1, t2, len(msg), ages[0], ages[1]))
        return t2 - t0

    # Both motors in one line; the shortest form the firmware accepts (probe_hat.py, DATASHEET 3.3).
    # The firmware matches abbreviated commands by prefix ("po" = port); `set` alone works because
    # configure() leaves both ports in pwm mode. 27 bytes = 2.37 ms on the wire at 0.15 duty.
    PWM_CMD = "po {l};set {a};po {r};set {b}"

    def coast(self):
        """Both motors coast. Doesn't wait for the UART to drain: tcdrain would block the caller 10-14 ms
        (DATASHEET 6) after the bytes are already on their way, as it did the loop's trip tick."""
        self._write(b"port 0 ; coast ; port 1 ; coast\r", drain=False)

    def emergency_coast(self):
        """Coast without taking the write lock, which a stuck loop or an interrupted write may hold: from the
        watchdog thread and from signal handlers. The leading CR ends any half-written command first."""
        import os
        try:
            os.write(self.ser.fileno(), b"\rport 0 ; coast ; port 1 ; coast\r")
        except (OSError, ValueError, AttributeError, serial.SerialException):
            pass                               # port already closed

    def vin(self):
        lines = self.command("vin", 0.2)
        for line in lines:
            if line.endswith("V"):
                return float(line.split()[0])
        return None

    def close(self):
        """Coast, stop the data streams, and close. Waits at most 0.5 s for the write lock (a stuck write may
        hold it), then coasts with the lock-free write instead."""
        if self.ser is None or not self.ser.is_open:
            return
        try:
            if self._lock.acquire(timeout=0.5):
                self._lock.release()
                self.coast()
                self._write(b"port 0 ; select ; port 1 ; select\r")
            else:
                self.emergency_coast()
        finally:
            self._stop.set()
            if self._reader is not None:
                self._reader.join(0.5)
            self.ser.close()


def install_safety(hat):
    """Coast both motors on normal exit, uncaught exception, SIGINT, SIGTERM and SIGHUP.

    The signal handler coasts with the lock-free write and exits; the full close (lock, stream stop) runs at
    exit, after the exception has unwound out of any `with self._lock`. A handler must never take the write
    lock: Python runs it in the main thread between bytecodes, possibly inside _write, or inside an earlier
    handler that holds the lock, and a threading.Lock taken twice by one thread waits forever. That is how a
    closed SSH terminal hung a run (SIGHUP, then run_safe.sh's SIGTERM 5 ms later; 2026-11-23)."""
    def _coast_quietly():
        try:
            hat.close()
        except Exception:
            pass

    atexit.register(_coast_quietly)
    stopping = []

    def _on_signal(signum, frame):
        if not stopping:                       # a second signal while handling the first: just exit
            stopping.append(signum)
            hat.emergency_coast()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)
