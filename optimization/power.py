"""Keep a long run at full power, and record whether it was.

keep_awake(): while a run is going, tell Windows the system and the display are in use
(SetThreadExecutionState), so the idle timers that turn the screen off or sleep the machine never
fire, and opt this process out of Windows' background power throttling (EcoQoS). On this laptop
(ASUS ROG Zephyrus G16, G-Helper) power draw drops once the machine sits idle, whatever the
G-Helper mode; these are the idle signals a process can control. Both requests belong to this
process and end with it: no system setting is changed. Elsewhere (Linux, a cluster) it does
nothing.

ghelper_mode(): the G-Helper performance mode (Silent, Balanced, Turbo or a custom one) and its GPU
settings, read from G-Helper's config file. Never changed here: the mode is a noise choice.

gpu_power(): the GPU's enforced power limit, SM clock and power draw from nvidia-smi, or {} when
there is no nvidia-smi. Logged with every generation so a throttled stretch of a run shows up in
its history (and explains a slow wall-clock) instead of going unnoticed.
"""

import contextlib
import shutil
import subprocess
import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002

GPU_FIELDS = ("gpu_power_limit_w", "gpu_power_draw_w", "gpu_sm_clock_mhz")
_QUERY = "enforced.power.limit,power.draw,clocks.sm"


_PROCESS_POWER_THROTTLING = 4                 # PROCESS_INFORMATION_CLASS ProcessPowerThrottling
_THROTTLE_EXECUTION_SPEED = 0x1               # EcoQoS: slower cores and clocks for "background" work
_THROTTLE_IGNORE_TIMER_RESOLUTION = 0x4


def _opt_out_of_power_throttling() -> bool:
    """Tell Windows this process is not background work: no EcoQoS, timer requests honored.

    Windows throttles processes it judges background (no visible window, the user away), which on
    a laptop left alone also slows the CPU thread that feeds the GPU, in any G-Helper mode. This is
    the process's own setting (SetProcessInformation), not a system one; it ends with the process.
    """
    import ctypes

    class State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    mask = _THROTTLE_EXECUTION_SPEED | _THROTTLE_IGNORE_TIMER_RESOLUTION
    state = State(1, mask, 0)                  # control these, and turn them off
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    return bool(kernel32.SetProcessInformation(ctypes.c_void_p(kernel32.GetCurrentProcess()),
                                               _PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                               ctypes.sizeof(state)))


@contextlib.contextmanager
def keep_awake(display: bool = True):
    """While the block runs, keep this run from looking idle to Windows: no sleep, no screen-off
    (with display=True), and no background power throttling (EcoQoS) for this process.
    Yields whether both requests were accepted."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes

    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | (_ES_DISPLAY_REQUIRED if display else 0)
    ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    try:
        ok = _opt_out_of_power_throttling() and ok
    except (AttributeError, OSError):          # Windows before 8 has no SetProcessInformation
        ok = False
    try:
        yield ok
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


_GHELPER_MODES = {0: "Balanced", 1: "Turbo", 2: "Silent"}   # G-Helper's built-in modes; custom ones are named


def ghelper_mode() -> dict:
    """The G-Helper performance mode in use and that mode's GPU settings, read from its config file;
    {} without G-Helper. Read-only: the mode is the user's choice (it trades noise for speed), so a
    run records it rather than changing it. gpu_boost_w is the mode's Dynamic Boost allowance (W)
    and gpu_temp_c its GPU temperature target."""
    import json
    import os
    from pathlib import Path

    path = Path(os.environ.get("APPDATA", "")) / "GHelper" / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    mode = config.get("performance_mode")
    if mode is None:
        return {}
    return {"ghelper_mode": config.get(f"mode_name_{mode}", _GHELPER_MODES.get(mode, str(mode))),
            "ghelper_gpu_boost_w": config.get(f"gpu_boost_{mode}"),
            "ghelper_gpu_temp_c": config.get(f"gpu_temp_{mode}")}


def gpu_power(index: int = 0) -> dict:
    """{gpu_power_limit_w, gpu_power_draw_w, gpu_sm_clock_mhz} for one GPU, or {} if unavailable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    try:
        out = subprocess.run([exe, f"--id={index}", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        values = [float(v) if v.strip() not in ("[N/A]", "N/A", "") else None for v in out.split(",")]
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    return dict(zip(GPU_FIELDS, values)) if len(values) == len(GPU_FIELDS) else {}


POWER_FIELDS = GPU_FIELDS + ("ghelper_mode",)


def power_state() -> dict:
    """gpu_power() plus the G-Helper mode name: what a run logs each generation."""
    state = gpu_power()
    mode = ghelper_mode().get("ghelper_mode")
    if mode is not None:
        state["ghelper_mode"] = mode
    return state
