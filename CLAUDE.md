# ENGR 10301H: balancing-robot simulation (project notes for Claude)

## Python environment

Always use the project venv, **`C:\venvs\balancebot`** (Python 3.12, NumPy, SciPy, cma, PyTorch with CUDA
on the RTX 5070 Ti, pytest). The system Python lacks torch, so the GPU engine's tests silently skip there.

- From the Bash tool: `/c/venvs/balancebot/Scripts/python.exe -m pytest` (or `... python.exe <script>`).
- From PowerShell: `& C:\venvs\balancebot\Scripts\python.exe -m pytest`, or activate first:
  `C:\venvs\balancebot\Scripts\Activate.ps1`.
- Run from the repo root. `pytest.ini` sets the test paths and the import paths.

## Tests

- `python -m pytest` runs the default (Pololu) robot. `ROBOT=legobot python -m pytest` runs the LEGO robot;
  run both after any change to the simulator, the tuner or a config.
- The LEGO robot's own code: `python -m pytest hardware/legobot/robot/test_control.py hardware/legobot/robot/measure/test_fit.py`.
- Every model change must land in all three engines (`simulation/balance_sim`, `simulation/batch_sim`,
  `batch_sim/torch_sim.py`); `test_equivalence.py` and `test_torch.py` hold them to each other.

## Where things are

`README.md` has the layout and the order everything runs in. For the LEGO robot, read
`hardware/legobot/SOFTWARE_PLAN.md` first.
