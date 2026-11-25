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

## Command hygiene for Antigravity agents

This section is for Antigravity agents working in the Antigravity clone (`C:\balancebot\engr-10301h-balancing-robot`)
and with the shared tooling in `C:\balancebot_training\`. Claude Code sessions use their own clone and permission
settings and can skip it. Antigravity's permission grants match command prefixes, so write each command to start
with an allowlisted prefix; anything else stops for approval.

- **Python:** `powershell -NoProfile -ExecutionPolicy Bypass -File C:\balancebot_training\tools\py.ps1 [-Cpu] [-Robot legobot] [-Threads N] <python args>`.
  It sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (default 4), `ROBOT`, and with `-Cpu` hides the GPU. Never prefix a
  command with `$env:...;` (and `$env:CUDA_VISIBLE_DEVICES=''` deletes the variable rather than hiding the GPU).
- **No chains or prefixes:** one command per call. No `;`/`&&` chains, no `cd ...;`, no `Start-Sleep ...;`
  (use a scheduled wakeup to wait). Run from the repo root.
- **Git:** one git command per call, especially writes and fetches (`git add` and `git commit` are separate calls).
  Omit `git -C <path>` when the cwd is already the checkout. Never `git push`, `git clean` or `git reset --hard`.
- **Desktop:** copy a `.ps1` to `desktop:C:/balancebot/` with `scp`, then run
  `ssh desktop "powershell -NoProfile -ExecutionPolicy Bypass -File C:\balancebot\<script>.ps1"`. No inline
  PowerShell strings over SSH.
- **Pi:** only through `C:\balancebot_training\tools_pi\`: `pi_run.ps1 <script.sh> [args]` (refuses lines that
  could drive the motors unless `-AllowDrive`), `pi_fetch.ps1 <glob> [subdir]` (logs into
  `C:\balancebot_training\legobot\pi_logs\`), `pi_deploy.ps1` (deploys the checkout you are in). Each is invoked as
  `powershell -NoProfile -ExecutionPolicy Bypass -File C:\balancebot_training\tools_pi\<name>.ps1 ...`. Never pipe
  scripts through `Git\bin\bash.exe`. Keep reusable Pi scripts in `C:\balancebot_training\shared\pi\`.
- **Queues and dashboard:** `C:\balancebot_training\tools\run_queue.ps1`, `submit_queue.ps1`, `dashboard.ps1`.
- **Handoffs between agents:** write shared files to `C:\balancebot_training\shared\`, never into or out of another
  conversation's scratch folder.

## Where things are

`README.md` has the layout and the order everything runs in. For the LEGO robot, read
`hardware/legobot/SOFTWARE_PLAN.md` first.
