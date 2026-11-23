I'm starting the software side of moving my balancing-robot simulation to a physical LEGO build. A separate session owns the physical robot design; this session owns the software. The goal is Phase A of `hardware/legobot/SOFTWARE_PLAN.md`, then a stop for my review.

**Read first, in this order:**
1. `hardware/legobot/SOFTWARE_PLAN.md`: the full plan (phases A–E, dependencies, recommended decisions).
2. `hardware/legobot/DATASHEET.md`, sections 1, 2 and 7 only: the LEGO robot's proposed config values with their spreads, and where the current model doesn't match it. Open other sections only when a task needs them. `legobot_numbers.py` prints every derived number.
3. `config.py` (how `ProjectConfig` is built) and `pivot_config.py` (the pattern for a second config that reuses `CONFIG`).
4. The `ScenarioConfig` and `CostConfig` docstrings in `config.py`: the method A3 must repeat for the LEGO motors.

**Decided (don't reopen):**
- Hardware: 2× LEGO SPIKE Large Angular Motor driving 69 mm wheels directly (39 g each with axle). Raspberry Pi 3B, Raspberry Pi Build HAT, SparkFun Auto pHAT (ICM-20948 IMU), 2S LiPo.
- The LEGO robot becomes the reference build and goes into the paper. `config.py`'s Pololu robot stays intact, and both configs are kept.
- The motors are modeled as lumped output-side motors (`gear_ratio = 1`; DATASHEET 2.2).

**Phase A tasks** (SOFTWARE_PLAN 3, Phase A):
- **A1:** `legobot_config.py` with DATASHEET section 2's values and spreads.
- **A2:** robot selection. Recommended: `--robot` flag plus a `ROBOT` environment variable read in `config.py`, defaulting to pololu so every existing command still works. Results go under `results/<robot>/`.
- **A3:** re-derive the scenario, cost limits and fall-penalty bound from the LEGO motors. No-load top speed is 0.54–0.67 m/s, against the current 1.3 m/s cruise.
- **A4:** LEGO baselines.

Stop after A3 and show me the derived scenario before tuning anything.

**Working rules:**
- Read the course rules first: `CLAUDE.md` in the `Purdue` folder and in `ENGR 10301H` (if present). Also follow my saved memories: quick runs while developing and full rigor only for final runs; the seed tiers, where the frozen validation set is touched once at the end; and fix methodology gaps rather than only documenting them.
- **Token efficiency is a priority. Don't sacrifice quality for it.**
  - **Opus does the judgment work in this session:** the A3 scenario derivation, actuator and sensor model design (B1, B2), and reviewing subagent output.
  - **Sonnet subagents (`model: "sonnet"`) for well-specified mechanical work:**
    - code surveys ("find every script that imports CONFIG");
    - adding `--robot` plumbing across scripts once the pattern is set;
    - running pytest and summarizing failures;
    - porting a reference-engine change to `batch_sim` / `torch_sim` against a written spec, with the equivalence tests as the gate;
    - doc and ARCHITECTURE.md updates;
    - drafting the DATASHEET section 8 measurement scripts.
  - Give each subagent a complete brief: files, exact change, how it's verified.
  - Check every subagent result yourself before reporting it.
  - Bulk read-only summaries can go to my local model (see my global `CLAUDE.md`).
  - Don't re-read whole large files (PHYSICS.md, DATASHEET.md); grep for the section you need.
- Every physics or sensor change must go into all three engines (`simulation/balance_sim`, `simulation/batch_sim`, `batch_sim/torch_sim.py`). The NumPy batched engine must stay bit-identical to the reference (`simulation/tests/test_equivalence.py`). Phase A shouldn't need engine changes.
- Git:
  - Current branch is `lqr-integral-and-latency`. It has my own uncommitted edits (ARCHITECTURE.md, the progress deck, `paper/methods.docx`, docs). Leave those alone.
  - The `hardware/legobot/` files are uncommitted too.
  - Ask before committing, and branch first (`legobot-config`).
