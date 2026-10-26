# TinyRL on ESP32-S3

Lightweight online reinforcement learning on a microcontroller. This repository contains the Python implementations, the ESP32-S3 firmware, the experimental data and the analysis scripts of the research described in:

> B. M. Oliva Chuquimia, S. B. Mafra, F. A. P. de Figueiredo. **Lightweight Reinforcement Learning for Microcontrollers: A Comparative Evaluation of Streaming Algorithms on ESP32-S3 Hardware.** National Institute of Telecommunications (Inatel), Brazil.

## 1. Project description

Three lightweight online RL agents are compared with a Standard DQN reference in terms of learning performance, persistent algorithmic state, analytical FLOPs per update and, for the lightweight agents, on-device latency on an ESP32-S3 (240 MHz). The agents are first evaluated in Python simulation and then trained directly on the microcontroller, with the environment simulated in firmware.

## 2. Algorithms and environments

| Agent | Representation | Python | ESP32-S3 firmware |
|---|---|---|---|
| **Stream-Q** | Tabular Q(λ) with replacing traces, discretized observations | `research/agents/stream_q.py` | `firmware/stream_q_{gridworld,cartpole}/` |
| **Stream-AC** | Linear softmax actor–critic, TD(0), entropy regularization | `research/agents/stream_ac.py` | `firmware/stream_ac_{gridworld,cartpole}/` |
| **TinyDQN** | `n_in→12→n_actions` MLP, manual backpropagation, 32-transition mini-buffer | `research/agents/tiny_dqn.py` | `firmware/tiny_dqn_{gridworld,cartpole}/` |
| **Standard DQN** (reference) | `n_in→64→64→n_actions` MLP, target network, 10,000-transition replay buffer | `research/agents/baseline_dqn.py` | — |

Environments: a 5×5 **GridWorld** (`research/envs/gridworld.py`), **CartPole-v1** and **MountainCar-v0** (Gymnasium 0.29.1). Python experiments use seeds 0–9 for the lightweight agents and 0–4 for Standard DQN; the ESP32-S3 experiments cover GridWorld and CartPole with seeds 0–9. Hyperparameters are in `research/experiments/config.yaml` and in each sketch's `config.h`.

```
research/   agents/, core/, envs/   algorithms and environments
            experiments/            configuration and experiment runners
            analysis/               scripts that compute the tables and figures
            results/                experimental data and results
firmware/   six Arduino sketches and the raw serial captures
environment/  pinned Python dependencies
tools/      reproduction, integrity check and serial capture
```

## 3. Installation

Python 3.10. Two environments are used: one for the experiments and one for the analysis.

```bash
python3.10 -m venv .venv-exp && .venv-exp/bin/pip install -r environment/requirements-experiments.txt
python3.10 -m venv .venv-ana && .venv-ana/bin/pip install -r environment/requirements-analysis.txt
```

On Windows use `py -3.10 -m venv ...` and `.venv-exp\Scripts\pip` / `.venv-ana\Scripts\python`.

## 4. Running the code

**Tables and figures from the stored data.** The following command copies `research/` to `_reproduction/<timestamp>/`, runs the analysis scripts there and compares the regenerated files with the stored ones; the stored files are not modified.

```bash
.venv-ana/bin/python tools/reproduce_paper_results.py
```

The scripts can also be run individually from `research/` (for example `python analysis/paper_numbers.py`); in that case they overwrite their outputs in `research/results/`.

**Python experiments.** Runners are in `research/experiments/`: `run_single.py` (lightweight agents), `run_baseline.py`, `run_full_comparison.py` and `run_resumable.py` (Standard DQN), `run_ablations.py` (GridWorld ablations), `run_ablations_cartpole.py` and `run_ablations_streamac_beta.py` (CartPole ablations), and `run_x3_streamac_mountaincar.py` (Stream-AC on MountainCar). By default they write into `research/results/`, so new runs should be written to a separate folder, for example:

```bash
cd research
../.venv-exp/bin/python -c "import sys,os,yaml; sys.path.insert(0,os.getcwd()); from experiments.run_single import run, CONFIG_PATH; cfg=yaml.safe_load(open(CONFIG_PATH)); run('stream_q','cartpole',0,cfg['envs']['cartpole']['episodes'],cfg,out_dir='../_new_runs')"
```

The CartPole ablation runners first require the reference checks of `run_controls.py` (C0, C2) and `run_ablations_cartpole.py --stage control` (C1), which compare new runs with `results/raw_final/`.

**ESP32-S3 firmware.** Each folder in `firmware/` is a complete Arduino sketch (agent, environment and `config.h` with hyperparameters and seed). The sketches compile with the following configuration:

* Arduino IDE 2.x or `arduino-cli` with the Arduino-ESP32 core 3.2.0
* Board *ESP32S3 Dev Module*, CPU frequency 240 MHz, *Arduino Runs On* = Core 1, *Events Run On* = Core 1, *USB CDC On Boot* = Enabled

```bash
arduino-cli core install esp32:esp32@3.2.0
arduino-cli compile -b esp32:esp32:esp32s3:CPUFreq=240,LoopCore=1,EventsCore=1,CDCOnBoot=cdc firmware/stream_q_cartpole
arduino-cli upload  -b esp32:esp32:esp32s3:CPUFreq=240,LoopCore=1,EventsCore=1,CDCOnBoot=cdc -p <PORT> firmware/stream_q_cartpole
```

To run a seed, set `#define SEED` in `config.h`, build and flash, and read the serial output at 115200 baud. Each episode is printed as `agent,env,seed,episode,reward,steps,memory_bytes,flops_per_update,wall_time_s`, where `wall_time_s` is measured around the complete episode. `tools/serial_logger.py --port <PORT> --out <file.csv>` saves the output to a CSV file. In `stream_ac_gridworld/`, the agent is implemented in the `.ino` file; the additional header files in that folder are not used by the sketch.

## 5. Available results

| Article item | Analysis script | Data | Output |
|---|---|---|---|
| Tables 3–5 | `consolidated_final.py` | `raw_final/`; Stream-AC/MountainCar: `derived/ablations_mountaincar_2000/stream_ac_documented/` | `derived/consolidated_final/summary_table_all_envs.csv` |
| Tables 6, 7 and 9 (GridWorld) | `paper_numbers.py` (sections 2–5) | `raw_final/`, `raw_hardware/`, `ablations/` | `raw_summaries/paper_numbers.md`, `paper_numbers_tost.csv` |
| Table 8 and Table 9 (CartPole) | `ablation_report_x1_x2.py`, `write_report_x1_x2.py` | `derived/ablations_cartpole_1500/` | `derived/x1_x2_report/` |
| Figure 1 | `flops_vs_timing_fig.py` | `raw_hardware/` | `figures/flops_vs_timing.png` |
| Figure 2 | `roofline_analysis.py` | `raw_hardware/` | `figures/roofline_model.png`, `raw_summaries/roofline_data.csv` |

Paths are relative to `research/analysis/` and `research/results/`. Data folders:

* `raw_final/`: Python runs, one CSV per agent/environment/seed (one row per episode).
* `raw_hardware/`: ESP32-S3 runs, extracted from the serial captures `firmware/CARTPOLE1.txt` and `firmware/GRIDWORLD1.txt`.
  The original serial captures are kept unmodified and contain some start-up messages in Spanish.
* `ablations/`: GridWorld ablations (5 seeds); `derived/ablations_cartpole_1500/`: CartPole ablations (10 seeds).

Notes on the data:

* **Stream-AC on MountainCar.** Table 5 uses the runs in `derived/ablations_mountaincar_2000/stream_ac_documented/`, obtained with the documented configuration. The older files `raw_final/stream_ac_mountaincar_*.csv` could not be regenerated with that configuration and are not used for the article's results; they remain because `paper_numbers.py` reads them in its section 1, which is not used for Tables 3–5.
* Also kept but not used for the main results: `raw_final/baseline_dqn_cartpole_*_b.csv` (300-episode runs, quoted only in the footnote of Table 4) and `ablations/ablation_bins.csv` (superseded by the 10-seed study of Table 8).
* **FLOPs column.** TinyDQN `flops_per_update` was corrected from 576 to 1152 in the CSV files (two training steps per environment step); the serial captures show the value printed at run time (576). Standard DQN FLOPs are stored with the original count and corrected in `paper_numbers.py` (Table 6).

Reproducibility notes:

* Exact, episode-by-episode reproduction of the Python runs was verified on the original environment (Windows, Python 3.10.0, numpy 2.2.6, gymnasium 0.29.1). On other platforms small floating-point differences can occur; regenerated figures may differ in font anti-aliasing.
* On the ESP32-S3, `randomSeed(0)` does not seed the software generator, so the seed-0 hardware runs cannot be regenerated; seeds ≥ 1 are deterministic. The seed-0 runs also show higher per-step times than seeds 1–9 in some configurations (notably Stream-Q and TinyDQN in GridWorld); Table 7 includes all ten seeds.
* Measured latencies depend on the Arduino-ESP32 core version, compiler settings and board.
* The outputs of the hyperparameter search and of the random-policy MountainCar diagnostic mentioned in the article are not included.

`MANIFEST_SHA256.csv` lists the SHA-256 of every file (`python tools/verify_manifest.py`). The included `.gitattributes` keeps line endings unchanged so that the hashes remain valid after a git checkout.

## 6. Citation and license

Please cite the article above. Citation metadata, an archival DOI and the license will be added when the repository is published.
