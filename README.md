# GR00T N1.6 × RoboCasa: Closed-Loop VLA Evaluation Pipeline

A fully automated closed-loop Vision-Language-Action (VLA) inference pipeline connecting [RoboCasa365](https://robocasa.ai/) simulation (local) with [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) policy inference (cloud).

---

## Architecture

```
┌─────────────────────────────┐        ngrok TCP Tunnel         ┌──────────────────────────────┐
│     Local (WSL2, RTX 4060)  │ ◄─────────────────────────────► │  Cloud (Google Colab, A100)  │
│                             │                                 │                              │
│  RoboCasa365 Simulation     │     Image Observations (→)      │  GR00T N1.6 Policy Server    │
│  - Renders RGB images       │     Action Commands  (←)        │  - VLM processing inputs     │
│  - Executes robot actions   │                                 │  - Flow matching denoising   │
│  - Franka Panda arm         │                                 │  - ROBOCASA_PANDA_OMRON tag  │
└─────────────────────────────┘                                 └──────────────────────────────┘
```

At each timestep:
1. RoboCasa renders image observations from the simulated Franka Panda arm
2. Observations are sent to the GR00T Policy Server via ngrok TCP tunnel
3. GR00T processes the inputs and returns an action chunk using VLM and flow matching
4. RoboCasa executes the action and advances the simulation

> RoboCasa365 and the official GR00T evaluation scripts provide a fully automated rollout pipeline. The image-action loop runs internally rather than requiring manual teleoperation to collect data.

---

## System Requirements
 
### Local (WSL2)
- OS: Windows 10/11 with WSL2 (Ubuntu 22.04 recommended)
- GPU: NVIDIA GPU with WSL2 passthrough (tested: RTX 4060, 8GB VRAM)
- Disk: **≥ 30GB free** on the drive hosting the WSL `.vhdx` file (kitchen assets ~5GB, dependencies ~10GB)
- RAM: ≥ 8GB (tested with 8GB)
- Software: Conda, Git, `uv`
 
### Cloud (Google Colab)
- Compute: A100 GPU Runtime (Colab's A100 provides 40GB VRAM, satisfying the [official ≥24GB requirement](https://www.mintlify.com/NVIDIA/Isaac-GR00T/getting-started/hardware-requirements) for GR00T N1.6-3B)
- Network: [ngrok](https://ngrok.com) account (required to establish the TCP tunnel)

---

## Part 1: Cloud Setup — GR00T Policy Server (Google Colab)

Run the following cells in a new Colab notebook with GPU runtime enabled.

### Cell 1 — Verify GPU
```python
!nvidia-smi
```

### Cell 2 — Install `uv` package manager
```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
```

### Cell 3 — Clone Isaac-GR00T (with submodules)
```python
!git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
import os
os.chdir('Isaac-GR00T')
```

### Cell 4 — Install dependencies
```bash
!bash scripts/deployment/dgpu/install_deps.sh
```

### Cell 5 — Fix matplotlib backend conflict
Colab sets `MPLBACKEND` to `matplotlib_inline.backend_inline`, which is not recognized by GR00T's `.venv`. Override before any inference:
```python
import os
os.environ['MPLBACKEND'] = 'Agg'
```

### Cell 6 — Install flash-attn
Must be installed into GR00T's `.venv`, not the system Python:
```bash
!.venv/bin/python -m pip install flash-attn==2.8.3 --no-build-isolation
```
Verify:
```bash
!.venv/bin/python -c "import flash_attn; print(flash_attn.__version__)"
```

### Cell 7 — (Optional) Verify GR00T inference works

This step tests that the model loads and runs correctly using the provided demo data. Uses the `GR1` embodiment as a quick sanity check, **not** the final evaluation pipeline.

```python
import os
os.chdir('/content/Isaac-GR00T')
os.environ['MPLBACKEND'] = 'Agg'

!MPLBACKEND=Agg .venv/bin/python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --dataset-path demo_data/gr1.PickNPlace \
    --embodiment-tag GR1 \
    --traj-ids 0 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

> Expected Output: MSE loss values for each trajectory.  
> The model weights (~6.5GB) are downloaded automatically from Hugging Face on first run. The first step prints `AUTOTUNE` logs, which is normal and subsequent steps will be clean.  

---

### Every Session: Start the Policy Server

**Step 1 — Launch ngrok TCP tunnel** 
```python
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # Register free at https://ngrok.com
tunnel = ngrok.connect(5555, "tcp")
print(tunnel.public_url)  # e.g. tcp://4.tcp.ngrok.io:16704 — note this down
```

**Step 2 — Start GR00T Policy Server** (runs continuously)
```python
import os
os.chdir('/content/Isaac-GR00T')
os.environ['MPLBACKEND'] = 'Agg'

!MPLBACKEND=Agg .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --model-path nvidia/GR00T-N1.6-3B \
    --use-sim-policy-wrapper \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555
```
Wait for: `Server is ready and listening on tcp://0.0.0.0:5555`

> --`use-sim-policy-wrapper` is required for RoboCasa evaluation.  
> --`embodiment-tag ROBOCASA_PANDA_OMRON` targets the Franka Panda + Omron gripper configuration used in RoboCasa365.  

---

## Part 2: Local Setup — RoboCasa365 Simulation (WSL2)

### Prerequisites

Install `uv` package manager (required by the setup script):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart terminal
```

Install system-level dependencies required for MuJoCo rendering and Python compilation:
```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa python3.10-dev
```

### Step 1 — Clone Isaac-GR00T
```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

### Step 2 — Run the official RoboCasa setup script
This script handles everything: creates an isolated `.venv`, installs RoboCasa365 (via Git submodule at `external_dependencies/robocasa`), installs all dependencies, and downloads ~5GB of kitchen assets automatically.

```bash
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

> This is the **official installation path** and it creates a self-contained environment specifically tuned for the GR00T × RoboCasa evaluation pipeline.  

The script will:
- Create `.venv` at `gr00t/eval/sim/robocasa/robocasa_uv/.venv`
- Install PyTorch 2.5.1, robosuite, robocasa, gymnasium, and other dependencies
- Download kitchen textures, fixtures, and Objaverse objects (~5GB total)
- Run a sanity check confirming `Env OK`

### Step 3 — Verify installation
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python -c "
import gymnasium as gym
import robocasa.utils.gym_utils.gymnasium_groot
envs = [e for e in gym.envs.registry.keys() if 'PandaOmron' in e]
print(f'Available tasks: {len(envs)}')
print(envs[:5])
"
```
Expected output: `Available tasks: 140+`

### (Optional) Visual verification — render a kitchen scene
To confirm environment rendering works correctly:
```bash
cd ~/workspace/Isaac-GR00T/external_dependencies/robocasa
~/workspace/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python -m robocasa.demos.demo_kitchen_scenes
```
A window will pop up showing a kitchen scene. This is also useful to check that WSL2 GPU passthrough and display are working correctly.

## Part 3: Running Evaluations

### Standard Evaluation
Update `NGROK_HOST` and `NGROK_PORT` from the Colab output, then run:

```bash
cd ~/workspace/Isaac-GR00T

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 3 \
    --policy_client_host <NGROK_HOST> \
    --policy_client_port <NGROK_PORT> \
    --max_episode_steps 720 \
    --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 1
```

### Key Parameters

| Parameter | Description | Default | Notes |
|---|---|---|---|
| `--n_episodes` | Number of episodes to run | 3 | Each episode is randomly initialized |
| `--max_episode_steps` | Max steps per episode | 720 | Increase for complex tasks |
| `--n_action_steps` | Action chunk size | 8 |**Trade-off:** Lower values enable action correction; larger values produce smoother continuous actions |
| `--env_name` | Task environment | - | See task list below |
| `--n_envs` | Parallel environments | 1 | Keep at 1 for remote server |
| `--render_mode` | Visualization | - | Use `human` for real-time GUI; omit for faster headless benchmarking. |

---

## Part 4: Benchmark Results

Tested with GR00T N1.6-3B, Franka Panda, Google Colab A100, ngrok TCP tunnel.

| Task | Type | n_action_steps=8 |Time/episode | n_action_steps=2 | Time/episode |
|---|---|---|---|---|---|
| `OpenDrawer_PandaOmron_Env` | Single-step, fixed target | **3/3 (100%)** | ~5 min | - | - |
| `CoffeeServeMug_PandaOmron_Env` | Pick-and-place | 0/3 (0%) | ~12min | **3/3 (100%)** | ~16 min |
| `StackBowlsInSink_PandaOmron_Env` | Multi-step, precise | 0/3 (0%) | ~12min | 0/3 (0%) | ~36 min |

**Key observations:**
- GR00T performs well on single-step, spatially fixed tasks (OpenDrawer: 100%)
- Pick-and-place tasks benefit significantly from smaller action chunks (n=2)
- Multi-step tasks requiring sequential grasping remain challenging for zero-shot.

---
 
## Task List (Sample)
 
```bash
# List all available Franka Panda tasks
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python -c "
import gymnasium as gym
import robocasa.utils.gym_utils.gymnasium_groot
for e in sorted(gym.envs.registry.keys()):
    if 'PandaOmron' in e:
        print(e)
"
```
 
Selected tasks of interest:
 
| Task | Description |
|---|---|
| `OpenDrawer_PandaOmron_Env` | Open a kitchen drawer |
| `PnPCounterToCab_PandaOmron_Env` | Pick from counter, place in cabinet |
| `CoffeeServeMug_PandaOmron_Env` | Move mug from coffee machine to counter |
| `TurnOnSinkFaucet_PandaOmron_Env` | Turn on sink faucet |
| `StackBowlsInSink_PandaOmron_Env` | Stack bowls in sink (multi-step) |
| `PrepareCoffee_PandaOmron_Env` | Multi-step coffee preparation |
 
---
 
## Troubleshooting
 
| Issue | Cause | Fix |
|---|---|---|
| `ValueError: Key backend: 'module://matplotlib_inline...'` | Colab MPLBACKEND pollution | Add `os.environ['MPLBACKEND'] = 'Agg'` before running |
| `ImportError: flash_attn not installed` | Installed to system Python, not `.venv` | Use `.venv/bin/python -m pip install flash-attn` |
| `evdev build error: Python.h not found` | Missing Python dev headers in WSL | `sudo apt install python3.10-dev` |
| Rollout hangs at `Episodes: 0%` | Waiting for server connection | Verify ngrok address and that Colab shows `Server is ready` |
| `ping: True` but no Colab output | Normal — server doesn't log ping requests | Connection is fine, rollout is running |
| `CUDA_HOME not set` (flash-attn on WSL) | No CUDA toolkit in WSL | Expected — flash-attn is not needed locally, skip |
| `FlashAttention only supports Ampere GPUs or newer` | Colab assigned T4/V100 instead of A100 | Switch to A100 runtime, or uninstall flash-attn (results are same without it) |
| ngrok `connection refused` warnings during model download | Server not ready yet, ngrok retrying | Normal — wait for `Server is ready` message before connecting |
 
---
 
## References
 
- [RoboCasa365](https://robocasa.ai/) — kitchen simulation framework
- [Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA foundation model for generalist robots
- [GR00T RoboCasa Eval Guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa/README.md) — primary reference for this pipeline setup
- [GR00T Hardware Requirements](https://www.mintlify.com/NVIDIA/Isaac-GR00T/getting-started/hardware-requirements) — hardware guide
