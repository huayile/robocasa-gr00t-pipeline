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
- Compute: A100 GPU Runtime (40GB VRAM)
  - A100 far meets [official ≥8GB requirement](https://www.mintlify.com/NVIDIA/Isaac-GR00T/getting-started/hardware-requirements) for GR00T N1.6-3B inference only
  - FlashAttention requires Ampere GPUs (e.g., A100); using T4/V100 may cause compatibility issues or require disabling flash-attn

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

## Part 5: Custom Simulation Environment (`fixed_pnp_env.py`)
 
A script for VLA benchmarking with configurable object types and positions. Located at `grasp_experiments/fixed_pnp_env.py`.
 
### Design
 
The environment places object A and B on the kitchen counter next to the sink. The robot must pick object A and place it into object B. Key properties:
 
- **Fixed scene**: `--seed 2` gives a well-tested layout. 
- **Specific objects**: specify exact object names (`banana`, `bowl`) rather than random categories
- **Precise positioning**: object XY positions are set relative to the robot base via MuJoCo joint control, rather than RoboCasa's random placement
- **Official pipeline**: uses `GrootRoboCasaEnv` wrapper and `create_grootrobocasa_env_class` for correct `ROBOCASA_PANDA_OMRON` observation/action format
 
### Coordinate System
 
Positions are specified relative to the robot base:
- `x`: left (-) / right (+)
- `y`: forward distance from robot (positive = away from robot)
- `z`: automatically set from RoboCasa placement (counter surface height)
 
The default configuration places both objects at ~0.41m from the robot base, where both objects are within the robot's reach.
 
### Implementation
To ensure fixed environment and overcome RoboCasa's default randomization without modifying its source code, some overrides were employed:
 
**Registration:** RoboCasa's `create_env_robosuite()` only accepts a fixed set of parameters. The solution is a dynamic subclass with parameters baked in at class definition time:
 
```python
class FixedPnPInstance(FixedPnP):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_a_group="banana", obj_a_x=-0.4, ...)
 
REGISTERED_ENVS["FixedPnP"] = FixedPnPInstance
create_grootrobocasa_env_class("FixedPnP", "PandaOmron", "panda_omron")
```
 
**Layout fixing:** `create_env_robosuite()` hardcodes a list of layouts via `env_kwargs.update(...)`, so passing `layout_and_style_ids` as a kwarg has no effect. The solution is to override `self.layout_and_style_ids` directly after the parent `__init__` completes — this works because `_reset_internal` reads from `self.layout_and_style_ids` at each episode reset:
 
```python
class FixedPnP(Kitchen):
    FIXED_LAYOUT = [(5, 1)]  # U_SHAPED_SMALL + SCANDANAVIAN
 
    def __init__(self, ...):
        super().__init__(*args, **kwargs)
        self.layout_and_style_ids = self.FIXED_LAYOUT  # override after parent sets it
```
 
This cleanly enforces layout [5,1] in both preview and inference modes without modifying any library code. Verify with:
```python
inner_env = env.unwrapped.env
print(inner_env.layout_id, inner_env.style_id)  # should print: 5 1
```
 
**Position control:** After RoboCasa's standard reset, `_reset_internal` overrides XY positions using MuJoCo free joint `qpos`, while preserving each object's Z from the original placement (correct surface height per object):
 
```python
def _reset_internal(self):
    super()._reset_internal()
    base = self.robots[0].base_pos
    jnt_a = self.sim.model.body_jntadr[self.sim.model.body_name2id("obj_a_main")]
    adr_a = self.sim.model.jnt_qposadr[jnt_a]
    z_a = self.sim.data.qpos[adr_a + 2]  # preserve original surface height
    self.sim.data.qpos[adr_a:adr_a+3] = [base[0] + self.obj_a_x, base[1] + self.obj_a_y, z_a]
    self.sim.forward()
```
 
> Keep objects at least 0.15m apart in XY to avoid physics overlap. The default configuration (`obj_a_x=-0.4, obj_b_x=0.4`) has been validated with layout [5,1] and seed=2, where both objects land stably on the counter.
 
### Usage
 
**Preview the scene interactively** (no policy server needed):
```bash
cd ~/workspace/Isaac-GR00T
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --preview --obj-a banana --obj-b bowl --seed 2
```
Controls: mouse drag to rotate, arrow keys to move robot, spacebar to toggle gripper, Ctrl+Q to exit.
 
**Run with GR00T policy server:**
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --host <NGROK_HOST> --port <NGROK_PORT> \
    --obj-a banana --obj-b bowl \
    --seed 2 --n-episodes 3 --n-action-steps 8 --save-video
```
 
**Systematic distance experiment (vary x separation):**
```bash
# Default: objects 0.8m apart (validated with layout [5,1])
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --host <NGROK_HOST> --port <NGROK_PORT> \
    --obj-a-x -0.4 --obj-b-x 0.4 --seed 2 --n-episodes 3
 
# depart: objects 1m apart
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --host <NGROK_HOST> --port <NGROK_PORT> \
    --obj-a-x -0.4 --obj-b-x 0.6 --seed 2 --n-episodes 3
```
 
### Parameters
 
| Parameter | Description | Default |
|---|---|---|
| `--preview` | Launch interactive mjviewer | - |
| `--obj-a` | Object A: specific name (`banana`, `apple`, `carrot`...) or category (`fruit`, `vegetable`...) | `banana` |
| `--obj-b` | Object B: specific name (`bowl`, `plate`, `cup`...) or category (`receptacle`) | `bowl` |
| `--obj-a-x` | Object A x offset from robot base, left(-)/right(+) in meters | `-0.4` |
| `--obj-a-y` | Object A y offset from robot base, forward distance in meters | `0.4` |
| `--obj-b-x` | Object B x offset from robot base, left(-)/right(+) in meters | `0.4` |
| `--obj-b-y` | Object B y offset from robot base, forward distance in meters | `0.4` |
| `--seed` | Seed for object randomization (kitchen layout fixed to [5,1]) | `2` |
| `--n-episodes` | Number of episodes | `3` |
| `--n-action-steps` | Action chunk size | `8` |
| `--save-video` | Save rollout videos to `grasp_experiments/videos/` | - |
 
Available object names (partial list): `banana`, `apple`, `carrot`, `cucumber`, `orange`, `tomato`, `sweet_potato`, `broccoli`, `can`, `bowl`, `plate`, `cup`, `mug`, `pot`, `pan`
 
Videos are saved to `grasp_experiments/videos/`. Copy to Windows:
```bash
cp -r ~/workspace/Isaac-GR00T/grasp_experiments/videos /mnt/c/Users/Username/Desktop/
```
 
---
 
## Task List
 
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
 
Examples include:
| Task | Description |
|---|---|
| `OpenDrawer_PandaOmron_Env` | Open a kitchen drawer |
| `PnPCounterToCab_PandaOmron_Env` | Pick from counter, place in cabinet |
| `CoffeeServeMug_PandaOmron_Env` | Move mug from coffee machine to counter |
| `StackBowlsInSink_PandaOmron_Env` | Pick and stack bowls in sink |
| `PrepareCoffee_PandaOmron_Env` | Multi-step coffee preparation |
|TurnOnSinkFaucet_PandaOmron_Env |Turn on sink faucet|
 
---
 
## Troubleshooting
 
| Issue | Cause | Fix |
|---|---|---|
| `ValueError: Key backend: 'module://matplotlib_inline...'` | Colab MPLBACKEND pollution | Add `os.environ['MPLBACKEND'] = 'Agg'` before running |
| `ImportError: flash_attn not installed` | Installed to system Python, not `.venv` | Use `.venv/bin/python -m pip install flash-attn` |
| `evdev build error: Python.h not found` | Missing Python dev headers in WSL | `sudo apt install python3.10-dev` |
| `apt install` 404 errors | Stale package cache | Run `sudo apt-get update` first |
| Rollout hangs at `Episodes: 0%` | Waiting for server connection | Verify ngrok address and that Colab shows `Server is ready` |
| `ping: True` but no Colab output | Normal — server doesn't log ping requests | Connection is fine, rollout is running |
| `CUDA_HOME not set` (flash-attn on WSL) | No CUDA toolkit in WSL | Expected — flash-attn is not needed locally, skip |
| `FlashAttention only supports Ampere GPUs or newer` | Colab assigned T4/V100 instead of A100 | Switch to A100 runtime, or uninstall flash-attn (results are identical without it) |
| `TypeError: create_env_robosuite() got unexpected keyword argument` | Custom env kwargs passed through wrapper | Use dynamic subclass pattern — bake parameters into class definition, not kwargs |
| `Server error: Video key must be shape (B, T, H, W, C)` | Missing T dimension in observation | Use official `create_grootrobocasa_env_class` + `run_rollout_gymnasium_policy` instead of manual obs construction |
| ngrok `connection refused` warnings during model download | Server not ready yet, ngrok retrying | Normal — wait for `Server is ready` message before connecting |
 
---
 
## References
 
- [RoboCasa365](https://robocasa.ai/) — kitchen simulation framework
- [Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA foundation model for generalist robots
- [GR00T RoboCasa Eval Guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa/README.md) — primary reference for this pipeline setup
- [GR00T Hardware Requirements](https://www.mintlify.com/NVIDIA/Isaac-GR00T/getting-started/hardware-requirements) — hardware guide
