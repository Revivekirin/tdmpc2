from pathlib import Path
import shlex
import subprocess
import sys
import torch

ROOT = Path("/nfs4/jhkim/repos/tdmpc2")
CKPT_DIR = ROOT / "checkpoints/maniskill2"

# 이름, dynamics_shift, schedule, 추가 override
COMMON = [
    ("nominal",       "nominal",     None,         []),
    ("gain-fixed",    "action_gain", "fixed",      ["dynamics_shift.parameter.shifted=0.7"]),
    ("delay-fixed",   "action_delay","fixed",      ["dynamics_shift.parameter.shifted=2"]),
    ("gain-abrupt10", "action_gain", "abrupt",     ["schedule.switch_step=10"]),
    ("gain-abrupt20", "action_gain", "abrupt",     ["schedule.switch_step=20"]),
    ("gain-repeated", "action_gain", "repeated",   ["schedule.interval=20"]),
    ("gain-sequence", "action_gain", "sequence",   []),
    ("gain-sine",     "action_gain", "sinusoidal", ["dynamics_shift.parameter.shifted=1.3", "schedule.interval=40"]),
    ("gain-drift",    "action_gain", "drift",      ["schedule.switch_step=10", "schedule.duration=40"]),
]
PICK_ONLY = [
    ("mass-fixed",     "object_mass",      "fixed", ["dynamics_shift.parameter.shifted=0.5"]),
    ("friction-fixed", "contact_friction", "fixed", ["dynamics_shift.parameter.shifted=0.5"]),
]

jobs = []
for task in ("pick-cube", "stack-cube"):
    for number in (1, 2, 3):
        ckpt = CKPT_DIR / f"{task}-{number}.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = payload["model"] if "model" in payload else payload
        episodic = any(k.startswith("_termination.") for k in state)
        for name, shift, schedule, overrides in COMMON + (PICK_ONLY if task == "pick-cube" else []):
            jobs.append((task, ckpt, episodic, name, shift, schedule, overrides))
        del payload, state

for index, (task, ckpt, episodic, name, shift, schedule, overrides) in enumerate(jobs, 1):
    mode = "standard" if name == "nominal" else "paired"
    cmd = [
        sys.executable, "evaluate_shift.py",
        f"task={task}",
        "model_size=5",
        f"episodic={str(episodic).lower()}",
        f"checkpoint={ckpt}",
        "eval_episodes=50",
        "seed=1",
        "save_video=false",
        f"benchmark_id={task}-{ckpt.stem}-{name}",
        f"dynamics_shift={shift}",
        "adaptation=frozen",
        f"evaluation={mode}",
    ]
    if schedule:
        cmd.append(f"schedule={schedule}")
    cmd.extend(overrides)
    print(f"\n[{index}/{len(jobs)}] {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT / "tdmpc2", check=True)

print(f"\nALL COMPLETE: {len(jobs)} conditions", flush=True)
