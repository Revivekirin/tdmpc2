"""Short deterministic random-action rollout; never trains a policy."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tdmpc2'))
from envs.nsgym import HalfCheetah


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/nsgym/configs/nominal.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON artifact; refuses to overwrite an existing result')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    physical = []
    env = HalfCheetah(**config, recorder=physical.append)
    transitions = []
    rng = np.random.default_rng(config['seed'])
    try:
        initial, _ = env.reset()
        for step in range(config['max_episode_steps']):
            action = rng.uniform(-.8, .8, size=env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            transitions.append(dict(step=step, action=action.tolist(), observation=obs.tolist(),
                                    reward=reward, terminated=terminated, truncated=truncated,
                                    info=info))
            if terminated or truncated:
                break
        result = dict(config=config, python=platform.python_version(),
                      source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
                      adapter_sha256=hashlib.sha256((ROOT/'tdmpc2/envs/nsgym/halfcheetah.py').read_bytes()).hexdigest(),
                      tested_ns_gym_revision='127df4b714fe4fcf680721d8e9278a96090acd7e',
                      installed_ns_gym_source=importlib.metadata.distribution('ns_gym').read_text('direct_url.json'),
                      versions={name: importlib.metadata.version(name) for name in
                                ['ns_gym','gymnasium','mujoco','numpy','torch']},
                      initial_observation=initial.tolist(), transitions=transitions,
                      physical_parameters=physical)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(f'PASS: {config["mode"]}, {len(transitions)} transitions; {args.output}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
