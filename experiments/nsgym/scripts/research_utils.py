"""Experiment bookkeeping only; no learning algorithm implementations."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tdmpc2'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def metadata(config, checkpoint=None):
    packages = ['ns_gym','gymnasium','mujoco','numpy','torch','tensordict','torchrl','matplotlib','hydra-core','omegaconf']
    versions = {}
    for name in packages:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    files = sorted((ROOT/'experiments/nsgym/scripts').glob('*.py')) + sorted((ROOT/'tdmpc2/envs/nsgym').glob('*.py'))
    return dict(command=[sys.executable, *sys.argv], config=config,
                config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                versions=versions, python=sys.version,
                git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                source_hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
                checkpoint_sha256=digest(checkpoint) if checkpoint else None,
                ns_gym_tested_revision='127df4b714fe4fcf680721d8e9278a96090acd7e',
                ns_gym_source=importlib.metadata.distribution('ns_gym').read_text('direct_url.json'),
                created_unix=time.time())


def csv_rows(path, rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap(values, seed=90210, repeats=10000):
    import numpy as np
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=a[rng.integers(len(a),size=(repeats,len(a)))].mean(axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]
