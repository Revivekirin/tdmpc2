"""Run unit/integration tests and save an honest machine-readable validation report."""
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'results/dynamics_shift_benchmark'
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    (output/'validation.txt').write_text(text)
    print(text)
    versions = {}
    for package in ['numpy','torch','gym','gymnasium','hydra-core','mani-skill2','sapien']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = dict(command=f'{sys.executable} scripts/validate_shift_benchmark.py',
                  python=platform.python_version(), packages=versions,
                  code_revision=subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                  tests_run=result.testsRun, tests_passed=result.testsRun-len(result.skipped)-len(result.errors)-len(result.failures),
                  skipped=[dict(test=test.id(), reason=reason) for test, reason in result.skipped],
                  failures=[dict(test=test.id(), traceback=trace) for test, trace in result.failures+result.errors],
                  frozen_checkpoint_evaluated=False, evaluation_artifacts=[],
                  note='Toy-environment/API-fixture tests are not simulator performance results. This command does not evaluate a model.')
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
