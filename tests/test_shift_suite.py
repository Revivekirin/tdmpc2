"""CLI regression tests require neither CUDA nor a checkpoint file."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SuiteTests(unittest.TestCase):
    def commands(self, *extra):
        result = subprocess.run(
            [sys.executable, str(ROOT/'scripts/evaluate_shift_suite.py'),
             '--checkpoint', '/missing/pick-cube-1.pt', '--dry-run', *extra],
            capture_output=True, text=True, check=True)
        return [shlex.split(line) for line in result.stdout.splitlines()]

    def test_released_single_task_checkpoint_uses_five_million_preset(self):
        suite = json.loads((ROOT/'benchmarks/dynamics_shift_suite.json').read_text())
        self.assertEqual(suite['model_size'], 5)
        commands = self.commands()
        self.assertEqual(len(commands), len(suite['conditions']))
        for command in commands:
            self.assertEqual([arg for arg in command if arg.startswith('model_size=')], ['model_size=5'])
        self.assertIn('model_size=5', (ROOT/'scripts/evaluate_dynamics_shift.sh').read_text())

    def test_custom_model_size_override_and_condition_selection(self):
        commands = self.commands('--model-size', '1', '--ids', 'E0')
        self.assertEqual(len(commands), 1)
        self.assertIn('model_size=1', commands[0])
        self.assertNotIn('model_size=5', commands[0])
        self.assertIn('benchmark_id=E0', commands[0])
