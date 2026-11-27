"""Load the NumPy-only benchmark without importing optional RL dependencies."""
import importlib.util
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'tdmpc2/envs/wrappers/dynamics_shift'
spec = importlib.util.spec_from_file_location('benchmark_shift', PATH / '__init__.py', submodule_search_locations=[str(PATH)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
sys.path.insert(0, str(ROOT / 'tdmpc2'))
