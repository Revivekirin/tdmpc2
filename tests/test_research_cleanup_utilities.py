"""Historical tree digest and checkpoint inspection contracts."""
import importlib.util
import hashlib
import ast
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / 'experiments/nsgym/scripts'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tdmpc2'))
sys.path.insert(0, str(SCRIPTS))

from experiments.nsgym.analysis.control_q.models import tree_hash as control_tree_hash


def training_tree_hash():
    """Load only the pure function; this Mac lacks the isolated NS-Gym runtime."""
    source = SCRIPTS / 'training_support.py'
    node = next(item for item in ast.parse(source.read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == 'tree_hash')
    namespace = {'hashlib': hashlib, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['tree_hash']


def test_tree_hash_historical_bytes_and_errors():
    assert hashlib.sha256((SCRIPTS / 'training_support.py').read_bytes()).hexdigest() == (
        'd59e808e54a17b6d69548131d4f0e94c01ae64e5e6a4c91d414139c2b9af85ba')
    assert hashlib.sha256((PROJECT / 'experiments/nsgym/analysis/control_q/models.py').read_bytes()).hexdigest() == (
        '353eedfcc4d994e2f1c94967b41bb77d2dce79262d36c3255f9fccdbd30ecf94')
    training_hash = training_tree_hash()
    array = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cases = [
        ({'z': [array[:, ::2], ('label', 3, 1.5, None, True)],
          2: torch.tensor([1, 2], dtype=torch.int64),
          'a': {'b': torch.zeros((0, 2))}},
         '9797ca10968c815086a68ae76234144a224189fef2a8e17a654cec9960c8a76d'),
        (torch.tensor([-1.5, 0, 2.25], dtype=torch.float64),
         '827edcfc00e3b17c08c014a2d551bf64f47f0d54b9544ce737df8013b777d600'),
        (array[:, ::2], '10444e30426ca0529e884ca01c9973666a359fe6f9d5aa613cf45f57e07d48f7'),
        ({}, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
        (torch.tensor(3.5, requires_grad=True),
         'c1a087434c3b52e023597328936d94948499e90be19e7c640fb619fe7cdebccc'),
    ]
    for value, expected in cases:
        assert training_hash(value) == expected
        assert control_tree_hash(value) == expected
    sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.]),
                                     (2,), check_invariants=True)
    for function in (training_hash, control_tree_hash):
        with pytest.raises(RuntimeError, match='unsupported memory format'):
            function(sparse)


@pytest.fixture
def inspection_modules(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model.pt').write_bytes(b'unchanged nominal bytes')
    fake = ModuleType('training_support')
    calls = []
    def load(path):
        calls.append(('load', path))
        return object(), {'config': {'task': fake.task}, 'environment_transitions': 750000}
    def evaluate(agent, config, seeds, **kwargs):
        calls.append(('evaluate', agent is None, tuple(seeds), kwargs))
        offset = 0 if agent is not None else 2 if kwargs.get('random_actions') else 4
        return [dict(episode_return=float(seed - 20000000 + offset),
                     goal_distance_final=float(10 + offset),
                     goal_progress=float(1 - offset)) for seed in seeds], None
    fake.load_checkpoint = load
    fake.evaluate = evaluate
    monkeypatch.setitem(sys.modules, 'training_support', fake)
    modules = []
    for name in ('inspect_pusher', 'inspect_reacher'):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return checkpoint, fake, calls, modules


@pytest.mark.parametrize('task', ['pusher', 'reacher'])
def test_inspection_cli_and_task_contract(inspection_modules, tmp_path, task, monkeypatch, capsys):
    checkpoint, fake, calls, modules = inspection_modules
    module = modules[0 if task == 'pusher' else 1]
    fake.task = f'nsgym-{task}'
    with pytest.raises(ValueError, match='At least two evaluation seeds'):
        module.inspect(checkpoint, tmp_path / 'too-short', 1)
    assert not calls
    fake.task = 'nsgym-reacher' if task == 'pusher' else 'nsgym-pusher'
    with pytest.raises(ValueError, match=f'{task.capitalize()} checkpoint required'):
        module.inspect(checkpoint, tmp_path / 'wrong-task', 2)
    fake.task = f'nsgym-{task}'
    output = tmp_path / task
    monkeypatch.setattr(sys, 'argv', [module.__file__, '--help'])
    with pytest.raises(SystemExit) as help_exit:
        module.main()
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert all(flag in help_text for flag in ('--checkpoint', '--output', '--episodes'))
    monkeypatch.setattr(sys, 'argv', [module.__file__])
    with pytest.raises(SystemExit) as usage_exit:
        module.main()
    assert usage_exit.value.code == 2
    capsys.readouterr()
    monkeypatch.setattr(sys, 'argv', [module.__file__, '--checkpoint', str(checkpoint),
                                      '--output', str(output), '--episodes', '2'])
    module.main()
    assert capsys.readouterr().out.strip() == str(output)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['checkpoint'] == str(checkpoint)
    assert summary['checkpoint_sha256'] == hashlib.sha256(b'unchanged nominal bytes').hexdigest()
    assert summary['environment_transitions'] == 750000
    assert summary['episodes'] == 2
    assert summary['evaluation_seeds'] == [20000001, 20000002]
    assert [summary[k] for k in ('policy_return_mean', 'random_return_mean',
                                'zero_action_return_mean', 'paired_gain_mean')] == [1.5, 3.5, 5.5, -2.0]
    assert [summary[k] for k in ('policy_final_goal_distance_mean',
                                'random_final_goal_distance_mean',
                                'zero_action_final_goal_distance_mean')] == [10.0, 12.0, 14.0]
    assert [summary[k] for k in ('policy_goal_progress_mean',
                                'random_goal_progress_mean',
                                'zero_action_goal_progress_mean')] == [1.0, -1.0, -3.0]
    assert len(summary['paired_gain_95pct_bootstrap']) == 2
    assert summary['paired_gain_95pct_bootstrap'] == [-2.0, -2.0]
    assert summary['interpretation'] == (
        'Interim nominal checkpoint inspection; no official Pusher success rate or online adaptation claim.'
        if task == 'pusher' else
        'Nominal validation only; return gain alone does not establish target reaching.')
    assert [entry[3] for entry in calls if entry[0] == 'evaluate'] == [
        {}, {'random_actions': True}, {'zero_actions': True}]
    assert all((output / f'{name}.csv').is_file()
               for name in ('policy', 'random', 'zero_action'))
    assert (output / 'policy.csv').read_text().splitlines() == [
        'episode_return,goal_distance_final,goal_progress',
        '1.0,10.0,1.0', '2.0,10.0,1.0']
    with pytest.raises(FileExistsError):
        module.inspect(checkpoint, output, 2)
