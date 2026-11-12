"""CPU contract tests; the server CUDA/physics smoke remains necessary."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_joint_final as e


class Store:
    def __init__(self):
        self.value = {}
        self.training = True

    def load_state_dict(self, value, strict=True):
        self.value = value

    def state_dict(self):
        return self.value

    def parameters(self):
        return iter([SimpleNamespace(requires_grad_=lambda value: None)])

    def eval(self):
        self.training = False

    def step(self):
        raise AssertionError('optimizer step forbidden')


class Loader(unittest.TestCase):
    def test_training_log_update_boundary_and_finite_metrics(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'training.jsonl'
            with path.open('w') as f:
                for n in range(1,3001):
                    f.write(json.dumps(dict(transition=n,updates=max(0,n-999),episode=n//1000,
                                            replay_episodes=n//1000,
                                            metrics={'loss':1.0} if n>=1000 else {}))+'\n')
            e.check_training_log(tmp)
            with path.open('r+') as f:
                data=f.readlines()
                bad=json.loads(data[999]);bad['updates']=0;data[999]=json.dumps(bad)+'\n'
                f.seek(0);f.writelines(data);f.truncate()
            with self.assertRaises(AssertionError):
                e.check_training_log(tmp)

    def test_joint_snapshot_loads_all_state_and_blocks_updates(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'snapshot-0003000.pt').write_bytes(b'fixture')
            identity = dict(parameter='bthigh_damping',severity=2.,adaptation_seed=42000001,
                            method='full_joint_rl',schedule='fixed')
            protocol = {'transitions':3000}
            config = {'task':'nsgym-halfcheetah'}
            payload = dict(identity=identity,checkpoint_sha256=e.ttt.SHA,protocol=protocol,
                           configuration=config,transitions=3000,updates=2001,
                           resume='evaluation_only_no_replay_or_environment_state',
                           model={'encoder':1},optim={'steps':2},pi_optim={'steps':3},scale={'value':4})
            (path / 'manifest.json').write_text(json.dumps(dict(identity=identity,checkpoint_config=config)))
            (path / 'completion.json').write_text(json.dumps(dict(snapshot_sha256='hash')))
            agent = SimpleNamespace(model=Store(),optim=Store(),pi_optim=Store(),scale=Store())
            agent.update = lambda: None
            fake_torch = SimpleNamespace(load=lambda *a,**kw:payload)
            fake_support = SimpleNamespace(tree_hash=lambda value:json.dumps(value,sort_keys=True))
            with patch.dict(sys.modules,{'torch':fake_torch,'training_support':fake_support}), \
                 patch.object(e.common,'digest',return_value='hash'), \
                 patch.object(e.ttt,'load_agent',return_value=(agent,{'config':config})):
                loaded, state_hash = e.load_agent(SimpleNamespace(checkpoint='checkpoint'),
                                                   {'path':str(path),'snapshot_sha256':'hash'},protocol)
            self.assertIs(loaded,agent)
            self.assertFalse(agent.model.training)
            self.assertEqual(agent.scale.state_dict(),payload['scale'])
            self.assertIn('encoder',state_hash)
            with self.assertRaisesRegex(RuntimeError,'cannot update'):
                agent.update()

    def test_resume_skips_completed_stream_without_rerunning(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            check=dict(parameter='bthigh_damping',severity=2.,adaptation_seed=42000001,
                       snapshot_sha256='hash')
            run=root/e.name(check);run.mkdir()
            (run/'completion.json').write_text(json.dumps(dict(status='PASS')))
            (root/'manifest.json').write_text(json.dumps(dict(mode='shifted',checkpoint_sha256=e.ttt.SHA,
                                                               audit_sha256='hash')))
            (root/'completion.json').write_text(json.dumps(dict(status='PARTIAL')))
            args=SimpleNamespace(output=str(root),resume=True,audit='audit',mode='shifted')
            with patch.object(e.ttt,'dependencies'), patch.object(e.common,'digest',return_value='hash'), \
                 patch.object(e,'evaluate_one',side_effect=AssertionError('reran completed stream')):
                status=e.evaluate(args,{'adaptation_seeds':[42000001],'evaluation_seeds':[43000001]},
                                  [check],{},smoke=False)
            self.assertEqual(status,'PASS')
            self.assertEqual(json.loads((root/'completion.json').read_text())['completed'],[run.name])


if __name__=='__main__':
    unittest.main()
