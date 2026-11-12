"""CPU bookkeeping tests; these do not substitute for the real CUDA smoke."""
import os
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_full_joint_rl as joint


class Contracts(unittest.TestCase):
    def test_grid_and_no_evaluation_seed_replay(self):
        c=joint.protocol()
        self.assertEqual(len(c['conditions'])*len(c['adaptation_seeds']),36)
        self.assertFalse(set(c['adaptation_seeds']) & set(c['evaluation_seeds']))
        self.assertEqual(c['transitions']-c['first_update']+1,c['expected_updates'])
        self.assertFalse(c['automatic_heldout_evaluation'])

    def test_reacher_uses_selected_grid_and_episode_cadence(self):
        c=joint.protocol('reacher')
        self.assertEqual(c['task'],'reacher')
        self.assertEqual(c['horizon'],50)
        self.assertEqual(c['transitions'],10000)
        self.assertEqual(len(c['conditions'])*len(c['adaptation_seeds']),72)
        self.assertFalse(set(c['adaptation_seeds']) & set(c['evaluation_seeds']))
        self.assertEqual(joint.expected_updates(100,c),2)
        self.assertEqual(joint.expected_updates(10000,c),398)
        self.assertFalse(joint.should_update(50,c,1))
        self.assertTrue(joint.should_update(75,c,1))
        self.assertFalse(joint.should_update(75,c,0))

    def test_halfcheetah_update_cadence_is_unchanged(self):
        c=joint.protocol()
        self.assertEqual(joint.expected_updates(1004,c),5)
        self.assertEqual(joint.expected_updates(3000,c),2001)

    def test_reacher_rejects_checkpoint_outside_nominal_selection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'selected';checkpoint.mkdir()
            (checkpoint/'model.pt').write_bytes(b'model')
            chosen=dict(checkpoint=str(checkpoint),model_sha256=joint.base.digest(checkpoint/'model.pt'),
                        pipeline_config_fingerprint=joint.fingerprint(joint.reacher_config()))
            selection=root/'selection.json';selection.write_text(json.dumps(chosen))
            args=SimpleNamespace(selection=selection,checkpoint=checkpoint)
            self.assertEqual(joint.checkpoint_identity(args,joint.protocol('reacher')),chosen['model_sha256'])
            args.checkpoint=root/'other';args.checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError,'differs'):
                joint.checkpoint_identity(args,joint.protocol('reacher'))

    def test_evaluation_checks_training_dependencies_without_rejecting_new_runner(self):
        runner='experiments/nsgym/scripts/run_full_joint_rl.py'
        config='experiments/nsgym/configs/reacher_pipeline_v1.json'
        manifest={'source_hashes':{runner:'old-training-runner-hash',
                                   config:joint.base.digest(joint.base.ROOT/config)}}
        audit=joint.evaluation_sources(manifest)
        self.assertTrue(audit['unchanged_training_dependencies'])
        self.assertEqual(audit['training_runner_sha256'],'old-training-runner-hash')
        manifest['source_hashes'][config]='changed-config-hash'
        with self.assertRaisesRegex(ValueError,'Training dependency changed'):
            joint.evaluation_sources(manifest)

    def test_occupied_gpu_is_rejected_even_with_free_memory(self):
        with patch.object(joint.subprocess,'check_output',side_effect=[
                'GPU-test, 40000, 0\n','GPU-test, 3530262\n']):
            with self.assertRaisesRegex(RuntimeError,'3530262'):
                joint.gpu_status(0)

    def test_idle_gpu_and_existing_own_context(self):
        with patch.object(joint.subprocess,'check_output',side_effect=[
                'GPU-test, 40000, 0\n','GPU-other, 3530262\n']):
            self.assertEqual(joint.gpu_status(1)['uuid'],'GPU-test')
        with patch.object(joint.subprocess,'check_output',side_effect=[
                'GPU-test, 40000, 80\n',f'GPU-test, {os.getpid()}\n']):
            self.assertEqual(joint.gpu_status(1)['other_pids'],[])

    def test_insufficient_memory_is_rejected(self):
        with patch.object(joint.subprocess,'check_output',side_effect=['GPU-test, 100, 0\n','']):
            with self.assertRaises(RuntimeError):joint.gpu_status(0)

    def test_explicit_sharing_allows_busy_gpu_but_preserves_memory_guard(self):
        with patch.object(joint.subprocess,'check_output',side_effect=[
                'GPU-test, 23164, 99\n','GPU-test, 3530262\nGPU-test, 3530735\n']):
            result=joint.gpu_status(0,allow_shared=True)
            self.assertEqual(result['other_pids'],[3530262,3530735])
            self.assertTrue(result['allow_shared_gpu'])
        with patch.object(joint.subprocess,'check_output',side_effect=[
                'GPU-test, 100, 99\n','GPU-test, 3530262\n']):
            with self.assertRaises(RuntimeError):joint.gpu_status(0,allow_shared=True)

    def test_only_original_optimizer_parameters_are_enabled(self):
        class Parameter:
            def requires_grad_(self,value):self.requires_grad=value
        encoder,dynamics,reward,q,pi,target=[Parameter() for _ in range(6)]
        agent=SimpleNamespace(model=SimpleNamespace(parameters=lambda:iter([encoder,dynamics,reward,q,pi,target])),
            optim=SimpleNamespace(param_groups=[{'params':[encoder,dynamics,reward,q]}]),
            pi_optim=SimpleNamespace(param_groups=[{'params':[pi]}]))
        joint.enable_training(agent)
        self.assertTrue(all(x.requires_grad for x in (encoder,dynamics,reward,q,pi)))
        self.assertFalse(target.requires_grad)


if __name__=='__main__':unittest.main()
