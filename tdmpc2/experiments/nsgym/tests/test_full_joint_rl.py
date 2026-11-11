"""CPU bookkeeping tests; these do not substitute for the real CUDA smoke."""
import os
from pathlib import Path
import sys
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
