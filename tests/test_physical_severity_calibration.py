"""Synthetic bookkeeping tests only; never simulator evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('calibration',ROOT/'scripts/calibrate_physical_severity.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def state(x=0.):
    return dict(qpos=[0],qvel=[0],eef_pose=[x,0,0,1,0,0,0],object_pose=[x,0,.02,1,0,0,0],
                object_velocity=[0,0,0],object_angular_velocity=[0,0,0],
                object_in_fingers={'panda_leftfinger':[x,0,0,1,0,0,0],'panda_rightfinger':[x,0,0,1,0,0,0]})


def run(contact=False):
    rows=[]
    for i in range(3):
        contacts=[]
        if contact:
            for name in ['panda_leftfinger','panda_rightfinger','plane']:
                contacts.append(dict(kind='object-plane' if name=='plane' else 'object-finger',actors=['cube',name],
                    points=[dict(position=[0.,0.,0.],normal=[0.,0.,1.],impulse=[0.,0.,.1])]))
        s=state(.001*(i+1))
        rows.append(dict(commanded=[.1,0,0,-1],executed=[.1,0,0,-1],after=s,
                         target_pose_world=[.1,0,0,1,0,0,0],average_object_acceleration=[0,0,0],
                         substeps=[dict(state=s,contacts=contacts,target_pose_world=[.1,0,0,1,0,0,0])]))
    return dict(initial=state(),rows=rows,inventory=dict(sim_freq=20,control_freq=20,mass=.064,inertia=[1,1,1]))


class CalibrationTests(unittest.TestCase):
    def test_contact_free_is_missing_not_zero(self):
        metrics=c.trajectory(run(),1e-5)
        self.assertIsNone(metrics['finger_path']);self.assertIsNone(metrics['plane_path'])
        self.assertEqual(metrics['simultaneous_fraction'],0)

    def test_contact_conditioned_relative_path(self):
        metrics=c.trajectory(run(True),1e-5)
        self.assertAlmostEqual(metrics['finger_path'],.004)
        self.assertAlmostEqual(metrics['plane_path'],.002)
        self.assertEqual(metrics['simultaneous_fraction'],1)

    def test_common_window_not_unmatched_tail(self):
        a=run();b=run();b['rows']=b['rows'][:2]
        response,covered,_=c.paired_response(a,b,{},'action_gain',1,1e-5)
        self.assertEqual(response['common_native_steps'],2)
        self.assertEqual(response['nominal_native_steps'],3)
        self.assertEqual(response['eef_rmse_m'],0)

    def test_reject_string_contact_vector(self):
        a=run(True);a['rows'][0]['substeps'][0]['contacts'][0]['points'][0]['impulse']='[0,0,.1]'
        with self.assertRaises(ValueError):c.trajectory(a,1e-5)

    def test_seed_overlap_rejected(self):
        m=dict(calibration_seeds=[4000],final_benchmark_seeds=[4000])
        with self.assertRaises(ValueError):c.validate_manifest(m)

    def test_selection_requires_measurements_review_and_feasibility(self):
        rows=[dict(shift_type='action_gain',calibration_seed=i,evidence_status='MEASURED',measured_physical_response={'free_space_eef_rmse_m':.02},control_performance={'success':1}) for i in range(3)]
        self.assertFalse(c.eligible(rows,None,3)[0])
        with tempfile.TemporaryDirectory() as tmp:
            proof=Path(tmp)/'SYNTHETIC.txt';proof.write_text('SYNTHETIC fixture only')
            review=dict(response_metric='free_space_eef_rmse_m',minimum_absolute_response=.01,minimum_frozen_success_rate=1.,feasibility_seed_set=[0,1,2],threshold_rationale='synthetic test',rationale='test',task_feasibility_rationale='test',task_feasibility_evidence=str(proof))
            self.assertTrue(c.eligible(rows,review,3)[0])
            rows[1]['evidence_status']='PARTIAL'
            self.assertFalse(c.eligible(rows,review,3)[0])

    def test_frozen_goal_distance_is_measured_from_pose(self):
        from common.severity_evaluation import NativeTrajectory
        from types import SimpleNamespace as S
        native=S(obj=S(pose=S(p=np.array([1.,0,0]))),goal_pos=np.array([0.,0,0]),tcp=S(pose=S(p=np.zeros(3),q=np.array([1,0,0,0]))))
        shift=S(unwrapped=native,parameter={'type':'action_gain'},indices=slice(None,-1),low=np.full(4,-1),high=np.ones(4))
        trajectory=NativeTrajectory(shift,1,True)
        record=dict(applied_parameter=1,commanded_action=np.zeros(4),executed_action=np.zeros(4),episode_native_step=0,reward=0,success=False,done=False)
        trajectory.observe(np.zeros(4),record)
        self.assertEqual(trajectory.rows[0]['object_goal_distance_m'],1.)

    def test_export_numeric_evidence_without_selecting_one_seed(self):
        # This entire fixture and its figures stay in a temporary SYNTHETIC directory.
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-calibration-') as tmp:
            base=Path(tmp)
            m=dict(protocol='synthetic-test',calibration_set_id=0,calibration_seeds=[4000],final_benchmark_seeds=[14000],
                   candidates={'action_gain':[1,.7]},motion_detection_m=1e-5,max_native_steps=40,
                   minimum_calibration_seeds=3,selection_reviews={})
            c.dump(base/'manifest.json',m)
            probe=base/'probes/action_gain/seed-4000';probe.mkdir(parents=True)
            c.dump(probe/'metadata.json',dict(seed=4000,shift='action_gain',values=[1,.7],no_learning=True,tape_clock='native_control_step',
                    git_head='SYNTHETIC',source_sha256={},versions={},repository_source_sha256='SYNTHETIC',actions_sha256='SYNTHETIC'))
            for value in [1,.7]:
                data=run()
                for row in data['rows']:row['executed'][0]*=value
                c.dump(probe/f'action_gain-{value:g}.json',data)
                directory=c.frozen_dir(base,m,'action_gain',value);directory.mkdir(parents=True)
                c.dump(directory/'severity-metadata.json',dict(config={'stage3a':{'mode':'frozen'}},checkpoint_sha256='SYNTHETIC',
                    environment_versions={},source_sha256='SYNTHETIC',code_revision='SYNTHETIC'))
                c.dump(directory/'severity-complete.json',dict(gradient_updates=0,parameter_drift=0))
                c.dump(directory/'shifted-seed-4000.json',dict(evaluation_seed=4000,initial_object_goal_distance_m=.5,
                    result=dict(success=1,episode_length=2,native_steps=3)))
                np.savez(directory/'shifted-seed-4000.npz',object_goal_distance_m=np.array([.4,.2,.01]))
            c.export(base/'manifest.json',base/'output')
            result=c.read(base/'output/calibrated_severity.yaml')
            self.assertFalse(result['ready_for_benchmark'])
            self.assertEqual(result['selected_severities'],{})


if __name__=='__main__':unittest.main()
