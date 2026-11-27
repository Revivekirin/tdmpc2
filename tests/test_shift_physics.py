"""SAPIEN API contract fixtures, NOT real physics validation."""
import types
import unittest
import numpy as np
from benchmark_imports import package
from test_shift_benchmark import config
from test_dynamics_shift import Native


class Material:
    def __init__(self, s, d, r): self.values = [s,d,r]
    def get_static_friction(self): return self.values[0]
    def get_dynamic_friction(self): return self.values[1]
    def get_restitution(self): return self.values[2]


class Shape:
    def __init__(self, material): self.material=material
    def get_collision_groups(self): return [1,1,0,0]
    def set_collision_groups(self,*groups): self.groups=groups
    def get_physical_material(self): return self.material
    def set_physical_material(self, material): self.material=material


class Scene:
    def __init__(self): self.actors=[]
    def get_all_actors(self): return self.actors.copy()
    def remove_actor(self, actor): self.actors.remove(actor)
    def create_physical_material(self,*args): return Material(*args)


class Builder:
    def __init__(self, actor): self.actor=actor
    def set_mass_and_inertia(self,mass,pose,inertia): self.properties=(mass,pose,inertia.copy())
    def build(self,name):
        result=Actor(self.actor.scene,name,self.actor.shapes[0].material)
        result.mass,result.cmass,result.inertia=self.properties
        return result


class Actor:
    def __init__(self,scene,name,material):
        self.scene,self.name=scene,name
        scene.actors.append(self)
        self.shapes=[Shape(material)]
        self.shapes[0].geometry=type('PlaneGeometry' if name=='ground' else 'BoxGeometry',(),{})()
        self.mass,self.inertia=2.,np.array([.1,.2,.3])
        self.cmass=types.SimpleNamespace(p=np.array([.01,.02,.03]),q=np.array([1.,0,0,0]))
        self.state=np.arange(13,dtype=float)
    def get_mass(self): return self.mass
    def get_inertia(self): return self.inertia.copy()
    def get_cmass_local_pose(self): return self.cmass
    def get_builder(self): return Builder(self)
    def get_pose(self): return self.state[:7].copy()
    def set_pose(self,pose): self.state[:7]=pose
    def get_velocity(self): return self.state[7:10].copy()
    def set_velocity(self,v): self.state[7:10]=v
    def get_angular_velocity(self): return self.state[10:13].copy()
    def set_angular_velocity(self,v): self.state[10:13]=v
    def pack(self): return self.state.copy()
    def unpack(self,state): self.state=state.copy()
    def get_name(self): return self.name
    def get_scene(self): return self.scene
    def get_collision_shapes(self): return self.shapes


class PhysicsNative(Native):
    def __init__(self):
        super().__init__(); self.configure()
    def configure(self):
        self.scene=Scene()
        self.shared=Material(.8,.6,.1)
        self.obj=Actor(self.scene,'cube',self.shared)
        self.ground=Actor(self.scene,'ground',self.shared)
        self.other=Actor(self.scene,'other',self.shared)
        self._actors=self.scene.get_all_actors()
    def reset(self,**kwargs):
        if kwargs.get('reconfigure'): self.configure()
        return super().reset(**kwargs)
    def get_obs(self): return self.state.copy()


class PhysicsTests(unittest.TestCase):
    def make(self,kind,schedule='fixed',**kwargs):
        cfg=config(kind,schedule,1.,1.5,**kwargs)
        cfg['application']='reset'
        cfg['parameter'].update(actor='obj',surfaces=['ground'])
        return package.make_shift(PhysicsNative(),cfg,task='pick-cube')

    def test_mass_readback_reset_reconfigure(self):
        env=self.make('object_mass'); before=env.env.obj.pack()
        env.reset(seed=7)
        self.assertEqual(env.readback['mass'],3.)
        np.testing.assert_allclose(env.readback['inertia'],[.15,.3,.45])
        np.testing.assert_array_equal(env.env.obj.pack(),before)
        self.assertIs(env.env._actors[0],env.env.obj)
        self.assertEqual(len(env.env.scene.actors),3)
        for reconfigure in [False,True,False]:
            env.reset(seed=7,reconfigure=reconfigure)
            self.assertEqual(env.readback['mass'],3.)
            self.assertEqual(len(env.env.scene.actors),3)
        env.backend.restore()
        self.assertEqual(env.env.obj.mass,2.)
        np.testing.assert_allclose(env.env.obj.inertia,[.1,.2,.3])

    def test_friction_readback_isolated_restore(self):
        env=self.make('contact_friction'); env.reset()
        np.testing.assert_allclose(env.readback['materials'],[[1.2,.9,.1]]*2)
        self.assertEqual(env.env.shared.values,[.8,.6,.1])
        self.assertIs(env.env.other.shapes[0].material,env.env.shared)
        env.reset(reconfigure=True)
        np.testing.assert_allclose(env.readback['materials'],[[1.2,.9,.1]]*2)
        env.backend.restore()
        self.assertIs(env.env.obj.shapes[0].material,env.env.shared)
        self.assertIs(env.env.ground.shapes[0].material,env.env.shared)

    def test_physics_sample_hold_and_resume(self):
        for kind in ['object_mass','contact_friction']:
            env=self.make(kind,'abrupt',switch_step=1)
            env.reset()
            records=[env.step(np.zeros(4))[3]['shift_record'] for _ in range(3)]
            self.assertEqual([r['applied_parameter'] for r in records],[1,1,1])
            self.assertFalse(records[-1]['mid_episode_supported'])
            env.reset()
            self.assertEqual(env.step(np.zeros(4))[3]['shift_record']['applied_parameter'],1.5)
            state=env.state_dict()
            clone=self.make(kind,'abrupt',switch_step=1); clone.reset(); clone.load_state_dict(state)
            self.assertEqual(clone.global_step,env.global_step)
            if kind=='object_mass': self.assertEqual(clone.readback['mass'],3.)
            else: np.testing.assert_allclose(clone.readback['materials'],[[1.2,.9,.1]]*2)

    def test_plane_surface_selector(self):
        env=self.make('contact_friction')
        env.backend.surface_names=['@plane']
        env.reset()
        self.assertEqual(len(env.readback['materials']),2)

    def test_physics_validation(self):
        with self.assertRaisesRegex(ValueError,'reset-only'):
            package.make_shift(PhysicsNative(),config('object_mass'),task='pick-cube')
        with self.assertRaisesRegex(ValueError,'only for pick-cube'):
            package.make_shift(PhysicsNative(),config('contact_friction'),task='stack-cube')
        env=self.make('contact_friction'); env.backend.surface_names=['missing']
        with self.assertRaisesRegex(ValueError,'surface'):
            env.reset()
