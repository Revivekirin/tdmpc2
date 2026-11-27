"""SAPIEN 2 reset-time property shifts. No assumed runtime mass setter.

Mass uses ActorBase.get_builder -> ActorBuilder.set_mass_and_inertia -> build,
replaces the task actor reference and the BaseEnv._actors cache in place.
Only PickCube's single free object is supported by the factory.
"""
from copy import deepcopy
import numpy as np
from .base import ShiftWrapper


class SapienPhysics:
    def __init__(self, env, parameter):
        self.env, self.parameter = env.unwrapped, parameter
        self.kind = parameter['type']
        self.actor_attr = parameter.get('actor', 'obj')
        self.surface_names = parameter.get('surfaces', ['@plane'])
        self.actor = None
        self.baseline = None
        self.materials = []
        self.factor = 1.

    def bind(self):
        actor = getattr(self.env, self.actor_attr, None)
        if actor is None:
            raise ValueError(f'Physics actor attribute {self.actor_attr!r} is missing.')
        if actor is self.actor:
            return
        self.actor = actor
        if self.kind == 'object_mass':
            if not all(callable(getattr(actor, name, None)) for name in
                       ('get_mass', 'get_inertia', 'get_cmass_local_pose', 'get_builder')):
                raise NotImplementedError('SAPIEN 2 actor mass readback/build API is unavailable.')
            self.baseline = dict(mass=float(actor.get_mass()), inertia=actor.get_inertia().copy(),
                                 cmass=actor.get_cmass_local_pose())
        else:
            actors = actor.get_scene().get_all_actors()
            surfaces = []
            for name in self.surface_names:
                matches = [a for a in actors if (any(type(s.geometry).__name__ == 'PlaneGeometry'
                           for s in a.get_collision_shapes()) if name == '@plane' else a.get_name() == name)]
                if len(matches) != 1:
                    raise ValueError(f'Expected exactly one contact surface named {name!r}.')
                surfaces.extend(matches)
            self.materials = []
            for selected in [actor] + surfaces:
                shapes = selected.get_collision_shapes()
                if not shapes:
                    raise ValueError(f'No collision shapes on {selected.get_name()}.')
                for shape in shapes:
                    material = shape.get_physical_material()
                    self.materials.append((shape, material, (material.get_static_friction(),
                                           material.get_dynamic_friction(), material.get_restitution())))
        self.factor = 1.

    def apply(self, factor):
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError('Physical multipliers must be positive and finite.')
        self.bind()
        if factor != self.factor:
            if self.kind == 'object_mass':
                old = self.actor
                builder = old.get_builder()
                if builder is None:
                    raise NotImplementedError('Actor builder is not retained by this SAPIEN build.')
                # Preserve pose, velocities, geometry/materials and center of mass.
                pose, velocity, angular_velocity = old.get_pose(), old.get_velocity(), old.get_angular_velocity()
                scene, name = old.get_scene(), old.get_name()
                cache = getattr(self.env, '_actors', None)
                if cache is None or not any(a is old for a in cache):
                    raise NotImplementedError('Expected ManiSkill2 BaseEnv actor cache for safe replacement.')
                builder.set_mass_and_inertia(self.baseline['mass'] * factor,
                                             self.baseline['cmass'], self.baseline['inertia'] * factor)
                groups = [shape.get_collision_groups() for shape in old.get_collision_shapes()]
                new = builder.build(name=name)
                new_shapes = new.get_collision_shapes()
                if len(new_shapes) != len(groups):
                    scene.remove_actor(new)
                    raise RuntimeError('Rebuilt actor collision shapes differ from the original.')
                for shape, group in zip(new_shapes, groups):
                    shape.set_collision_groups(*group)
                new.set_pose(pose)
                new.set_velocity(velocity)
                new.set_angular_velocity(angular_velocity)
                setattr(self.env, self.actor_attr, new)
                self.env._actors = [new if a is old else a for a in cache]
                scene.remove_actor(old)
                self.actor = new
            else:
                for shape, original, values in self.materials:
                    # Never mutate a shared material (e.g. robot fingers or other actors).
                    material = original if factor == 1 else self.actor.get_scene().create_physical_material(
                        values[0] * factor, values[1] * factor, values[2])
                    shape.set_physical_material(material)
            self.factor = factor
        readback = self.readback()
        if self.kind == 'object_mass':
            np.testing.assert_allclose(readback['mass'], self.baseline['mass'] * factor, rtol=1e-5)
            np.testing.assert_allclose(readback['inertia'], self.baseline['inertia'] * factor, rtol=1e-5)
            np.testing.assert_allclose(readback['center_of_mass'], self.baseline['cmass'].p, atol=1e-7)
            np.testing.assert_allclose(readback['center_of_mass_orientation'], self.baseline['cmass'].q, atol=1e-7)
        else:
            for actual, (_, _, baseline) in zip(readback['materials'], self.materials):
                np.testing.assert_allclose(actual, [baseline[0]*factor, baseline[1]*factor, baseline[2]], rtol=1e-5)
        return readback

    def readback(self):
        if self.kind == 'object_mass':
            return dict(mass=float(self.actor.get_mass()), inertia=self.actor.get_inertia().copy(),
                        center_of_mass=self.actor.get_cmass_local_pose().p.copy(),
                        center_of_mass_orientation=self.actor.get_cmass_local_pose().q.copy(),
                        nominal_mass=self.baseline['mass'], inertia_policy='scale_with_mass')
        return dict(materials=[[s.get_physical_material().get_static_friction(),
                                s.get_physical_material().get_dynamic_friction(),
                                s.get_physical_material().get_restitution()] for s, _, _ in self.materials],
                    nominal_materials=[list(values) for _, _, values in self.materials],
                    combine_rule='PhysX default average; SAPIEN 2 exposes no combine-mode readback')

    def restore(self):
        if self.actor is not None:
            self.apply(1.)


class PhysicsShiftWrapper(ShiftWrapper):
    mid_episode_supported = False

    def __init__(self, env, config):
        super().__init__(env, config)
        if config.get('application', 'step') != 'reset':
            raise ValueError('Physics shifts are reset-only: application=reset is required.')
        if any(np.asarray(v).shape or float(v) <= 0 for v in self.schedule.possible_values()):
            raise ValueError('Mass/friction schedules require positive scalar multipliers.')
        if self.parameter['nominal'] != 1:
            raise ValueError('Physical nominal multiplier must be 1.')
        self.backend = SapienPhysics(env, self.parameter)
        self.held_parameter = None
        self.readback = None

    def reset(self, *args, **kwargs):
        self.backend.restore()
        super().reset(*args, **kwargs)
        self.held_parameter = self.schedule.value(self.clock_time())
        self.readback = self.backend.apply(self.held_parameter[0])
        # State observation is regenerated after the actor reference is replaced.
        return self.env.unwrapped.get_obs()

    def current_parameter(self):
        if self.held_parameter is None:
            raise RuntimeError('Reset the physics wrapper first.')
        return self.held_parameter

    def transform(self, action, value):
        return action.copy()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['shift_record']['physics_readback'] = deepcopy(self.readback)
        info['shift_record']['application'] = 'reset_sample_and_hold'
        return obs, reward, done, info

    def state_dict(self):
        return dict(super().state_dict(), held_parameter=deepcopy(self.held_parameter))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.held_parameter = deepcopy(state['held_parameter'])
        if self.held_parameter is not None:
            self.readback = self.backend.apply(self.held_parameter[0])
