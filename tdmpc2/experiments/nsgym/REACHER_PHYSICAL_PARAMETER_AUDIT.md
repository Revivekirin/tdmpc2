# Reacher physical parameter audit

Audited locally: NS-Gym 1.0.12.post2, Gymnasium 1.3.0, MuJoCo 3.3.7, Reacher-v5. Server inventory repeats these checks and records installed versions/source. The loader requires checkpoint environment package versions to match.

| Family | Official key | MuJoCo field | Nominal | Unit |
|---|---|---|---:|---|
| D0 | joint0_damping | dof_damping[joint0.id] | 1 | N m s/rad |
| D1 | joint1_damping | dof_damping[joint1.id] | 1 | N m s/rad |
| M0 | body0_mass | body_mass[body0.id] | 0.03560471674068433 | kg |
| M1 | body1_mass | body_mass[body1.id] | 0.03560471674068433 | kg |

Joint IDs equal their DOF addresses in this model; checked before use. The official mass setters change no body inertia. Both links have diagonal inertia approximately [3.917566039e-5, 3.917566039e-5, 1.738347935e-6] kg m². Positive mass and valid principal inertia triangle inequalities hold throughout the grid. Retain the official abstract mass-only intervention, explicitly not uniform material-density scaling at fixed geometry. No unrequested inertia intervention is introduced.

The experiment-only adapter applies the official setter and dependency resolver before reset, then reasserts the same target through StepWiseUpdate/DiscreteScheduler. Persistent parameters remain fixed across resets. Model constants are recomputed while full MuJoCo integration state is restored. Readback and unchanged inertia are asserted at reset and every step. Policy receives only the existing 10-dimensional observation, never parameter metadata.

Inventory saves exact setter source, actual values, dt/frame skip, action bounds, actuator gear, force limits and reward weights. Action-component saturation (|a|≥0.99) is separate from actual actuator force saturation; when no force limit exists, the latter is unavailable, not zero.
