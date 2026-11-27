# HalfCheetah Dynamics+Q adaptation protocol

Primary method: `dynamics_q_118`; source model SHA256:
`f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b`.
Configuration: `configs/dynamics_q.json`. See DYNAMICS_Q_IMPLEMENTATION_AUDIT.md
for target math, parameter ownership and departures from original Joint RL.

- Fixed `bthigh_damping` and `torso_mass`, each multiplied by
  {0.5,0.75,1,1.25,1.5,2}; no performance-based exclusions or reselection.
- Adaptation seeds 42000001–42000003. Held-out seeds 43000001–43000020.
- 12 labeled physical conditions × 3 adaptation seeds = 36 independent starts.
  Equivalent k=1 family labels are retained but not independent nominal tasks.
- 3,000 real environment transitions per stream (three 1,000-step episodes),
  118 dynamics steps, 118 Q steps. Matrix: 108,000 adaptation interactions,
  4,248 dynamics optimizer steps, 4,248 Q steps, 8,496 total optimizer steps.
- Fresh optimizers and replay per stream. No random seed-exploration phase.
  MPPI collection uses eval_mode=True. Episode reset advances the initially seeded
  adaptation RNG stream; held-out seeds never enter replay.
- Original Frozen/Decision manifests, logs, seed lists, checkpoint/config and
  physical-reference provenance must pass the existing audit before real work.
- One nonnominal damping ×2 smoke uses 100 transitions and 2+2 optimizer steps.
  It verifies real checkpoint loading, physics and all parameter invariants.
  Full adaptation is a separate explicit command gated by that real smoke.
- Save only the final scientific snapshot (3,000 transitions). No intermediate
  best-snapshot selection. Snapshot includes model, both new optimizer states,
  replay/counters, configuration and integrity hashes. It is for evaluation,
  not exact mid-episode continuation.
- Incomplete streams fail visibly. Resume preserves verified PASS children and
  archives incomplete attempts before restarting them from nominal. It never
  appends fresh updates to a partial snapshot.
- Full audit requires all 36 streams and validates logs, snapshots, independent
  nominal starts, parameter invariance, replay boundaries/seeds and 118+118 steps.

## Held-out evaluation

Prediction uses the exact hashed original Frozen transition bank: 20 episodes ×
1,000 transitions for each final snapshot, 720,000 fixed-data predictions, no new
environment transitions. Encoder remains nominal, making latent MSE comparable
with Frozen and Dynamics-only. Absolute and relative differences are retained;
relative error is missing when the Frozen denominator is effectively zero.

Shifted control uses 20 fresh matched evaluation seeds per snapshot: 720 episodes,
720,000 environment transitions. It records return/per-step return, velocity,
control cost, action magnitude/saturation, actuator-force diagnostics, episode
length, terminal/truncation and physical readback. Action saturation is not
actuator-force saturation; absent force limits remain missing rather than zero.
All model/optimizer states remain unchanged and evaluation adds no replay.

Nominal retention is an independent optional mode with the same 720-episode
budget. Shifted k=1 models measure nominal adaptation; they do not replace testing
every nonnominal-adapted model back in nominal physics.

Aggregation requires complete D+Q prediction and shifted batches, complete existing
TTT prediction/shifted batches and existing Joint shifted evaluation. It joins
parameter × severity × adaptation seed × evaluation seed exactly, checking the
shared Frozen return and fixed-bank hashes. It computes D+Q−D, D+Q−Frozen and
Joint−D+Q, retaining all signs and seed identities. Mean/median/std/95% bootstrap
CI first average 20 evaluation seeds within each adaptation seed; n=3 streams
is the sampling unit. Frozen baselines are reused, not independent replicas.
Training-time diagnostics are descriptive traces, never independent time samples.

## Interpretation boundaries

Matching 3,000 interactions and 118 decision events does not isolate Q as a causal
factor: D+Q uses reward supervision, additional optimizer computation and target
updates, and its learned planner changes future replay/actions. Q target sampling
also consumes Torch RNG; identical adaptation seeds do not force identical data.
Joint remains a procedure-level baseline with 2,001 model/actor updates, restored
Adam, larger sequence batches and eval_mode=False collection. Its changing encoder
makes Joint latent self-consistency MSE unsuitable for the fixed-encoder MSE plot.

Matched-replay and high-update Dynamics-only matrices are not implemented or
launched in this primary ablation. Existing Dynamics-only_118 is consumed as the
reference, not rerun. No Q-only or encoder adaptation is added. Improvement does
not by itself establish Q-value mismatch as the cause of earlier Joint behavior.
