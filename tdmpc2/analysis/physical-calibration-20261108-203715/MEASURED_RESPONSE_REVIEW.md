# SAPIEN physical-response review (2026-11-08)

User-executed server run `physical-calibration-20261108-203715`; this review uses its copied raw probes, Frozen episode records, [calibration CSV](measured-v1/physical_severity_calibration.csv), and [measured graph](measured-v1/physical_response_by_severity.png). Historical pilots and theoretical reference plots are not treated as new measurements. Server HEAD `b809299d46f1c37b5c20fb1e8cbbea40062771de`, ManiSkill2 0.4.1, SAPIEN 2.2.1, Gym 0.21.0. Frozen/probe source digests match; checkpoint SHA256 `1781f0fa959ee51fe96450e5bf5bf24f1d87ce515a89730e6b964c560144a258`. All 17 Frozen completions record zero gradient updates and zero parameter drift.

| Shift | Physical-response verdict | Research suitability |
|---|---|---|
| Gain | **VERIFIED** | EEF delta-command attenuation, not motor torque gain loss. |
| Delay | **VERIFIED** | Native arm-command transport delay; separate hidden queue and arm/gripper timing from ordinary physical parameters. |
| Mass | **PARTIAL** | Mass/inertia and support impulse change verified; grasp–lift coverage fails for 2×/3× at seed 704000. |
| Friction | **PARTIAL** | Material change and both contact kinds verified; measured relative paths do not isolate pure slip, so grasp-slip severity remains unproven. |

`PARTIAL` means the present probe does not support the full research claim. No shift is `UNSUPPORTED` at the implementation/readback level.

## Provenance and measurement window

The installed arm controller is `pd_ee_delta_pos`: normalized arm command ±1 maps to EE-frame position delta ±0.1 m; `use_delta=true`, `use_target=false`, `interpolate=false`, stiffness 1000, damping 100, force limit 100. Gripper is normalized absolute position −0.01 to 0.04 m. Simulator runs at 500 Hz; native controller at 20 Hz; policy decision period is two native steps. Probe replays Frozen nominal **native-step** NPZ actions, not policy decisions. The three calibration seeds are 704000–704002. Nominal tapes last 9, 17, and 13 native steps, despite the configured upper bound of 40. Gain/Delay object-contact-free candidate windows comprise only two native steps per seed. Each native step has 25 physics substeps. Final benchmark seeds 804000–804009 were not run or used for selection.

There are 51 candidate–seed probe comparisons: 49 `MEASURED`, 2 `PARTIAL`. Frozen evaluation totals 102 nominal/shifted episodes. Numeric 3-vector position, normal, and impulse arrays are present in raw contact points. Target pose (base/world frame) and measured EEF world pose are separate. Contact impulse is N·s, not force. Nominal/shifted responses use a common native-step prefix. Object-contact-free masks do not rule out every possible robot collision.

## Results

**Gain.** Free-space **measured EEF** RMSE from nominal is 5.9–6.5 mm at 0.9, 17.8–19.7 mm at 0.7, 29.7–33.0 mm at 0.5, and 41.8–46.3 mm at 0.3. Target RMSE is separately recorded; at 0.3 it is 108.3–127.6 mm. All candidate Frozen evaluations succeed 3/3. Mean episode length rises from nominal 7.0 to 22.0 policy decisions at 0.3. Gain 0.3 is a strong, feasible **provisional** command-attenuation candidate.

**Delay.** Issued/executed history verifies 1/2/3 native-step arm delays and unchanged current gripper command. At 20 Hz these are 50/100/150 ms. Measured EEF onset differs from nominal by exactly 50/100/150 ms at all three seeds, using the documented 10 μm displacement threshold. This is observed onset, not an independent causal motor-latency estimate. EEF RMSE is 36.0–42.5 / 63.0–76.7 / 81.9–108.2 mm. Frozen success is 3/3, 1/3, 0/3 at delays 1, 2, 3; delay 1 is the feasible provisional candidate.

**Mass.** Nominal cube mass is 0.064000003 kg. Readback at 1.5/2/3× is 0.096/0.128/0.192 kg, with inertia scaled equally and COM pose preserved. In seed 704000, initial support-plane vertical impulse per physics substep is 0.00125568 N·s nominal, then 0.00188352/0.00251136/0.00376704 N·s: direct mass-proportional simulator response. Yet on that seed's nine-step prescribed tape, nominal lift is 49.1 mm while 2×/3× lift is 7.65/8.30 mm; shifted dual-finger contact fraction is zero. Those two comparisons are `PARTIAL`. Seeds 704001/704002 do show dual-finger contact and lift; 3× lift is about 3.8/4.3 mm below nominal. All mass candidate Frozen closed-loop evaluations succeed 3/3, which establishes controller feasibility under changed actions but does not repair missing prescribed-tape grasp evidence. The 1.5× probe covers all three seeds, though two seeds show small response changes. Final mass severity is deferred.

**Friction.** Cube and ground static/dynamic material coefficients change together from nominal 1/1 to requested 0.7/0.5/0.3; both fingers remain 2/2. Raw contacts include object–finger and object–plane with sustained intervals in every seed, so this is not a contact-free test. At 0.3, nominal-relative plane-contact horizontal path difference is +2.18 to +9.74 mm. Finger-contact relative-position path differences are −13.47, +1.15, −6.56 mm across seeds. At 0.5, the plane difference spans +1.09 to +15.74 mm, so even that response is not monotonic over every seed. These signed differences compare cumulative paths under changing contact timing; they do not measure pure tangential slip or friction force. Effective material-pair combine coefficient was not read back. Every friction Frozen evaluation succeeds 3/3 with nominal-like mean length 7.0 decisions, so PickCube success does not distinguish candidates here.

## Selection state

The supplied [calibrated YAML](measured-v1/calibrated_severity.yaml) correctly has `selected_severities: {}` and `ready_for_benchmark: false`. `MEASURED` denotes complete input evidence for the script's coverage rule, not final research approval. The researcher still must record shift-specific minimum physical response, minimum Frozen success, task-feasibility evidence, and the rationale in `selection_reviews`. With only three calibration seeds, success fractions are descriptive, not statistically conclusive. Different shifts must not be normalized by one factor or physical-response unit.

Prioritize Gain 0.3 and Delay 1 for that review. For Mass 2×/3×, use a separate native tape that actually includes both-finger grasp and lift at seed 704000, keeping this run unchanged. For Friction, inspect contact-point tangential relative motion or a task where slip changes outcome before finalizing a grasp-related severity. The existing graph plots the calculated Mass 2×/3× seed-704000 points but does not mark their `PARTIAL` coverage; do not read those lines as validated grasp–lift curves. Existing theoretical reference graphs remain separate in `analysis/physical_dynamics/`.
