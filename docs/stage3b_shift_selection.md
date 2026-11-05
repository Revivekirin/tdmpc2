# Stage 3B shift selection

Selected from the current local files, not an old ZIP. Local base revision:
`6c7333d` plus pre-existing uncommitted analysis/documentation changes.
No SSH host/repository path was supplied; latest server revision is UNVERIFIED.
The server revision must be checked before applying these changes there.

| Shift | Fixed magnitude | Existing evidence | Decision |
|---|---:|---|---|
| Action delay | 1 native step | Stage 3A reference condition; audit E5-d1 success 90%, length 26.38 vs nominal 98%, 9.80 | Retain reference |
| Action gain | 0.7 | Stage 2 MECHANISM_RESULTS: success 100→100%, length 8.24→10.60, latent MSE .000392→.001  --recovered-root results/stage3b-recovered/runs \
  --execute
)
action_gain/set-2의 완료 결과 없음: 재수집 가능
[2026-11-05 19:08:23,742][absl][INFO] - MUJOCO_GL=egl, attempting to import specified OpenGL backend.
[2026-11-05 19:08:23,768][OpenGL.acceleratesupport][INFO] - No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
[2026-11-05 19:08:23,893][absl][INFO] - MuJoCo library version is: 3.1.2
[2026-11-05 19:08:25.892] [svulkan2] [error] GLFW error: X11: The DISPLAY environment variable is missing
[2026-11-05 19:08:25.892] [svulkan2] [warning] Continue without GLFW.
Episode length: 100
Discount factor: 0.95
[2026-11-05 19:08:28.160] [svulkan2] [warning] A second renderer will share the same internal context with the first one. Arguments passed to constructor will be ignored.
{"fixed_data_path": "/nfs4/jhkim/repos/tdmpc2/results/stage3b-generalization/data/action_gain/set-2.pt", "sha256": "30fcbcab4c2a44fe13061fb22a0088c7cedeaaeab87c3fb703e17effb539963e", "native_steps": 198, "policy_decisions": 104}
Traceback (most recent call last):
  File "/nfs4/jhkim/repos/tdmpc2/scripts/run_stage3b.py", line 136, in <module>
    planned, skipped, archives = remaining_commands(planned, suite, output, args.recovered_root, args.checkpoint)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfs4/jhkim/repos/tdmpc2/scripts/run_stage3b.py", line 37, in remaining_commands
    raise ValueError(f'Missing fixed dataset: {dataset}')
ValueError: Missing fixed dataset: /nfs4/jhkim/repos/tdmpc2/results/stage3b-generalization/data/action_delay/set-0.pt642. Earlier audit E1-g07: 98%, 13.86 | Retain control/prediction decoupling condition |
| Object mass | ×1.5 | Audit E6-m15: 96% success, length 12.78 vs nominal 98%, 9.80 | Select existing severity with measurable length change |
| Contact friction | ×0.5 | Audit E7-f05: 94% success, length 13.68 vs nominal 98%, 9.80 | Select existing severity with success/length change |

Sources: `DYNAMICS_SHIFT_AUDIT.md`, first “서버 결과 수신 후 갱신” section
(overrides the historical unverified judgments later in that file), and
`MECHANISM_RESULTS.md`. Audit rows are 50 episodes each, not independent training
replicates. Do not pool these two historical evaluations or claim significance.
The selected values already exist in the Hydra presets and benchmark suite.
No new pilot is needed on the available evidence; no pilot or real Stage 3B run
was executed locally. Both methods use exactly the same magnitudes.

Existing physical validation: identity 21/21, preservation/restoration 6/6,
controlled contact response 2/2, cached E6/E7 readback passed. The mass intervention
scales mass AND inertia through actor rebuild at reset; friction scales object
and configured plane materials at reset. These are distinct from action-delay
history/partial observability. Native server API readback/source provenance and
nested simulator state remain limitations of those historical validations.

Native step is one control call (historically 50 ms at 20 Hz), not a physics
integration substep. ManiSkill repeats actions twice unless native termination
ends the repeat early. Always use measured native counts, not decisions ×2.
