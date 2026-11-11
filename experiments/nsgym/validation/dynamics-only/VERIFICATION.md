# Local bounded verification — 2026-11-11

Runtime: `/private/tmp/tdmpc2-nsgym-modern/bin/python`; CPU; real NS-Gym HalfCheetah physics; synthetic model/policy only.

Evidence: `results/nsgym/dynamics-only-validation-02/completion.json`, each of12 run manifests/completions,48 evaluation completions, and `analysis/` (18 PNGs and per-seed tables).

Online576 transitions + held-out2304 transitions =2880 real simulator transitions. Offline48000 observed-transition pairs were read from the existing Frozen fixed/abrupt NPZ without modifying them.12 run completions and48 evaluation completions verified; manifest JSON schema validated with jsonschema. No real750k checkpoint execution.

`python -m unittest discover -s experiments/nsgym/tests -p test_dynamics_only_ttt.py -v`:2 tests PASS (1.741s). Includes predictor-only equality, finite metrics, Frozen0updates, transition alignment assertions, episode-only update boundaries, abrupt onset/reset, snapshot mid-episode serialization and evaluation, rejected held-out replay/update calls, and detected frozen-parameter corruption. Fixtures are temporary for these tests; retained full evidence is in the directory above.

Both CLI `--help`, full-protocol `--dry-run`, Python compilation, aggregation and figure generation passed. Held-out MSE and interaction/gradient-cost figures were visually inspected. All figures carry synthetic fixture labels. k1.5-only smoke intentionally leaves the other predefined grid cells empty; no CI is reported for a single adaptation stream.

Actual750k CUDA smoke, full fixed/abrupt adaptation and large held-out evaluations: **NOT_RUN**. The existing source result tree and checkpoint files were not written. No server connection or long training run was launched.
