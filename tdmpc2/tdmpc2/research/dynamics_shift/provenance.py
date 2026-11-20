"""Explicitly audited runner source identities from before the loop migration.

These hashes remain historical provenance.  They do not replace current source
hashes in new artifacts and cannot waive a model, environment or config change.
"""

from collections.abc import Mapping


PRE_RUNNER_MIGRATION_SHA256 = {
    'experiments/nsgym/scripts/run_dynamics_only_ttt.py':
        '55650ca64d8796a0178a60092d6f626fce3053a9829e5c0f3b174e9126c0a2dc',
    'experiments/nsgym/scripts/run_dynamics_q.py':
        '46d04214d16db99cb2f299e9683dbe56418d96555ef81e5bfca9a9fdc77c4728',
    'experiments/nsgym/scripts/run_reacher_dynamics_q.py':
        'f094e4fb763f343a011a14468a463eb83e9a0e5643ff9c4876addf00c5efa57a',
    'experiments/nsgym/scripts/run_full_joint_rl.py':
        'dc580556eb65ec930470e7d33f89c4020f7c233964eb86fc5d39962681b1ed4f',
    'experiments/nsgym/scripts/reacher_pipeline.py':
        '8db72b049ee080660f9a9ee8526b3d633812389a32db06a31d37cdb07faa60ab',
}

# Approved candidate source bytes after the controlled loop migration.  If a
# future edit changes any of these files, old manifests must be revalidated.
RUNNER_MIGRATION_SHA256 = {
    'experiments/nsgym/scripts/run_dynamics_only_ttt.py':
        '1e0dfeb0a251587cac69d1aceb231f6727ca5eb84fa36a8fa524fc8c8de9c9f8',
    'experiments/nsgym/scripts/run_dynamics_q.py':
        '57cc3c4910d190db32d6d1657c3e715aec5bd35d7a11ce879928ed617ec55ddb',
    'experiments/nsgym/scripts/run_reacher_dynamics_q.py':
        '8f00a05f21fac60ffe8255c6ef2311299069c8f364610c8719a2f39c27c4fe3a',
    'experiments/nsgym/scripts/run_full_joint_rl.py':
        'fd5b95781c2b744118dd03db2d8707f5977f54c073482544b7c8405dfed02a6a',
    'experiments/nsgym/scripts/reacher_pipeline.py':
        '1528ce13585aa3bef61ede890f844083990449a173df3672ef8e88efeb0eab86',
}


def compatible_sources(recorded: Mapping[str, str], current: Mapping[str, str]) -> bool:
    """Accept only exact hashes or a verified pre-migration runner source hash."""
    return recorded.keys() == current.keys() and all(
        recorded[path] == digest or
        (path in PRE_RUNNER_MIGRATION_SHA256 and
         recorded[path] == PRE_RUNNER_MIGRATION_SHA256[path] and
         digest == RUNNER_MIGRATION_SHA256[path])
        for path, digest in current.items())
