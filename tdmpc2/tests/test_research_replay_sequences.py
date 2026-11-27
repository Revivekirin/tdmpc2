"""Sequence lookup stays episode-safe and leaves legacy transition sampling alone."""

import random
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tdmpc2"))
from research.dynamics_shift.replay import AdaptationReplayBuffer


def record(episode, step, global_step, *, truncated=False):
    return dict(episode=episode, episode_step=step, global_step=global_step,
                terminated=False, truncated=truncated)


def test_contiguous_windows_never_cross_reset_or_gap():
    replay = AdaptationReplayBuffer([
        record(0, 0, 1), record(0, 1, 2), record(0, 2, 3, truncated=True),
        record(1, 0, 4), record(1, 1, 5), record(1, 3, 7),
    ], maxlen=8)
    before = list(replay)
    pairs = replay.contiguous_sequences(2)
    triples = replay.contiguous_sequences(3)
    assert [[item['global_step'] for item in group] for group in pairs] == [[1, 2], [2, 3], [4, 5]]
    assert [[item['global_step'] for item in group] for group in triples] == [[1, 2, 3]]
    assert replay.contiguous_sequences(7) == []
    assert list(replay) == before


def test_sequence_sampling_is_uniform_without_replacement_and_rng_owned_by_caller():
    replay = AdaptationReplayBuffer([record(0, i, i + 1) for i in range(6)], maxlen=6)
    rng = random.Random(42)
    before = rng.getstate()
    candidates = replay.contiguous_sequences(3)
    assert rng.getstate() == before
    expected_rng = random.Random(42)
    expected = expected_rng.sample(candidates, 2)
    selected = replay.sample_sequences(rng, 2, 3)
    assert selected == expected and len({window[0]['global_step'] for window in selected}) == 2
    assert rng.getstate() == expected_rng.getstate()
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            replay.contiguous_sequences(invalid)
    with pytest.raises(ValueError):
        replay.sample_sequences(rng, 5, 3)


def test_legacy_sample_bytes_and_rng_state_are_unchanged():
    replay = AdaptationReplayBuffer([record(0, i, i + 1) for i in range(5)], maxlen=5)
    expected_rng, actual_rng = random.Random(713), random.Random(713)
    expected = expected_rng.sample(list(replay), 3)
    observed = replay.sample(actual_rng, 3)
    assert observed == expected
    assert actual_rng.getstate() == expected_rng.getstate()
    tuple_replay = AdaptationReplayBuffer([(1, 2, 3)], maxlen=2)
    with pytest.raises(ValueError, match='metadata'):
        tuple_replay.contiguous_sequences(1)


def test_capacity_eviction_and_termination_do_not_create_false_windows():
    replay = AdaptationReplayBuffer(maxlen=4)
    for row in (record(0, 0, 1), record(0, 1, 2),
                dict(record(0, 2, 3), terminated=True),
                record(1, 0, 4), record(1, 1, 5)):
        replay.append(row)
    assert [item['global_step'] for item in replay] == [2, 3, 4, 5]
    assert [[item['global_step'] for item in group]
            for group in replay.contiguous_sequences(2)] == [[2, 3], [4, 5]]
    assert replay.contiguous_sequences(3) == []
