from __future__ import annotations

import itertools

from make_oversmoothing_data import (
    INITIAL_VALUES,
    NUM_STEPS,
    render_table,
    spread,
    trajectory,
)


def test_spread_shrinks_every_step_to_near_zero() -> None:
    spreads = [spread(state) for state in trajectory()]
    assert all(later < earlier for earlier, later in itertools.pairwise(spreads))
    assert spreads[-1] < 0.05 * spreads[0]


def test_trajectory_starts_from_initial_values_and_has_all_steps() -> None:
    states = trajectory()
    assert states[0] == INITIAL_VALUES
    assert len(states) == NUM_STEPS + 1


def test_output_is_deterministic() -> None:
    assert render_table(trajectory()) == render_table(trajectory())
