"""Repeated neighbor averaging on the five-node example graph of the deck.

Each step replaces a node's value by the mean over itself and its neighbors
(the row-normalized propagation with self-loops). The script writes one row per
step so the slide can plot every node and the spread between nodes.
"""

from __future__ import annotations

from pathlib import Path

NUM_NODES = 5
NUM_STEPS = 10
# 0-indexed edges of the deck's example graph: 1-2, 2-3, 3-5, 5-4, 4-1, 2-4, 3-4.
EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 4), (4, 3), (3, 0), (1, 3), (2, 3))
# Distinct starting values so the nodes are clearly distinguishable at step 0.
INITIAL_VALUES: tuple[float, ...] = (3.0, 1.0, -1.0, 2.0, -3.0)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "generated" / "oversmoothing.dat"


def neighborhoods() -> list[list[int]]:
    """Each node's own index followed by its neighbors (self-loop included)."""
    groups = [[node] for node in range(NUM_NODES)]
    for u, v in EDGES:
        groups[u].append(v)
        groups[v].append(u)
    return groups


def average_step(values: tuple[float, ...], groups: list[list[int]]) -> tuple[float, ...]:
    return tuple(sum(values[j] for j in group) / len(group) for group in groups)


def trajectory(num_steps: int = NUM_STEPS) -> list[tuple[float, ...]]:
    groups = neighborhoods()
    states = [INITIAL_VALUES]
    for _ in range(num_steps):
        states.append(average_step(states[-1], groups))
    return states


def spread(values: tuple[float, ...]) -> float:
    return max(values) - min(values)


def render_table(states: list[tuple[float, ...]]) -> str:
    header = "k " + " ".join(f"n{i + 1}" for i in range(NUM_NODES)) + " spread"
    rows = [
        f"{k} " + " ".join(f"{x:.4f}" for x in state) + f" {spread(state):.4f}"
        for k, state in enumerate(states)
    ]
    return "\n".join([header, *rows]) + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(render_table(trajectory()))


if __name__ == "__main__":
    main()
