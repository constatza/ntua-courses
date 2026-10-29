from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from make_supervised_figures import EXPECTED_FILENAMES, generate_figures


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SupervisedFigureTests(unittest.TestCase):
    def test_generate_figures_writes_expected_nonempty_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_figures(Path(directory))

            self.assertEqual(tuple(path.name for path in outputs), EXPECTED_FILENAMES)
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))

    def test_generate_figures_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_outputs = generate_figures(Path(first))
            second_outputs = generate_figures(Path(second))

            self.assertEqual(
                tuple(file_hash(path) for path in first_outputs),
                tuple(file_hash(path) for path in second_outputs),
            )


if __name__ == "__main__":
    unittest.main()
