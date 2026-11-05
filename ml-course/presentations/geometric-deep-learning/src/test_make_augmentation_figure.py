from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("make_augmentation_figure.py")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("make_augmentation_figure", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_image_is_in_the_deck_assets(generator) -> None:
    assert generator.SOURCE_IMAGE.is_file()
    assert generator.SOURCE_IMAGE.parents[1] == MODULE_PATH.parents[1] / "assets"


def test_generator_writes_every_figure_deterministically(generator, tmp_path: Path) -> None:
    first = generator.generate_figures(tmp_path / "first")
    second = generator.generate_figures(tmp_path / "second")

    assert tuple(path.name for path in first) == generator.EXPECTED_FILENAMES
    assert all(path.stat().st_size > 0 for path in first)
    assert tuple(map(_file_hash, first)) == tuple(map(_file_hash, second))
