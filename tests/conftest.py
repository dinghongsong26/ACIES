"""Test-suite determinism.

Several simulators place items on a RANDOM free cell each spawn (simulator.warehouse goods,
simulator.patrol alerts, via `random.choice`). A few sim tests assert "everything is delivered /
cleared within a step budget", which occasionally failed on an unlucky random layout even though the
controller logic was correct. Seeding the RNGs before every test makes the whole suite deterministic.
Tests that set their own seed inside the test body still override this (the body runs after the fixture).
"""
import random

import pytest

try:
    import numpy as _np
except ImportError:
    _np = None


@pytest.fixture(autouse=True)
def _seed_rngs():
    random.seed(0)
    if _np is not None:
        _np.random.seed(0)
    yield
