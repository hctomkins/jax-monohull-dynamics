import pytest
from forces.force_utils import moments_about


def test_moments_about():
    assert moments_about(force=(1, 0), at=(1, 1), about=(0, 0)) < 0
    assert moments_about(force=(0, 1), at=(-1, 0), about=(0, 0)) < 0
    assert moments_about(force=(-1, -1), at=(-1, 0), about=(0, 0)) > 0
    assert moments_about(force=(0, -1), at=(-1, 0), about=(0, 0)) > 0