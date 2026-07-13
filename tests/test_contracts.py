"""Bootstrap package contract tests."""

import risk_agent


def test_package_marker_is_importable() -> None:
    """The research environment exposes its package marker."""
    assert risk_agent.__doc__
