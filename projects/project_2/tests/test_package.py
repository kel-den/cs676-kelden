"""Tests for the public package surface in __init__.py. All offline."""

import personaforge
from personaforge import AmbiguousPersonaError, Orchestrator, __version__


def test_version_is_0_2_0():
    assert __version__ == "0.2.0"


def test_every_name_in_all_is_importable():
    for name in personaforge.__all__:
        assert hasattr(personaforge, name), f"{name} listed in __all__ but not exported"


def test_public_names_are_available_from_the_top_level():
    assert AmbiguousPersonaError.__name__ == "AmbiguousPersonaError"
    assert Orchestrator.__name__ == "Orchestrator"
