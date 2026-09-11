"""Activation policies. No model, no torch."""

import pytest

from engine.core.types.diffusion import AdapterState
from engine.core.types.errors import ConfigError
from engine.routing.phase import Phase, PhaseRouter, PhaseSchedule


def test_static_is_live_throughout():
    schedule = PhaseSchedule.static("a")
    assert schedule.active_at(0.0) == {"a": 1.0}
    assert schedule.active_at(0.99) == {"a": 1.0}


def test_split_switches_at_the_boundary():
    schedule = PhaseSchedule.split("early", "late", at=0.5)
    assert schedule.active_at(0.49) == {"early": 1.0}
    assert schedule.active_at(0.5) == {"late": 1.0}


def test_overlapping_phases_sum():
    schedule = PhaseSchedule(Phase(0.0, 1.0, {"a": 0.7}), Phase(0.0, 1.0, {"a": 0.4}))
    assert schedule.active_at(0.1) == pytest.approx({"a": 1.1})


def test_unordered_interval_rejected():
    with pytest.raises(ConfigError):
        Phase(0.6, 0.4, {"a": 1.0})


def test_boundary_inside_a_block_is_rejected():
    schedule = PhaseSchedule.split("a", "b", at=0.3)
    with pytest.raises(ConfigError, match="falls inside a block"):
        schedule.validate_against_blocks(steps=64, blocks=2)


def test_aligned_boundary_passes():
    PhaseSchedule.split("a", "b", at=0.5).validate_against_blocks(steps=64, blocks=2)


def test_router_mutates_shared_state():
    state = AdapterState()
    router = PhaseRouter(state, PhaseSchedule.split("a", "b"))
    router.at(0, 10)
    assert state.active == {"a": 1.0}
    router.at(9, 10)
    assert state.active == {"b": 1.0}


def test_router_restores_on_exit():
    state = AdapterState()
    state.set("kept")
    with PhaseRouter(state, PhaseSchedule.static("temp")) as router:
        router.at(0, 4)
        assert state.active == {"temp": 1.0}
    assert state.active == {"kept": 1.0}
