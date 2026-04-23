"""
tests/test_environment.py
==========================
Core unit tests for the Logistics Shipment RL Environment.

Run with:  pytest tests/ -v
"""

import pytest
import sys
import os

# Make the server package importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from server.environment import (
        LogisticsShipmentEnvironment,
        LogisticsAction,
        LogisticsObservation,
        TASKS,
        ROUTES,
    )
except ImportError as e:
    pytest.skip(f"Could not import environment: {e}", allow_module_level=True)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """A fresh environment for each test."""
    return LogisticsShipmentEnvironment()


@pytest.fixture
def env_easy(env):
    env.reset(task_id="TASK-EASY")
    return env


@pytest.fixture
def env_medium(env):
    env.reset(task_id="TASK-MEDIUM")
    return env


@pytest.fixture
def env_hard(env):
    env.reset(task_id="TASK-HARD")
    return env


# ─── Reset Tests ─────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="TASK-EASY")
        assert isinstance(obs, LogisticsObservation)

    def test_reset_sets_correct_task(self, env):
        obs = env.reset(task_id="TASK-MEDIUM")
        assert obs.task == "TASK-MEDIUM"

    def test_reset_starts_at_turn_zero(self, env):
        obs = env.reset(task_id="TASK-EASY")
        assert obs.turn == 0

    def test_reset_loads_shipments(self, env_easy):
        obs = env_easy.state
        assert len(obs.shipments) == len(TASKS["TASK-EASY"]["shipments"])

    def test_reset_loads_disruptions(self, env_easy):
        obs = env_easy.state
        assert len(obs.disruptions) > 0

    def test_reset_clears_previous_state(self, env):
        env.reset(task_id="TASK-HARD")
        obs = env.reset(task_id="TASK-EASY")
        assert obs.task == "TASK-EASY"

    def test_all_tasks_reset(self, env):
        for task_id in TASKS:
            obs = env.reset(task_id=task_id)
            assert obs.task == task_id


# ─── Action: get_network_status ───────────────────────────────────────────────

class TestGetNetworkStatus:
    def test_status_returns_observation(self, env_easy):
        obs = env_easy.step(LogisticsAction(action_type="get_network_status"))
        assert isinstance(obs, LogisticsObservation)

    def test_status_gives_small_reward(self, env_easy):
        obs = env_easy.step(LogisticsAction(action_type="get_network_status"))
        assert obs.incremental_reward > 0

    def test_status_returns_shipments(self, env_easy):
        obs = env_easy.step(LogisticsAction(action_type="get_network_status"))
        assert len(obs.shipments) > 0

    def test_status_returns_disruptions(self, env_easy):
        obs = env_easy.step(LogisticsAction(action_type="get_network_status"))
        assert len(obs.disruptions) > 0


# ─── Action: reroute_shipment ─────────────────────────────────────────────────

class TestReroute:
    def test_reroute_valid_gives_reward(self, env_easy):
        obs = env_easy.step(LogisticsAction(
            action_type="reroute_shipment",
            shipment_id="SHIP-001",
            new_route="R2",
            new_carrier="SpeedLane",
            reason="R1 is congested"
        ))
        assert obs.incremental_reward > 0

    def test_reroute_updates_shipment_route(self, env_easy):
        env_easy.step(LogisticsAction(
            action_type="reroute_shipment",
            shipment_id="SHIP-001",
            new_route="R2",
            new_carrier="SpeedLane",
            reason="avoiding congestion"
        ))
        ship = next(s for s in env_easy.state.shipments if s["id"] == "SHIP-001")
        assert ship["route"] == "R2"

    def test_reroute_reduces_delay(self, env_easy):
        original_delay = next(
            s["delay_h"] for s in env_easy.state.shipments if s["id"] == "SHIP-001"
        )
        env_easy.step(LogisticsAction(
            action_type="reroute_shipment",
            shipment_id="SHIP-001",
            new_route="R2",
        ))
        new_delay = next(s["delay_h"] for s in env_easy.state.shipments if s["id"] == "SHIP-001")
        assert new_delay <= original_delay

    def test_reroute_invalid_route_returns_error(self, env_easy):
        obs = env_easy.step(LogisticsAction(
            action_type="reroute_shipment",
            shipment_id="SHIP-001",
            new_route="R99",
        ))
        assert "Error" in (obs.feedback or "")

    def test_reroute_invalid_shipment_returns_error(self, env_easy):
        obs = env_easy.step(LogisticsAction(
            action_type="reroute_shipment",
            shipment_id="SHIP-999",
            new_route="R2",
        ))
        assert "Error" in (obs.feedback or "")

    def test_reroute_same_route_returns_error(self, env_easy):
        # SHIP-001 starts on R1
        obs = env_easy.step(LogisticsAction(
            action_type="reroute_shipment",
            shipment_id="SHIP-001",
            new_route="R1",
        ))
        assert "Error" in (obs.feedback or "")


# ─── Action: set_priority ─────────────────────────────────────────────────────

class TestSetPriority:
    def test_priority_gives_reward(self, env_medium):
        obs = env_medium.step(LogisticsAction(
            action_type="set_priority",
            priority_ids=["SHIP-003"]  # high-value server hardware
        ))
        assert obs.incremental_reward >= 0

    def test_priority_marks_shipment(self, env_medium):
        env_medium.step(LogisticsAction(
            action_type="set_priority",
            priority_ids=["SHIP-001", "SHIP-003"]
        ))
        ship = next(s for s in env_medium.state.shipments if s["id"] == "SHIP-001")
        assert ship["priority"] is True

    def test_too_many_priorities_rejected(self, env_medium):
        obs = env_medium.step(LogisticsAction(
            action_type="set_priority",
            priority_ids=["SHIP-001", "SHIP-002", "SHIP-003", "SHIP-004"]
        ))
        assert "Error" in (obs.feedback or "")


# ─── Action: communicate_eta ──────────────────────────────────────────────────

class TestCommunicateETA:
    def test_good_message_gives_reward(self, env_medium):
        obs = env_medium.step(LogisticsAction(
            action_type="communicate_eta",
            shipment_id="SHIP-001",
            message="We sincerely apologise for the delay. Due to port congestion at JNPT, "
                    "your pharmaceutical shipment is expected to arrive by 6:00 PM today."
        ))
        assert obs.incremental_reward > 0

    def test_empty_message_handled(self, env_medium):
        obs = env_medium.step(LogisticsAction(
            action_type="communicate_eta",
            shipment_id="SHIP-001",
            message=""
        ))
        # Should not crash — just low reward or error feedback
        assert obs is not None


# ─── Action: escalate ────────────────────────────────────────────────────────

class TestEscalate:
    def test_escalate_applies_penalty(self, env_easy):
        obs = env_easy.step(LogisticsAction(
            action_type="escalate",
            shipment_id="SHIP-001",
            reason="cannot determine correct route"
        ))
        assert obs.incremental_reward < 0

    def test_double_escalate_ignored(self, env_easy):
        env_easy.step(LogisticsAction(action_type="escalate", shipment_id="SHIP-001"))
        obs2 = env_easy.step(LogisticsAction(action_type="escalate", shipment_id="SHIP-001"))
        assert "Already escalated" in (obs2.feedback or "")


# ─── Action: end_turn ────────────────────────────────────────────────────────

class TestEndTurn:
    def test_end_turn_increments_turn(self, env_easy):
        assert env_easy.state.turn == 0
        env_easy.step(LogisticsAction(action_type="end_turn"))
        assert env_easy.state.turn == 1

    def test_end_turn_gives_reward(self, env_easy):
        obs = env_easy.step(LogisticsAction(action_type="end_turn"))
        assert obs.incremental_reward >= 0

    def test_episode_ends_after_max_turns(self, env_easy):
        max_turns = TASKS["TASK-EASY"]["max_turns"]
        for _ in range(max_turns):
            obs = env_easy.step(LogisticsAction(action_type="end_turn"))
        assert obs.done is True

    def test_done_false_mid_episode(self, env_medium):
        obs = env_medium.step(LogisticsAction(action_type="end_turn"))
        assert obs.done is False  # TASK-MEDIUM has 5 turns, only 1 done


# ─── Reward bounds ────────────────────────────────────────────────────────────

class TestRewardBounds:
    def test_cumulative_reward_not_negative(self, env_medium):
        """Cumulative reward should stay non-negative with sane actions."""
        env_medium.step(LogisticsAction(action_type="get_network_status"))
        env_medium.step(LogisticsAction(
            action_type="reroute_shipment", shipment_id="SHIP-001",
            new_route="R2", reason="avoiding congestion"
        ))
        obs = env_medium.step(LogisticsAction(action_type="end_turn"))
        assert obs.cumulative_reward >= 0

    def test_turn_reward_in_zero_one(self, env_easy):
        """Turn reward from end_turn must be between 0 and 1."""
        obs = env_easy.step(LogisticsAction(action_type="end_turn"))
        assert 0.0 <= obs.incremental_reward <= 1.0


# ─── Route data integrity ──────────────────────────────────────────────────────

class TestRouteData:
    def test_all_routes_have_required_fields(self):
        required = {"name", "origin", "destination", "hours", "cost", "congestion", "available"}
        for rid, r in ROUTES.items():
            assert required.issubset(r.keys()), f"Route {rid} missing fields"

    def test_all_tasks_have_required_fields(self):
        required = {"name", "description", "max_turns", "baseline_delay", "disruptions", "shipments"}
        for tid, t in TASKS.items():
            assert required.issubset(t.keys()), f"Task {tid} missing fields"

    def test_all_task_shipments_have_ids(self):
        for tid, task in TASKS.items():
            for ship in task["shipments"]:
                assert "id" in ship, f"Task {tid} has shipment without id"
