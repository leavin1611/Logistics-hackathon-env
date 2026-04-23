"""
LogisticsShipmentRL — Core Environment (Standalone)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict, List
from scenarios import get_scenario, ROUTES, CARRIERS
from grader import compute_reward
from models import (
    LogisticsObservation, LogisticsAction, LogisticsState,
    ShipmentStatus, DisruptionEvent, RouteOption
)


class LogisticsEnvironment:
    """
    Subclasses OpenEnv's base Environment to provide the REST endpoints seamlessly.
    """
    def __init__(self):
        self.state_data = {}

    def setup(self, **kwargs) -> Dict[str, Any]:
        """Called once at environment server boot. Load static assets here."""
        return {"routes": ROUTES, "carriers": CARRIERS}

    async def _reset(self, scenario_id: str = "SCN-001", seed: int | None = None) -> Dict[str, Any]:
        """Generates the initial observation."""
        scenario = get_scenario(scenario_id, seed)
        
        # Hydrate initial internal state
        self.state_data = {
            "step": 1,
            "max_steps": 5,
            "scenario": scenario,
            "shipments": scenario.shipments.copy(),
            "disruptions": scenario.disruptions.copy(),
            "routes": ROUTES,
            "cumulative_reward": 0.0,
            "delay_saved": 0.0,
            "cost_usd": 0.0
        }
        
        obs = self._build_observation("Initial layout. Network is experiencing disruptions.")
        return obs.model_dump()

    async def _step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply agent routing logic and advance the network state forward 1 hour."""
        action = LogisticsAction(**action_dict)
        self.state_data["step"] += 1
        done = self.state_data["step"] > self.state_data["max_steps"]
        
        # --- Simplified Simulation ---
        # 1. Apply reroutes
        additional_cost = 0.0
        saved_delay_hours = 0.0
        for s_id, reroute in action.rerouting_decisions.items():
            for s in self.state_data["shipments"]:
                if s["shipment_id"] == s_id:
                    s["assigned_route"] = reroute.new_route
                    if reroute.new_carrier:
                        s["assigned_carrier"] = reroute.new_carrier
                    # Simplification: Rerouting generally saves X hours but costs Y dollars
                    saved_delay_hours += 2.0 
                    additional_cost += 150.0

        # 2. Advance Time
        for s in self.state_data["shipments"]:
            if s["current_status"] != "delivered":
                s["sla_buffer_hours"] -= 1.0  # hour passed
                
                # If they passed SLA, mark delayed
                if s["sla_buffer_hours"] < 0:
                    s["current_status"] = "delayed"
                    s["current_delay_hours"] += 1.0
        
        # 3. Dynamic Events (Hardcoded for demo: clear a disruption on step 3)
        field_updates = []
        if self.state_data["step"] == 3:
            if len(self.state_data["disruptions"]) > 0:
                removed = self.state_data["disruptions"].pop()
                field_updates.append(f"[FIELD UPDATE] {removed['event_id']} at {removed['location']} has been cleared.")
        
        # 4. Grading
        # Construct metrics from internal state
        metric_shipments = [ShipmentStatus(**s) for s in self.state_data["shipments"]]
        grader_context = {
            "baseline_delay": 10.0,
            "new_delay": max(0.0, 10.0 - saved_delay_hours),
            "base_cost": 1000.0,
            "new_cost": 1000.0 + additional_cost,
            "penalties_avoided": 3000.0 if saved_delay_hours > 0 else 0.0,
            "agent_shipments": metric_shipments
        }
        
        reward_val, breakdown = compute_reward(action_dict, grader_context)
        self.state_data["cumulative_reward"] += reward_val
        self.state_data["delay_saved"] += saved_delay_hours
        self.state_data["cost_usd"] += additional_cost
        
        # 5. Build Result
        obs = self._build_observation("Action applied. 1 hour elapsed.", field_updates)
        obs.previous_action_feedback = f"Re-routed {len(action.rerouting_decisions)} shipments."
        obs.previous_reward = reward_val
        obs.previous_reward_breakdown = breakdown

        return {
            "observation": obs.model_dump(),
            "reward": reward_val,
            "done": done,
            "info": {"sla_compliance": breakdown["sla_compliance"]}
        }

    async def _state(self) -> Dict[str, Any]:
        """Returns the final global metadata once the episode ends."""
        return LogisticsState(
            episode_id="EP-1234",
            step_count=self.state_data["step"],
            max_steps=self.state_data["max_steps"],
            done=self.state_data["step"] > self.state_data["max_steps"],
            scenario_id=self.state_data["scenario"].scenario_id,
            total_shipments=len(self.state_data["shipments"]),
            total_delay_saved_hours=self.state_data["delay_saved"],
            total_rerouting_cost_usd=self.state_data["cost_usd"],
            sla_violations_count=len([s for s in self.state_data["shipments"] if s["sla_buffer_hours"] < 0]),
            sla_compliance_rate=0.8,
            cumulative_reward=self.state_data["cumulative_reward"],
            reward_breakdown={}
        ).model_dump()
        
    def _build_observation(self, status: str, field_updates: List[str] = None) -> LogisticsObservation:
        """Helper building the Observation dump."""
        return LogisticsObservation(
            scenario_id=self.state_data["scenario"].scenario_id,
            scenario_title=self.state_data["scenario"].title,
            network_snapshot=status,
            active_shipments=[ShipmentStatus(**s) for s in self.state_data["shipments"]],
            total_shipments=len(self.state_data["shipments"]),
            delayed_shipments=len([s for s in self.state_data["shipments"] if s["sla_buffer_hours"] < 0]),
            sla_at_risk_count=len([s for s in self.state_data["shipments"] if 0 <= s["sla_buffer_hours"] <= 2]),
            disruption_events=[DisruptionEvent(**d) for d in self.state_data["disruptions"]],
            active_disruptions_count=len(self.state_data["disruptions"]),
            available_routes=[RouteOption(**r) for r in self.state_data["routes"].values()],
            weather_forecast=self.state_data["scenario"].weather_forecast,
            carrier_availability=CARRIERS,
            current_total_delay_hours=10.0,
            sla_violations=[s["shipment_id"] for s in self.state_data["shipments"] if s["sla_buffer_hours"] < 0],
            on_time_shipments=len([s for s in self.state_data["shipments"] if s["sla_buffer_hours"] >= 0]),
            step_number=self.state_data["step"],
            max_steps=self.state_data["max_steps"],
            episode_done=self.state_data["step"] > self.state_data["max_steps"],
            previous_action_feedback="Waiting for agent action.",
            previous_reward=0.0,
            previous_reward_breakdown={},
            cumulative_reward=self.state_data["cumulative_reward"],
            total_delay_saved_hours=self.state_data["delay_saved"],
            total_rerouting_cost_usd=self.state_data["cost_usd"],
            sla_compliance_rate=0.8,
            field_updates=field_updates or []
        )
