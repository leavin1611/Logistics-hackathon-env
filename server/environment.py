"""
Logistics Shipment RL Environment
===================================
Meta PyTorch OpenEnv Hackathon — Real-World Task Simulation

This implements the strict OpenEnv `Environment` interface with pure Pydantic types
for Observation, Action, and State. No "simulated" MCP abstractions are used here.
"""

import copy
import random
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from openenv.core.env import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

# ---------------------------------------------------------------------------
# Shared route/carrier data
# ---------------------------------------------------------------------------
ROUTES = {
    "R1": {"name": "NH-48 Express (Mumbai–Pune)",          "origin": "Mumbai", "destination": "Pune",  "hours": 3.5, "cost": 120, "congestion": "heavy",    "available": True},
    "R2": {"name": "Western Highway Alt (Mumbai–Pune)",    "origin": "Mumbai", "destination": "Pune",  "hours": 4.0, "cost": 105, "congestion": "light",    "available": True},
    "R3": {"name": "NH-44 North Corridor (Delhi–Agra)",    "origin": "Delhi",  "destination": "Agra",  "hours": 4.5, "cost": 160, "congestion": "moderate", "available": True},
    "R4": {"name": "Yamuna Expressway (Delhi–Agra Alt)",   "origin": "Delhi",  "destination": "Agra",  "hours": 3.2, "cost": 175, "congestion": "clear",    "available": True},
    "R5": {"name": "Chennai–Bangalore NH-48",              "origin": "Chennai","destination": "Bangalore","hours": 5.0,"cost": 200,"congestion": "heavy",   "available": True},
    "R6": {"name": "Bangalore Alt Bypass",                 "origin": "Chennai","destination": "Bangalore","hours": 5.5,"cost": 185,"congestion": "light",   "available": True},
}

CARRIERS = ["FastFreight", "SpeedLane", "IndiaFreight", "CoastCargo", "NorthStar", "BlueLine"]

# ---------------------------------------------------------------------------
# Task Definitions (Easy / Medium / Hard)
# ---------------------------------------------------------------------------
TASKS = {
    "TASK-EASY": {
        "name": "Port Backlog Clearance",
        "description": "Single disruption at JNPT. Clear 2 delayed shipments within 3 turns.",
        "max_turns": 3,
        "baseline_delay": 5.5,
        "disruptions": ["Port congestion at JNPT (Mumbai): 4h backlog on R1"],
        "shipments": [
            {"id": "SHIP-001", "cargo": "Fresh Vegetables (perishable)", "origin": "Mumbai", "destination": "Pune", "carrier": "FastFreight", "route": "R1", "sla_buffer_h": -1.0, "delay_h": 3.0, "value": 12000, "priority": False, "status": "DELAYED", "notes": "Stuck at gate"},
            {"id": "SHIP-002", "cargo": "Auto Parts", "origin": "Mumbai", "destination": "Pune", "carrier": "SpeedLane", "route": "R1", "sla_buffer_h": 0.5, "delay_h": 2.5, "value": 31000, "priority": False, "status": "IN_TRANSIT", "notes": "Moving slowly"},
        ],
    },
    "TASK-MEDIUM": {
        "name": "Mumbai Crisis Coordination",
        "description": "Port congestion + accident + carrier strike. Manage 4 shipments over 5 turns.",
        "max_turns": 5,
        "baseline_delay": 11.0,
        "disruptions": ["Port congestion at JNPT: 6h backlog", "Khopoli accident: R1 +2.5h delay", "Carrier strike (FastFreight): 40% loss"],
        "shipments": [
            {"id": "SHIP-001", "cargo": "Fresh Pharmaceuticals (perishable)", "origin": "Mumbai", "destination": "Pune", "carrier": "FastFreight", "route": "R1", "sla_buffer_h": -2.0, "delay_h": 3.5, "value": 45000, "priority": False, "status": "DELAYED", "notes": "Reefer stuck"},
            {"id": "SHIP-002", "cargo": "Consumer Electronics", "origin": "Delhi", "destination": "Agra", "carrier": "NorthStar", "route": "R3", "sla_buffer_h": 1.5, "delay_h": 0.0, "value": 28000, "priority": False, "status": "IN_TRANSIT", "notes": "On time"},
            {"id": "SHIP-003", "cargo": "Server Hardware (high-value)", "origin": "Mumbai", "destination": "Pune", "carrier": "SpeedLane", "route": "R1", "sla_buffer_h": -4.0, "delay_h": 5.0, "value": 180000, "priority": True, "status": "DELAYED", "notes": "Customs blocked"},
            {"id": "SHIP-004", "cargo": "Industrial Chemicals (hazmat)", "origin": "Mumbai", "destination": "Pune", "carrier": "FastFreight", "route": "R1", "sla_buffer_h": -1.0, "delay_h": 2.5, "value": 22000, "priority": False, "status": "DELAYED", "notes": "Queued"},
        ],
    },
    "TASK-HARD": {
        "name": "Multi-Port Network Collapse",
        "description": "Simultaneous failures at 3 ports + weather event. 7 shipments, 7 turns.",
        "max_turns": 7,
        "baseline_delay": 28.0,
        "disruptions": ["JNPT CLOSED", "Chennai Port: 50% capacity", "BlueLine bankruptcy: stranded shipments", "Cold chain failure"],
        "shipments": [
            {"id": "SHIP-001", "cargo": "COVID Vaccines", "origin": "Mumbai", "destination": "Pune", "carrier": "BlueLine", "route": "R1", "sla_buffer_h": -6.0, "delay_h": 8.0, "value": 950000, "priority": True, "status": "DELAYED", "notes": "Stranded"},
            {"id": "SHIP-002", "cargo": "Election Ballots", "origin": "Delhi", "destination": "Agra", "carrier": "BlueLine", "route": "R3", "sla_buffer_h": -3.0, "delay_h": 4.0, "value": 0, "priority": True, "status": "DELAYED", "notes": "CRITICAL"},
            {"id": "SHIP-003", "cargo": "Surgical Equipment", "origin": "Chennai", "destination": "Bangalore", "carrier": "CoastCargo", "route": "R5", "sla_buffer_h": -2.0, "delay_h": 5.0, "value": 340000, "priority": False, "status": "DELAYED", "notes": "Suspended soon"},
            {"id": "SHIP-004", "cargo": "Petroleum (hazmat)", "origin": "Mumbai", "destination": "Pune", "carrier": "FastFreight", "route": "R1", "sla_buffer_h": -1.0, "delay_h": 3.0, "value": 88000, "priority": False, "status": "DELAYED", "notes": "Hazmat required"},
            {"id": "SHIP-005", "cargo": "Consumer Electronics", "origin": "Delhi", "destination": "Agra", "carrier": "NorthStar", "route": "R3", "sla_buffer_h": 2.0, "delay_h": 2.0, "value": 120000, "priority": False, "status": "IN_TRANSIT", "notes": "Hub cyber incident"},
            {"id": "SHIP-006", "cargo": "Blood Bank Supplies", "origin": "Chennai", "destination": "Bangalore", "carrier": "IndiaFreight", "route": "R5", "sla_buffer_h": -4.0, "delay_h": 6.0, "value": 75000, "priority": False, "status": "DELAYED", "notes": "Reefer failure"},
            {"id": "SHIP-007", "cargo": "Agricultural Seeds", "origin": "Mumbai", "destination": "Pune", "carrier": "SpeedLane", "route": "R1", "sla_buffer_h": -8.0, "delay_h": 0.0, "value": 15000, "priority": False, "status": "CRITICAL", "notes": "Spoils in 24h"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Strict Pydantic Models for OpenEnv Compliance
# ---------------------------------------------------------------------------

class LogisticsAction(Action):
    """The strict typed action model representing what the AI can do."""
    action_type: Literal["get_network_status", "reroute_shipment", "set_priority", "communicate_eta", "escalate", "end_turn"] = Field(
        ..., description="Which function to execute"
    )
    shipment_id: Optional[str] = Field(None, description="Shipment ID if applicable")
    new_route: Optional[str] = Field(None, description="New route ID (for reroute)")
    new_carrier: Optional[str] = Field(None, description="New carrier (for reroute)")
    reason: Optional[str] = Field(None, description="Reason for action")
    message: Optional[str] = Field(None, description="Customer ETA message")
    priority_ids: Optional[List[str]] = Field(None, description="Shipment IDs to prioritize")


class LogisticsObservation(Observation):
    """The strict typed observation model representing what the AI sees."""
    task: str = Field(..., description="Current task ID")
    turn: int = Field(..., description="Current turn number")
    max_turns: int = Field(..., description="Maximum turns for this task")
    disruptions: List[str] = Field(default_factory=list, description="Active disruptions")
    shipments: List[Dict[str, Any]] = Field(default_factory=list, description="All shipments")
    feedback: Optional[str] = Field(None, description="Feedback from last action")
    incremental_reward: float = Field(0.0, description="Reward gained on the exact last step")
    turn_reward: Optional[float] = Field(None, description="Total reward for completed turn")
    cumulative_reward: float = Field(0.0, description="Total running reward")
    reward_breakdown: Optional[Dict[str, float]] = Field(None, description="Detailed score split")
    route_load: Dict[str, float] = Field(default_factory=dict, description="Current background traffic load (0.0 - 1.0)")


class LogisticsState(State):
    """The strict typed state model representing the internal environment tracking."""
    task_id: str = "TASK-MEDIUM"
    turn: int = 0
    cumulative_reward: float = 0.0
    incremental_reward: float = 0.0
    actions_this_turn: int = 0
    turn_committed: bool = False
    
    # Internal arrays
    shipments: List[Dict[str, Any]] = Field(default_factory=list)
    disruptions: List[str] = Field(default_factory=list)
    priority_set: List[str] = Field(default_factory=list)
    communications: Dict[str, str] = Field(default_factory=dict)
    escalations: List[str] = Field(default_factory=list)
    reroutings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    # Theme #1: Multi-Agent / Resource Scarcity
    route_load: Dict[str, float] = Field(default_factory=dict, description="Current background load per route (0.0 to 1.0)")


# ---------------------------------------------------------------------------
# Scoring Helpers
# ---------------------------------------------------------------------------
def _score_message(message: str) -> float:
    txt = message.lower()
    score = 0.0
    if any(w in txt for w in ["sorry", "apologis", "apolog", "regret"]):
        score += 0.20
    if any(w in txt for w in ["eta", "arrive", "delivery", "reschedule", "expect", "pm", "am", "hour"]):
        score += 0.40
    if any(w in txt for w in ["due to", "because", "weather", "port", "delay", "congestion", "strike", "customs"]):
        score += 0.30
    if len(message) > 80:
        score += 0.10
    return min(1.0, score)

def _message_feedback(score: float) -> str:
    if score >= 0.9: return "Excellent empathetic message with clear ETA."
    elif score >= 0.7: return "Good, but lacks either apology or specific cause/ETA."
    else: return "Poor. Include apology, cause of delay, and specific ETA next time."


# ---------------------------------------------------------------------------
# Environment Engine
# ---------------------------------------------------------------------------

class LogisticsShipmentEnvironment(Environment[LogisticsAction, LogisticsObservation, LogisticsState]):
    """
    Pure Python explicit RL Environment honoring the Hackathon `openenv` spec strictly.
    No "MCP wrapper" translation - direct Action models to Observation models.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self._state = LogisticsState(episode_id=str(uuid4()), step_count=0)
        self._task_def = TASKS["TASK-MEDIUM"]

    @property
    def state(self) -> LogisticsState:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "TASK-MEDIUM",
        **kwargs: Any,
    ) -> LogisticsObservation:
        
        self._task_def = TASKS.get(task_id, TASKS["TASK-MEDIUM"])
        
        # Deepcopy the data so we can mutate it this episode
        initial_shipments = copy.deepcopy(self._task_def["shipments"])
        
        if seed is not None:
            random.seed(seed)
            for s in initial_shipments:
                s["sla_buffer_h"] += round(random.uniform(-0.5, 0.5), 1)

        self._state = LogisticsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            shipments=initial_shipments,
            disruptions=list(self._task_def["disruptions"])
        )

        return self._build_observation("Environment reset complete.")

    def step(self, action: LogisticsAction, timeout_s: Optional[float] = None, **kwargs: Any) -> LogisticsObservation:
        
        self._state.incremental_reward = 0.0 # reset instantaneous counter
        self._state.step_count += 1
        
        feedback = ""
        done = False
        
        # Theme #1: Simulate other agents' actions (background traffic)
        if action.action_type != "get_network_status":
            self._simulate_background_traffic()
            
        if action.action_type == "get_network_status":
            self._state.actions_this_turn += 1
            self._state.incremental_reward = 0.01
            feedback = "Network status fetched."
            
        elif action.action_type == "reroute_shipment":
            feedback = self._handle_reroute(action)
            self._state.actions_this_turn += 1
            
        elif action.action_type == "set_priority":
            feedback = self._handle_priority(action)
            self._state.actions_this_turn += 1
            
        elif action.action_type == "communicate_eta":
            feedback = self._handle_communication(action)
            self._state.actions_this_turn += 1
            
        elif action.action_type == "escalate":
            feedback = self._handle_escalate(action)
            self._state.actions_this_turn += 1
            
        elif action.action_type == "end_turn":
            feedback, done = self._handle_end_turn()
            
        else:
            feedback = f"Unknown action: {action.action_type}"

        self._state.cumulative_reward += self._state.incremental_reward

        obs = self._build_observation(feedback)
        obs.done = done
        obs.reward = self._state.incremental_reward
        return obs

    def _build_observation(self, feedback: str) -> LogisticsObservation:
        return LogisticsObservation(
            task=self._state.task_id,
            turn=self._state.turn,
            max_turns=self._task_def["max_turns"],
            disruptions=self._state.disruptions,
            shipments=self._state.shipments,
            feedback=feedback,
            incremental_reward=round(self._state.incremental_reward, 3),
            cumulative_reward=round(self._state.cumulative_reward, 3),
            route_load=self._state.route_load,
            done=False,
            reward=0.0
        )

    # -------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------

    def _handle_reroute(self, action: LogisticsAction) -> str:
        s_id = action.shipment_id
        new_r = action.new_route
        if not s_id or not new_r: return "Error: Missing shipment_id or new_route"
        
        shipment = next((s for s in self._state.shipments if s["id"] == s_id), None)
        if not shipment: return f"Error: Shipment {s_id} not found."
        if new_r not in ROUTES: return f"Error: Route {new_r} not valid."
        if shipment["route"] == new_r: return "Error: Already on that route."
        
        # Theme #1: Check Capacity
        current_load = self._state.route_load.get(new_r, 0.2)
        if current_load > 0.85:
            self._state.incremental_reward -= 0.05
            return f"Error: Route {new_r} is at critical capacity ({(current_load*100):.1f}%). Reroute failed. Other agents are saturating this corridor."

        old_cong = ROUTES.get(shipment["route"], {}).get("congestion", "unknown")
        new_cong = ROUTES[new_r]["congestion"]
        
        savings_map = {("heavy", "light"): 2.5, ("heavy", "clear"): 3.0, ("heavy", "moderate"): 1.5, ("moderate", "light"): 1.0, ("moderate", "clear"): 1.5}
        savings = savings_map.get((old_cong, new_cong), 0.5)

        shipment["route"] = new_r
        if action.new_carrier: shipment["carrier"] = action.new_carrier
        shipment["delay_h"] = max(0.0, shipment["delay_h"] - savings)
        if shipment["delay_h"] == 0: shipment["status"] = "IN_TRANSIT"

        urgency_bonus = 0.05 if shipment["sla_buffer_h"] < 0 else 0.0
        step_reward = min(0.15, savings / 20.0) + urgency_bonus
        self._state.incremental_reward += step_reward
        
        return f"Rerouted {s_id} to {new_r}. Delay saved: {savings}h. Immediate reward: {step_reward:.3f}."

    def _handle_priority(self, action: LogisticsAction) -> str:
        s_ids = action.priority_ids
        if not s_ids: return "Error: priority_ids missing."
        if len(s_ids) > 3: return "Error: Max 3 priority shipments allowed."
        
        self._state.priority_set = s_ids
        for s in self._state.shipments:
            s["priority"] = s["id"] in s_ids
            
        correct = [sid for sid in s_ids if any(s["id"] == sid and (s["value"] > 50000 or s["sla_buffer_h"] < 0) for s in self._state.shipments)]
        reward = len(correct) * 0.03
        self._state.incremental_reward += reward
        return f"Priorities assigned to {s_ids}. Immediate reward: {reward:.3f}."

    def _handle_communication(self, action: LogisticsAction) -> str:
        if not action.shipment_id or not action.message: return "Error: missing shipment_id or message."
        
        # In a real environment, we would log this for a dashboard or trigger an external mock API.
        self._state.communications[action.shipment_id] = action.message
        
        score = _score_message(action.message)
        shipment = next((s for s in self._state.shipments if s["id"] == action.shipment_id), None)
        bonus = 0.02 if shipment and shipment["sla_buffer_h"] < 0 else 0.0
        
        step_rew = (score * 0.10) + bonus
        self._state.incremental_reward += step_rew
        return f"Message logged for {action.shipment_id}. Reward: {step_rew:.3f}. Feedback: {_message_feedback(score)}"

    def _handle_escalate(self, action: LogisticsAction) -> str:
        if not action.shipment_id: return "Error: missing shipment_id."
        if action.shipment_id not in self._state.escalations:
            self._state.escalations.append(action.shipment_id)
            self._state.incremental_reward -= 0.1
            return f"{action.shipment_id} escalated to human. Penalty -0.1 applied."
        return "Already escalated."

    def _handle_end_turn(self) -> tuple[str, bool]:
        if self._state.turn_committed:
            return "Turn already committed.", False
            
        self._state.turn_committed = True
        
        # Compute multi-dimensional turn reward
        total_delay = sum(s["delay_h"] for s in self._state.shipments)
        baseline = self._task_def["baseline_delay"]
        delay_saved = max(0.0, baseline - total_delay)
        delay_score = min(1.0, delay_saved / (baseline * 0.8))

        on_time = sum(1 for s in self._state.shipments if s["sla_buffer_h"] >= 0)
        sla_score = on_time / len(self._state.shipments)

        delayed = [s for s in self._state.shipments if s["sla_buffer_h"] < 0]
        comm_delayed = {sid for sid in self._state.communications if any(s["id"] == sid and s["sla_buffer_h"] < 0 for s in self._state.shipments)}
        coverage = len(comm_delayed) / len(delayed) if delayed else 1.0
        quality = sum(_score_message(m) for m in self._state.communications.values()) / len(self._state.communications) if self._state.communications else 0.0
        comm_score = (0.5 * coverage) + (0.5 * quality)

        escalation_penalty = len(self._state.escalations) * 0.1
        esc_score = max(0.0, 1.0 - escalation_penalty)
        act_bonus = 0.05 if self._state.actions_this_turn >= 3 else 0.0

        turn_rew = min(1.0, (0.40 * delay_score + 0.30 * sla_score + 0.20 * comm_score + 0.10 * esc_score + act_bonus))
        
        self._state.incremental_reward = turn_rew
        self._state.turn += 1

        for s in self._state.shipments:
            s["sla_buffer_h"] -= 1.0
            if s["sla_buffer_h"] < 0 and s["status"] == "IN_TRANSIT":
                s["status"] = "DELAYED"

        done = self._state.turn >= self._task_def["max_turns"]

        # Reset turn state
        self._state.reroutings.clear()
        self._state.priority_set.clear()
        self._state.communications.clear()
        self._state.escalations.clear()
        self._state.actions_this_turn = 0
        self._state.turn_committed = False

        msg = f"Turn committed! Score: {turn_rew:.3f} | Delay: {delay_score:.2f}, SLA: {sla_score:.2f}, Comm: {comm_score:.2f}, Esc: {esc_score:.2f}"
        if done: msg += f" | 🏁 Episode Complete!"
        return msg, done

    def _simulate_background_traffic(self):
        """Simulate the actions of other agents in the network (Theme #1)."""
        # Routes have different base loads and volatility
        for r_id in ROUTES:
            base = 0.3 if ROUTES[r_id]["congestion"] == "clear" else 0.6
            # Random volatility to simulate other agents' spikes
            self._state.route_load[r_id] = min(1.0, max(0.0, base + random.uniform(-0.2, 0.4)))
