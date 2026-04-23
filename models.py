"""
LogisticsShipmentRL — Models
Pydantic schemas describing the Action, Observation, and State API contracts.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core Entities (Sub-models)
# ---------------------------------------------------------------------------

class ShipmentStatus(BaseModel):
    shipment_id: str
    origin: str
    destination: str
    cargo_type: Literal["standard", "perishable", "hazmat", "high_value"]
    cargo_description: str
    current_location: str
    current_status: Literal["in_transit", "delayed", "at_hub", "customs_hold", "delivered"]
    assigned_carrier: str
    assigned_route: str
    estimated_arrival: str          # e.g., "+6h", or ISO timestamp
    sla_deadline: str
    sla_buffer_hours: float         # Negative value means SLA breached
    current_delay_hours: float
    value_usd: float
    is_priority: bool
    notes: str


class DisruptionEvent(BaseModel):
    event_id: str
    event_type: Literal["breakdown", "weather", "port_congestion", "customs_hold",
                         "strike", "accident", "road_closure", "capacity_shortage"]
    location: str
    affected_routes: List[str]
    affected_shipments: List[str]
    severity: Literal["low", "medium", "high", "critical"]
    estimated_duration_hours: float
    estimated_additional_delay_hours: float
    description: str
    can_be_bypassed: bool


class RouteOption(BaseModel):
    route_id: str
    route_name: str
    origin: str
    destination: str
    distance_km: float
    estimated_hours: float
    cost_usd: float
    carrier_options: List[str]
    current_congestion: Literal["clear", "light", "moderate", "high"]
    weather_risk: Literal["none", "low", "moderate", "high"]
    is_available: bool

# ---------------------------------------------------------------------------
# API Contracts: Action
# ---------------------------------------------------------------------------

class ReroutingDecision(BaseModel):
    new_route: str = Field(description="Route ID from available_routes (e.g., 'R2')")
    new_carrier: Optional[str] = Field(default=None, description="Carrier name, or None to keep existing")
    reason: str = Field(description="One-sentence justification for this re-routing")


class LogisticsAction(BaseModel):
    """
    The full action submitted by the AI agent per step.
    The agent receives a heavily delayed network and must repair it.
    """
    reasoning: str = Field(
        description="Chain-of-thought analysis explaining strategy and risk assessment."
    )
    rerouting_decisions: Dict[str, ReroutingDecision] = Field(
        default_factory=dict,
        description="Shipment ID -> Re-routing decision. Only include active changes.",
    )
    priority_shipments: List[str] = Field(
        default_factory=list,
        description="Identify up to 3 shipment IDs to expedite handling.",
    )
    customer_communications: Dict[str, str] = Field(
        default_factory=dict,
        description="ETA messages keyed by shipment ID, sent to customers.",
    )
    escalations: List[str] = Field(
        default_factory=list,
        description="Shipments passing human limit, requiring a real dispatcher.",
    )

# ---------------------------------------------------------------------------
# API Contracts: Observation
# ---------------------------------------------------------------------------

class LogisticsObservation(BaseModel):
    """Full network snapshot returned by the Environment."""
    scenario_id: str
    scenario_title: str
    network_snapshot: str                 # Natural language context summary
    
    # State tracking
    active_shipments: List[ShipmentStatus]
    total_shipments: int
    delayed_shipments: int
    sla_at_risk_count: int                # Shipments within 2 hrs of breaking SLA
    
    # Dynamics
    disruption_events: List[DisruptionEvent]
    active_disruptions_count: int
    available_routes: List[RouteOption]
    weather_forecast: str
    carrier_availability: Dict[str, int]  # Name -> trucks available
    field_updates: List[str]              # Alerts from the field
    
    # Global metrics
    current_total_delay_hours: float
    sla_violations: List[str]
    on_time_shipments: int
    
    # Turn metrics
    step_number: int
    max_steps: int
    episode_done: bool
    previous_action_feedback: str
    previous_reward: float
    previous_reward_breakdown: Dict[str, float]
    
    # Cumulative stats
    cumulative_reward: float
    total_delay_saved_hours: float
    total_rerouting_cost_usd: float
    sla_compliance_rate: float
    
    action_hint: str = (
        "Re-route shipments impacted by disruptions to available routes to save delay time. "
        "Prioritize rescuing negative sla_buffer_hours shipments and write clear customer_communications."
    )

# ---------------------------------------------------------------------------
# End-of-Episode State
# ---------------------------------------------------------------------------

class LogisticsState(BaseModel):
    """End-of-episode comprehensive metadata."""
    episode_id: str
    step_count: int
    max_steps: int
    done: bool
    scenario_id: str
    total_shipments: int
    total_delay_saved_hours: float
    total_rerouting_cost_usd: float
    sla_violations_count: int
    sla_compliance_rate: float
    cumulative_reward: float
    reward_breakdown: Dict[str, float]


class LogisticsStepResult(BaseModel):
    observation: LogisticsObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
