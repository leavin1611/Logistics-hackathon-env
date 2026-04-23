"""
LogisticsShipmentRL — Scenario Library
Pre-built disruption events and procedural simulation generators.
"""

from typing import Dict, List
from pydantic import BaseModel
import random


class ScenarioTemplate(BaseModel):
    scenario_id: str
    title: str
    description: str
    weather_forecast: str
    difficulty: int
    shipments: List[Dict]
    disruptions: List[Dict]


ROUTES = {
    "R1": {
        "route_id": "R1", "route_name": "NH-48 Express (Mumbai–Pune)",
        "origin": "Mumbai", "destination": "Pune",
        "distance_km": 148, "estimated_hours": 3.5, "cost_usd": 120,
        "carrier_options": ["C1", "C2"],
        "current_congestion": "clear", "weather_risk": "none", "is_available": True,
    },
    "R2": {
        "route_id": "R2", "route_name": "Western Express Highway (Alt)",
        "origin": "Mumbai", "destination": "Pune",
        "distance_km": 162, "estimated_hours": 4.0, "cost_usd": 105,
        "carrier_options": ["C2", "C3"],
        "current_congestion": "light", "weather_risk": "low", "is_available": True,
    },
    "R3": {
        "route_id": "R3", "route_name": "NH-44 North Corridor (Delhi–Agra)",
        "origin": "Delhi", "destination": "Agra",
        "distance_km": 206, "estimated_hours": 4.5, "cost_usd": 160,
        "carrier_options": ["C4", "C5"],
        "current_congestion": "moderate", "weather_risk": "none", "is_available": True,
    },
}

CARRIERS = {"C1": 8, "C2": 5, "C3": 12, "C4": 4, "C5": 7}

SCENARIO_TEMPLATES = [
    ScenarioTemplate(
        scenario_id="SCN-001",
        title="Mumbai Port Congestion",
        description="A major system failure at JNPT Port has delayed thousands of containers.",
        weather_forecast="Heavy rain affecting the Western Highway.",
        difficulty=3,
        shipments=[
            {
                "shipment_id": "SHIP-001", "origin": "Mumbai", "destination": "Pune",
                "cargo_type": "perishable", "cargo_description": "Fresh Pharmaceuticals",
                "current_location": "Mumbai Hub", "current_status": "delayed",
                "assigned_carrier": "C1", "assigned_route": "R1",
                "estimated_arrival": "+8h", "sla_deadline": "+4h", "sla_buffer_hours": -4.0,
                "current_delay_hours": 4.0, "value_usd": 45000, "is_priority": True,
                "notes": "Reefer container power failing. Must reroute to R2 to avoid complete gridlock.",
            },
            {
                "shipment_id": "SHIP-002", "origin": "Delhi", "destination": "Agra",
                "cargo_type": "standard", "cargo_description": "Consumer Electronics",
                "current_location": "In Transit R3", "current_status": "in_transit",
                "assigned_carrier": "C4", "assigned_route": "R3",
                "estimated_arrival": "+3h", "sla_deadline": "+5h", "sla_buffer_hours": 2.0,
                "current_delay_hours": 0.0, "value_usd": 12000, "is_priority": False,
                "notes": "Route clear.",
            }
        ],
        disruptions=[
            {
                "event_id": "EVT-001", "event_type": "port_congestion", "location": "Mumbai Port",
                "affected_routes": ["R1"], "affected_shipments": ["SHIP-001"],
                "severity": "critical", "estimated_duration_hours": 12.0,
                "estimated_additional_delay_hours": 6.0, "description": "Total gridlock.",
                "can_be_bypassed": True,
            }
        ]
    )
]

def generate_random_scenario(seed: int | None = None) -> ScenarioTemplate:
    """Generate a random dynamic array of disruptions and shipping traffic."""
    rng = random.Random(seed)
    n_shipments = rng.randint(4, 8)
    n_disruptions = rng.randint(2, 4)

    types = ["standard", "perishable", "hazmat", "high_value"]
    cities = ["Mumbai", "Delhi", "Pune", "Chennai"]

    shipments = []
    for i in range(n_shipments):
        orig, dest = rng.sample(cities, 2)
        sla_buf = round(rng.uniform(-2, 6), 1)
        cargo = rng.choice(types)
        shipments.append({
            "shipment_id": f"SHIP-{i+1:03d}",
            "origin": orig, "destination": dest,
            "cargo_type": cargo, "cargo_description": f"{cargo.title()} payload",
            "current_location": f"{orig} Hub",
            "current_status": "delayed" if sla_buf < 0 else "in_transit",
            "assigned_carrier": rng.choice(list(CARRIERS.keys())),
            "assigned_route": rng.choice(list(ROUTES.keys())),
            "estimated_arrival": f"+{rng.randint(2, 8)}h",
            "sla_deadline": f"+{rng.randint(2, 12)}h",
            "sla_buffer_hours": sla_buf,
            "current_delay_hours": max(0.0, -sla_buf),
            "value_usd": rng.randint(5000, 150000),
            "is_priority": rng.random() > 0.8,
            "notes": "Routine logistics status.",
        })

    disruptions = []
    for j in range(n_disruptions):
        disruptions.append({
            "event_id": f"EVT-{j+1:03d}",
            "event_type": rng.choice(["weather", "breakdown", "port_congestion", "road_closure"]),
            "location": rng.choice(cities),
            "affected_routes": [rng.choice(list(ROUTES.keys()))],
            "affected_shipments": [s["shipment_id"] for s in shipments if rng.random() > 0.7],
            "severity": rng.choice(["low", "medium", "high", "critical"]),
            "estimated_duration_hours": round(rng.uniform(2, 8), 1),
            "estimated_additional_delay_hours": round(rng.uniform(1, 5), 1),
            "description": "Dynamic disruption localized to this area.",
            "can_be_bypassed": rng.random() > 0.2,
        })

    return ScenarioTemplate(
        scenario_id=f"SCN-RND-{rng.randint(1000, 9999)}",
        title="Procedural Network Disruption",
        description="Randomly simulated field breakdown for agent tests.",
        weather_forecast="Mixed, mostly clear.",
        difficulty=rng.randint(2, 5),
        shipments=shipments,
        disruptions=disruptions,
    )


def get_scenario(scenario_id: str, seed: int | None = None) -> ScenarioTemplate:
    if scenario_id.startswith("SCN-RND"):
        return generate_random_scenario(seed)
    try:
        return next(t for t in SCENARIO_TEMPLATES if t.scenario_id == scenario_id)
    except StopIteration:
        return generate_random_scenario(seed)
