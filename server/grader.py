"""
LogisticsShipmentRL — Grader Module
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict
from models import LogisticsAction

def calculate_delay_score(baseline_delay: float, new_delay: float) -> float:
    """40% Weight: Returns 0.0 to 1.0 based on hours of delay saved relative to do-nothing baseline."""
    hours_saved = max(0.0, baseline_delay - new_delay)
    # Assume 10 hours saved is a "perfect" score for normalization
    return min(1.0, hours_saved / 10.0)

def calculate_cost_efficiency(base_cost: float, new_cost: float, penalties_avoided: float) -> float:
    """30% Weight: Re-routing cost relative to SLA breach penalty avoided."""
    additional_cost = max(0.0, new_cost - base_cost)
    if penalties_avoided == 0.0:
        return 1.0 if additional_cost == 0.0 else 0.0
    efficiency = max(0.0, 1.0 - (additional_cost / penalties_avoided))
    return efficiency

def calculate_sla_compliance(shipments: list) -> float:
    """20% Weight: % of shipments mathematically still within SLA window."""
    if not shipments:
        return 1.0
    on_time = sum(1 for s in shipments if s.sla_buffer_hours >= 0)
    return on_time / len(shipments)

def grade_communication_quality(action: LogisticsAction) -> float:
    """10% Weight: Grades the LLM's text output via heuristics for clarity."""
    score = 0.0
    
    # Needs to actually send something if shipments are delayed
    comms = action.customer_communications
    if not comms:
        return 0.5 
    
    avg_score = 0.0
    for sms in comms.values():
        txt = sms.lower()
        sub = 0.0
        # Check professional tone/clear ETA
        if "sorry" in txt or "apolog" in txt:
            sub += 0.3
        if "eta" in txt or "arrive" in txt or "reschedule" in txt:
            sub += 0.4
        if "reason" in txt or "due to" in txt or "weather" in txt or "port" in txt:
            sub += 0.3
        avg_score += min(1.0, sub)
        
    score = avg_score / len(comms)
    return score

def compute_reward(action_dict: Dict[str, Any], state_info: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    """Root scorer aggregating the 4 values."""
    action = LogisticsAction(**action_dict)
    
    # Destructure metrics provided by environment's internal update simulation
    baseline = state_info.get("baseline_delay", 10.0)
    actual = state_info.get("new_delay", 10.0)
    
    base_c = state_info.get("base_cost", 1000.0)
    new_c = state_info.get("new_cost", 1000.0)
    penalties = state_info.get("penalties_avoided", 5000.0)

    shipments = state_info.get("agent_shipments", [])
    
    d_score = calculate_delay_score(baseline, actual)
    c_score = calculate_cost_efficiency(base_c, new_c, penalties)
    s_score = calculate_sla_compliance(shipments)
    m_score = grade_communication_quality(action)
    
    weighted_sum = (
        0.40 * d_score +
        0.30 * c_score +
        0.20 * s_score +
        0.10 * m_score
    )
    
    # Meta Hackathon Phase 2 strict bounds requirement
    weighted_sum = min(max(weighted_sum, 0.001), 0.999)
    
    breakdown = {
        "delay_score": d_score,
        "cost_efficiency": c_score,
        "sla_compliance": s_score,
        "comm_quality": m_score
    }
    
    return weighted_sum, breakdown
