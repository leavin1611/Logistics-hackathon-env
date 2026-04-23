# Contributing to Logistics Shipment RL Environment

Thank you for your interest in contributing! This environment is designed to be **extendable by the research community** — you can add new scenarios, disruption types, reward dimensions, or entire carrier networks without touching the core grading logic.

---

## 🗺️ Architecture Overview

```
logistics_shipment_env/
├── server/
│   ├── environment.py    ← Core RL engine (Pydantic models, reward logic)
│   ├── app.py            ← FastAPI server (do not modify entry points)
│   └── grader.py         ← Reward calculator helpers
├── inference.py          ← Baseline agent (hackathon grader runs this)
├── dashboard.html        ← Live visual dashboard (standalone)
├── examples/             ← Demo clients and training scripts
└── openenv.yaml          ← Environment manifest
```

---

## ➕ Adding a New Scenario

Scenarios live in `server/environment.py` in the `TASKS` dictionary. 
Add a new key — existing tasks are completely unaffected.

```python
# In server/environment.py → TASKS dict
"TASK-KOLKATA": {
    "name": "Kolkata Port Flood Response",
    "description": "Monsoon floods shut KOPT. Reroute 5 fresh cargo shipments within 4 turns.",
    "max_turns": 4,
    "baseline_delay": 16.0,
    "disruptions": [
        "KOPT closed: 12h backlog due to flooding",
        "NH-12 (Kolkata–Dhanbad): impassable",
    ],
    "shipments": [
        {
            "id": "SHIP-001",
            "cargo": "Fresh Fish (perishable)",
            "origin": "Kolkata",
            "destination": "Dhanbad",
            "carrier": "CoastCargo",
            "route": "R3",
            "sla_buffer_h": -3.0,
            "delay_h": 6.0,
            "value": 18000,
            "priority": True,
            "status": "DELAYED",
            "notes": "Spoils in 8h",
        },
        # ... add more shipments
    ],
}
```

Then add it to `openenv.yaml`:

```yaml
tasks:
  - id: TASK-KOLKATA
    name: "Kolkata Port Flood Response"
    description: "Monsoon floods at KOPT. 5 shipments, 4 turns."
    difficulty: medium
```

---

## ➕ Adding a New Route

Routes live in the `ROUTES` dictionary in `server/environment.py`.

```python
"R7": {
    "name": "Kolkata–Dhanbad NH-12",
    "origin": "Kolkata",
    "destination": "Dhanbad",
    "hours": 6.0,
    "cost": 210,
    "congestion": "heavy",
    "available": True,
}
```

---

## ➕ Adding a New Action Type

1. Add the new literal to `LogisticsAction.action_type` in `server/environment.py`
2. Add a handler method `_handle_youractionname()` in `LogisticsShipmentEnvironment`
3. Wire it up in the `step()` method's if/elif chain
4. Add it to the `SYSTEM_PROMPT` in `inference.py` so the baseline agent knows about it

---

## ➕ Extending the Reward Function

The reward function in `_handle_end_turn()` uses 4 weighted dimensions.
You can add new dimensions without breaking existing ones:

```python
# Example: add a 5th "speed_bonus" dimension
speed_bonus = 0.1 if all(s["delay_h"] == 0 for s in self._state.shipments) else 0.0

# Then re-weight:
turn_rew = min(1.0, (
    0.35 * delay_score +
    0.25 * sla_score +
    0.20 * comm_score +
    0.10 * esc_score +
    0.10 * speed_bonus +
    act_bonus
))
```

---

## 🧪 Running Tests

```bash
cd logistics_shipment_env
pip install pytest
pytest tests/ -v
```

---

## 📤 Submitting a Pull Request

1. Fork this repository
2. Create a branch: `git checkout -b feature/new-scenario-kolkata`
3. Add your scenario / route / feature
4. Run the test suite to confirm nothing broke
5. Open a Pull Request with a clear description of what you added

---

## 📋 Code Style

- All data models must use **Pydantic v2** (`BaseModel`)
- All reward values must be floats strictly in **(0, 1)** range
- Use type hints everywhere
- Keep action handlers pure (no external API calls)
