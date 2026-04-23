# 🚛 LogisticsShipmentRL — Environment Specification

> **Event:** Meta PyTorch OpenEnv Hackathon 2026  
> **Domain:** Supply Chain / Route Optimization  
> **Type:** Multi-Agent Reinforcement Learning Environment

## 1. Concept

**LogisticsShipmentRL** is a multi-step Reinforcement Learning environment built on the OpenEnv framework. An LLM agent acts as an **AI Logistics Coordinator**. The agent handles real-world supply chain disruptions — truck breakdowns, port congestion, weather delays, customs holds — by making intelligent re-routing and communication decisions under time pressure.

## 2. Gameplay & Rules

Each episode represents a **5-hour coordination window** (5 steps total). Every step simulates one hour of real-world time.

The agent receives a **shipment network snapshot** containing:
- 🚛 **Active Shipments:** 5–12 active shipments with SLA deadlines.
- ⚠️ **Disruptions:** 2–5 active events like port congestion, strikes, or bad weather.
- 🔄 **Routes:** Alternative routes with varying costs and delivery times.
- 📡 **Live Updates:** Feedback dynamically injected per step.

**The agent's objective is to:**
1. Re-route delayed shipments to bypass disruptions.
2. Prioritize high-value and perishable cargo.
3. Communicate clear ETA updates to affected customers.

## 3. Communication Contract (API)

### Agent -> Environment (Action)
The LLM agent must respond with a JSON object conforming to the `LogisticsAction` Pydantic model:
- `reasoning`: Chain-of-thought analysis explaining strategy.
- `rerouting_decisions`: Dictionary mapping shipment IDs to new routes.
- `priority_shipments`: List of up to 3 shipment IDs to fast-track.
- `customer_communications`: Dictionary of messages to send to customers.
- `escalations`: Any shipments needing a human dispatcher.

### Environment -> Agent (Observation)
The environment provides the current state via the `LogisticsObservation` Pydantic model:
- `network_snapshot`: Rich natural language description of the state.
- `active_shipments`: List of shipments and their individual statuses/SLAs.
- `disruption_events`: Active disruptions and estimated completion times.
- `available_routes`: Routes and their live viability.
- `current_total_delay_hours`: Network health metric.

## 4. Evaluation (Grader)

The environment calculates a float reward (0.0 to 1.0) based on shaped constraints:
1. **Delay Reduction (40%):** Total delay hours saved vs. a do-nothing baseline.
2. **Cost Efficiency (30%):** Re-routing cost relative to SLA breach penalty avoided.
3. **SLA Compliance (20%):** Percentage of shipments successfully delivered within the SLA window.
4. **Communication Quality (10%):** LLM-judged clarity and professionalism of the `customer_communications` output.

## 5. Development Roadmap

- [ ] Project directory initialization (`openenv.yaml`, `__init__.py`).
- [ ] Define precise domain models (`models.py`).
- [ ] Implement client interface for OpenEnv (`client.py`).
- [ ] Create the server backend for state management and simulation (`server/logistics_environment.py`).
- [ ] Build the reward calculation logic (`server/grader.py`).
- [ ] Design the procedural scenario generator (`server/scenarios.py`).
- [ ] Finalize the interactive FastAPI app (`server/app.py`).
