# Changelog

All notable changes to the Logistics Shipment RL Environment.

## [0.1.0] — 2026-04-09

### 🚀 Initial Release — Meta PyTorch OpenEnv Hackathon

**Core Environment**
- Multi-turn logistics crisis management simulator
- 6 action types: `get_network_status`, `reroute_shipment`, `set_priority`, `communicate_eta`, `escalate`, `end_turn`
- Strict Pydantic v2 typed action/observation/state models
- 4-dimensional reward function (delay reduction, SLA compliance, communication quality, escalation control)
- 3 difficulty tiers: TASK-EASY, TASK-MEDIUM, TASK-HARD

**Infrastructure**
- FastAPI + Uvicorn HTTP server (OpenEnv compatible)
- Docker deployment to HuggingFace Spaces
- `inference.py` baseline agent with Meta LiteLLM proxy compliance
- GitHub Actions CI pipeline

**Documentation & Tooling**
- `dashboard.html` — Live interactive visual dashboard
- `DESIGN.md` — Architecture decisions and reward anatomy
- `CONTRIBUTING.md` — Guide for extending scenarios, routes, and actions
- `CITATION.md` — BibTeX and APA citation
- `README.md` — Mermaid state diagrams, episode transcripts, full setup guide

**Analysis Scripts**
- `examples/demo.py` — Zero-setup terminal demo client
- `examples/benchmark.py` — Multi-task deterministic benchmark suite
- `examples/complexity_analysis.py` — Mathematical state/action space analysis
- `examples/reward_analysis.py` — ASCII-art reward curve visualization
- `examples/train_grpo.py` — GRPO training starter script

**Quality**
- 30+ pytest unit tests covering all action types and edge cases
- BSD-3-Clause license
