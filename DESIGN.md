# Design Notes: Logistics Shipment RL Environment

## Why This Problem?

India's logistics sector processes over **$200B** of freight annually. During major disruptions (port strikes, monsoon floods, highway accidents), human dispatchers must manually triage hundreds of concurrent shipments in minutes. This environment simulates that exact crisis — testing whether an LLM can act as autonomous operational infrastructure.

## Key Architectural Decisions

### 1. Turn-Based with Sub-Step Budgets

The environment uses a **turn-based loop** rather than continuous steps. Each turn represents one dispatcher "shift" of decisions. Within each turn, the agent has a budget of 3 actions before calling `end_turn`.

This design choice was deliberate:
- It prevents the agent from repeatedly calling `get_network_status` (a known LLM failure mode with tool-use)
- It mirrors real dispatcher workflows, where decisions are batched before being committed
- It creates natural exploration-exploitation trade-offs: spend your budget on rerouting or communication?

### 2. Incremental + Terminal Rewards

Most RL environments give rewards only at the end of an episode. This environment gives **incremental rewards at every step** (`+0.01` for status check, `+0.05–0.15` for rerouting, `+0.02–0.12` for communication) plus a structured **terminal reward** from 4 weighted dimensions.

This is critical for GRPO training: the credit assignment problem is much easier when the agent gets meaningful signal at each action, not just a sparse signal at the end of 7 turns.

### 3. The `escalate` Action is Intentionally Penalized

The `escalate` action (`-0.1` reward) exists as a deliberate test of agent judgment. A naive agent that doesn't know how to reroute a shipment might always escalate. An intelligent agent learns that it can earn more reward by attempting reroutes — even imperfect ones — than by immediately deferring to humans.

### 4. NLP-Scored Communication

The `communicate_eta` action is graded by a lightweight heuristic that checks for:
- Empathetic language ("apologize", "sorry", "regret")
- Specific ETA commitment ("arrive by 6pm", "reschedule to Monday")
- Cause of delay ("port congestion", "carrier strike", "weather")
- Message length (longer = more effort)

This tests a different capability than rerouting math: can the agent produce professional, customer-facing communication under pressure?

### 5. Three Difficulty Tiers for Benchmarking

| Tier | Challenge | Design Intent |
|------|-----------|---------------|
| EASY | 2 shipments, 1 disruption, 3 turns | Validates basic rerouting ability |
| MEDIUM | 4 shipments, 3 simultaneous disruptions, 5 turns | Tests triage prioritization under pressure |
| HARD | 7 shipments, 4 critical failures, 7 turns | Stress-tests maximum crisis management capacity |

An agent that scores well on EASY but poorly on HARD reveals brittleness under complexity — a real finding, not a grader artifact.

## Reward Function Anatomy

```
turn_reward = min(1.0, (
    0.40 × delay_score    # Hours saved vs. do-nothing baseline
  + 0.30 × sla_score      # % shipments still within SLA window
  + 0.20 × comm_score     # Coverage × quality of customer messages
  + 0.10 × esc_score      # 1.0 - (0.1 × number of escalations)
  + act_bonus             # +0.05 if agent used ≥ 3 actions per turn
))
```

Delay reduction (40%) is weighted highest because it has the most direct economic impact. Communication (20%) represents the "soft skills" test that distinguishes capable agents from purely algorithmic optimizers.

## Known Limitations

1. **Route graph is static**: Routes don't dynamically close or change congestion mid-episode. A future version could add stochastic disruption evolution.
2. **Single-agent only**: The environment is not designed for multi-agent scenarios where separate dispatchers handle different regions.
3. **No partial observability**: The agent always sees the full network state after `get_network_status`. A more challenging variant would restrict observations to a regional viewport.

## Future Extensions

- `TASK-KOLKATA` — Monsoon flood response at KOPT
- `TASK-VIZAG` — Cyclone-induced port closure on the eastern coast  
- `TASK-MUNDRA` — Container ship grounding at India's largest private port
- Dynamic route congestion that evolves each turn
- Multi-modal transport (sea + rail + road intermodal chains)
