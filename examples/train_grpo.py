"""
GRPO Training with LogisticsShipmentRL Environment
===================================================
Trains an LLM to act as an AI Logistics Coordinator using Group Relative
Policy Optimization (TRL + OpenEnv).

Usage:
    # Option 1: HF Space environment
    python train_grpo.py --model Qwen/Qwen3-1.7B --env-url https://YOUR_USERNAME-logistics-shipment-env.hf.space

    # Option 2: Local environment (run server first)
    #   PYTHONPATH=src uvicorn envs.logistics_shipment_env.server.app:app --port 8000 --reload
    python train_grpo.py --model Qwen/Qwen3-1.7B --env-url http://localhost:8000 --vllm-mode colocate

Install:
    pip install trl vllm datasets transformers
    pip install git+https://github.com/meta-pytorch/OpenEnv.git
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for Logistics Shipment RL")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B",        help="Model id (HF hub or local)")
    p.add_argument("--env-url",        default="http://localhost:8000",   help="URL for the logistics environment server")
    p.add_argument("--dataset-size",   type=int, default=500,            help="Training dataset size")
    p.add_argument("--max-turns",      type=int, default=5,              help="Max turns per episode")
    p.add_argument("--epochs",         type=int, default=1)
    p.add_argument("--lr",             type=float, default=5e-6)
    p.add_argument("--grad-accum",     type=int, default=32)
    p.add_argument("--num-gen",        type=int, default=2,              help="Rollout generations per prompt")
    p.add_argument("--output-dir",     default="outputs/logistics-grpo")
    p.add_argument("--vllm-mode",      choices=["colocate","server"], default="colocate")
    p.add_argument("--vllm-url",       default="http://localhost:8000",  help="vLLM server URL (if --vllm-mode=server)")
    p.add_argument("--push-to-hub",    action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------
# System prompt the LLM sees as the "logistics coordinator"
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI Logistics Coordinator managing a fleet of shipments under active disruption.

Your goal is to minimise delivery delays, maintain SLA compliance, communicate proactively with customers, and avoid unnecessary escalations.

## YOUR TOOLS
You interact with the environment via MCP tool calls. Available tools:
- get_network_status()              → See all shipments, disruptions, route options
- reroute_shipment(id, route, carrier, reason)  → Switch a delayed shipment to a better route
- set_priority([ids])               → Fast-track up to 3 critical shipments
- communicate_eta(id, message)      → Send customer ETA update (graded for quality)
- escalate(id, reason)              → Flag for human dispatcher (-0.1 reward each!)
- end_turn()                        → Commit all decisions and receive your reward

## STRATEGY
1. Always call get_network_status() first.
2. Re-route shipments with negative sla_buffer_hours away from congested routes.
3. Prioritise high-value or perishable cargo.
4. Send clear, empathetic ETA updates to delayed customers.
5. Only escalate if truly unresolvable — each escalation costs -0.1 reward.
6. Always end with end_turn() to commit your actions.

- Delay Reduction:       40%
- SLA Compliance:        30%
- Communication Quality: 20%
- Escalation Control:    10%

## MULTI-AGENT WARNING
Routes now have limited capacity (Theme #1). Check `route_load` in the network status. If a route is > 85% loaded, rerouting to it will fail.
""".strip()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def build_user_prompt(turn: int, network_status: dict) -> str:
    """Format the current turn's situation as a user message."""
    delayed = [s for s in network_status.get("shipments", []) if s["sla_buffer_hours"] < 0]
    disruptions = network_status.get("disruptions", [])
    load = network_status.get("route_load", {})
    load_str = ", ".join([f"{r}: {v*100:.0f}%" for r, v in load.items()])
    return (
        f"Turn {turn}/{network_status.get('max_turns', 5)}. "
        f"Network Load: {load_str}. "
        f"Disruptions: {'; '.join(disruptions[:2])}. "
        f"Delayed shipments: {[s['id'] for s in delayed]}. "
        f"Cumulative reward so far: {network_status.get('cumulative_reward', 0.0):.3f}. "
        "What are your next actions? Use the MCP tools to respond."
    )


def rollout_once(trainer, env, tokenizer, system_prompt: str, max_turns: int) -> dict:
    """
    Run one full episode of the Logistics environment.
    Returns token ids, logprobs, and reward signals for GRPO.
    """
    from trl.experimental.openenv import generate_rollout_completions

    env.reset()

    prompt_ids_all, completion_ids_all, logprobs_all = [], [], []
    turn_rewards, sla_rewards, comm_rewards = [], [], []

    for turn in range(max_turns):
        # Get current state
        status = env.call_tool("get_network_status")
        if status.get("turns_remaining", 1) == 0:
            break

        user_prompt = build_user_prompt(turn + 1, status)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        # Generate model action
        rollout = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids_all.extend(rollout["prompt_ids"])
        completion_ids_all.extend(rollout["completion_ids"])
        logprobs_all.extend(rollout["logprobs"])
        action_text = rollout.get("text") or tokenizer.decode(
            rollout["completion_ids"], skip_special_tokens=True
        )

        # Parse tool calls from the LLM's text (simple heuristic)
        _execute_tool_calls(env, action_text)

        # Commit the turn
        result = env.call_tool("end_turn")
        breakdown = result.get("reward_breakdown", {})
        turn_rewards.append(float(result.get("turn_reward", 0.0)))
        sla_rewards.append(float(breakdown.get("sla_compliance", 0.0)))
        comm_rewards.append(float(breakdown.get("communication_quality", 0.0)))

        if result.get("done"):
            break

    return {
        "prompt_ids":        prompt_ids_all,
        "completion_ids":    completion_ids_all,
        "logprobs":          logprobs_all,
        "delay_reward":      turn_rewards[-1] if turn_rewards else 0.0,
        "sla_reward":        sla_rewards[-1]  if sla_rewards  else 0.0,
        "comm_reward":       comm_rewards[-1] if comm_rewards  else 0.0,
    }


def _execute_tool_calls(env, text: str) -> None:
    """
    Naive tool-call parser: looks for known tool names in the LLM's output
    and calls them. For production, use the MCP structured output.
    """
    text_lower = text.lower()

    # If LLM mentions rerouting, attempt it for first delayed shipment
    if "reroute" in text_lower or "r2" in text_lower:
        try:
            env.call_tool(
                "reroute_shipment",
                shipment_id="SHIP-001",
                new_route="R2",
                new_carrier="SpeedLane",
                reason="Extracted from model output: avoid congested R1",
            )
        except Exception:
            pass

    # If LLM mentions customer communication
    if any(w in text_lower for w in ["eta", "reschedul", "sorry", "apologis", "delay"]):
        try:
            # Find first delayed shipment from the status
            env.call_tool(
                "communicate_eta",
                shipment_id="SHIP-001",
                message=text[:300],  # Use the model's own words
            )
        except Exception:
            pass

    # If LLM mentions priority
    if "priority" in text_lower or "ship-003" in text_lower:
        try:
            env.call_tool("set_priority", shipment_ids=["SHIP-001", "SHIP-003"])
        except Exception:
            pass


# --------------------------------------------------------------------------
# Reward functions (TRL GRPO format)
# --------------------------------------------------------------------------

def reward_delay(completions: list, **kwargs) -> list[float]:
    """Primary reward: delay hours saved."""
    return [float(r) for r in kwargs.get("delay_reward", [0.0] * len(completions))]

def reward_sla(completions: list, **kwargs) -> list[float]:
    """Secondary reward: SLA compliance rate."""
    return [float(r) for r in kwargs.get("sla_reward", [0.0] * len(completions))]

def reward_communication(completions: list, **kwargs) -> list[float]:
    """Tertiary reward: communication quality."""
    return [float(r) for r in kwargs.get("comm_reward", [0.0] * len(completions))]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Lazy imports so the file can be read without GPU deps
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from envs.logistics_shipment_env import LogisticsShipmentEnv

    print(f"🚛 Logistics Shipment GRPO Training")
    print(f"   Model:   {args.model}")
    print(f"   Env:     {args.env_url}")
    print(f"   Dataset: {args.dataset_size} prompts × {args.num_gen} rollouts")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Environment client (sync wrapper)
    raw_env = LogisticsShipmentEnv(base_url=args.env_url)
    env = raw_env.sync().__enter__()

    # Dataset — each row is a prompt seeded with the task description
    dataset_prompt = (
        "You are managing a logistics network with 4 active shipments under disruption. "
        "Minimise delays, maintain SLA, and communicate clearly with customers."
    )
    dataset = Dataset.from_dict({"prompt": [dataset_prompt] * args.dataset_size})

    # GRPO config
    grpo_config = GRPOConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        per_device_train_batch_size=1,
        warmup_steps=10,
        num_generations=args.num_gen,
        max_completion_length=512,
        max_prompt_length=1024,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_url if args.vllm_mode == "server" else None,
        output_dir=args.output_dir,
        report_to="none",
        logging_steps=1,
        save_steps=20,
        push_to_hub=args.push_to_hub,
    )

    # Rollout function
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict:
        all_prompt_ids, all_comp_ids, all_logprobs = [], [], []
        d_rewards, s_rewards, c_rewards = [], [], []

        for _ in prompts:
            ep = rollout_once(trainer, env, tokenizer, SYSTEM_PROMPT, args.max_turns)
            all_prompt_ids.append(ep["prompt_ids"])
            all_comp_ids.append(ep["completion_ids"])
            all_logprobs.append(ep["logprobs"])
            d_rewards.append(ep["delay_reward"])
            s_rewards.append(ep["sla_reward"])
            c_rewards.append(ep["comm_reward"])

        return {
            "prompt_ids":     all_prompt_ids,
            "completion_ids": all_comp_ids,
            "logprobs":       all_logprobs,
            "delay_reward":   d_rewards,
            "sla_reward":     s_rewards,
            "comm_reward":    c_rewards,
        }

    # Trainer
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[reward_delay, reward_sla, reward_communication],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("\n🏋️  Starting GRPO training...\n")
    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        if args.push_to_hub:
            trainer.push_to_hub()
        print(f"\n✅ Training complete. Model saved to {args.output_dir}")
    finally:
        env.__exit__(None, None, None)


if __name__ == "__main__":
    main()
