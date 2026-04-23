"""
inference.py — Baseline Inference Script
=========================================
Required by the Meta OpenEnv Hackathon.

Environment variables:
  - OPENAI_API_KEY : your API key (Groq or OpenAI)
  - API_BASE_URL   : LLM endpoint  (default: https://api.openai.com/v1)
  - MODEL_NAME     : model to use  (default: gpt-4o-mini)
  - HF_TOKEN       : HuggingFace token
  - TASK_ID        : which task to run (default: TASK-MEDIUM)
  - MAX_TURNS      : max turns per episode (default: 7)

Stdout format (STRICTLY required by hackathon grader):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys

# Load .env file automatically ONLY if we are testing locally (Grader injects API_KEY)
if "API_KEY" not in os.environ:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

from openai import OpenAI

# Ensure we can import the local OpenEnv packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import our strictly typed Pydantic environment
try:
    from envs.logistics_shipment_env.server.environment import (
        LogisticsShipmentEnvironment, LogisticsAction
    )
except ImportError:
    from server.environment import LogisticsShipmentEnvironment, LogisticsAction

# ---------------------------------------------------------------------------
# Configuration — All must have defaults per hackathon rules
# ---------------------------------------------------------------------------
API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",     "gpt-4o-mini")

# The Meta Grader specifically injects "API_KEY"
API_KEY = os.environ.get("API_KEY")

# For local fallback if API_KEY isn't set, use OPENAI_API_KEY or HF_TOKEN
if not API_KEY:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    API_KEY = os.environ.get("OPENAI_API_KEY") or HF_TOKEN

MAX_TURNS      = int(os.environ.get("MAX_TURNS",  "7"))
TASK_ID        = os.environ.get("TASK_ID",        "TASK-MEDIUM")

if not API_KEY:
    print("ERROR: API_KEY is not set. Set it in your .env file or environment.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

SYSTEM_PROMPT = """You are an AI Logistics Coordinator managing real-world shipment disruptions.

Each turn has a STRICT budget of 3 actions maximum before you MUST call end_turn.
Action budget per turn: get_network_status (1x), then 1-2 fix actions, then end_turn.

Your goals:
1. Minimise total shipment delay by rerouting the most delayed shipments first.
2. Maximize SLA compliance.
3. Send ONE professional ETA update to the most critical delayed shipment.
4. ALWAYS call end_turn after at most 3 other actions.

Available actions (respond with exactly ONE JSON object):
- {"action_type": "get_network_status"}
- {"action_type": "reroute_shipment", "shipment_id": "SHIP-XXX", "new_route": "R2", "new_carrier": "SpeedLane", "reason": "..."}
- {"action_type": "set_priority", "priority_ids": ["SHIP-001"]}
- {"action_type": "communicate_eta", "shipment_id": "SHIP-XXX", "message": "We apologise for the delay to your shipment. We expect delivery by 6pm due to port congestion."}
- {"action_type": "escalate", "shipment_id": "SHIP-XXX", "reason": "..."}
- {"action_type": "end_turn"}   <-- REQUIRED after every 1-3 actions to commit the turn

IMPORTANT: After calling communicate_eta, reroute_shipment, or get_network_status 1-3 times,
you MUST call end_turn immediately. Do NOT repeat the same action type more than once per turn.

Respond ONLY with a single valid JSON object. No markdown, no explanation.
"""


def ask_llm(step: int, network_status: dict) -> dict:
    """Ask the LLM what action to take. Raises on failure — no simulated fallback."""
    user_msg = (
        f"Step {step}. Current network status:\n"
        f"{json.dumps(network_status, indent=2)}\n\n"
        f"What is your next action? Respond ONLY with a JSON object."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def run_episode(task_id: str = TASK_ID) -> dict:
    """
    Run one full episode.
    Returns dict with keys: success, steps, rewards, total_reward
    """
    env   = LogisticsShipmentEnvironment()
    obs   = env.reset(task_id=task_id)

    task_name   = task_id
    rewards     = []
    step_global = 0
    turn        = 0
    done        = False

    # -------------------------------------------------------
    # Required stdout: [START] task=X env=logistics model=Y
    # -------------------------------------------------------
    print(f"[START] task={task_name} env=logistics_shipment_env model={MODEL_NAME}")
    sys.stdout.flush()

    while not done and turn < MAX_TURNS:
        turn += 1
        # ---- At the start of each turn, get fresh status ----
        obs = env.step(LogisticsAction(action_type="get_network_status"))
        step_global += 1
        print(
            f"[STEP] step={step_global} action=get_network_status "
            f"reward={obs.incremental_reward:.2f} done={str(obs.done).lower()} error=null"
        )
        sys.stdout.flush()

        # ---- Ask LLM for 1-3 fix actions, then end_turn ----
        for sub_step in range(4):   # max 3 fix actions + 1 forced end_turn
            network_status = obs.model_dump()
            # Tell the LLM exactly how many actions it has left
            network_status["_instructions"] = (
                f"Turn {turn}/{MAX_TURNS}. Sub-step {sub_step+1}/3. "
                f"You have {3 - sub_step} fix action(s) remaining, then you MUST call end_turn. "
                f"DO NOT call get_network_status again - use the data already provided."
            )

            error_str  = "null"
            action_str = "end_turn"
            reward_val = 0.0

            try:
                raw_action  = ask_llm(step_global + 1, network_status)
                action_obj  = LogisticsAction(**raw_action)
                action_str  = action_obj.action_type

                # Disallow repeated get_network_status inside a turn
                if action_obj.action_type == "get_network_status" and sub_step > 0:
                    action_obj = LogisticsAction(action_type="end_turn")
                    action_str = "end_turn(skipped_status)"

                obs         = env.step(action_obj)
                reward_val  = round(obs.incremental_reward, 4)
                step_global += 1

            except Exception as exc:
                error_str  = str(exc).replace("\n", " ")[:100]
                action_str = "error"
                reward_val = 0.0
                done       = True

            print(
                f"[STEP] step={step_global} action={action_str} "
                f"reward={reward_val:.2f} done={str(obs.done).lower()} error={error_str}"
            )
            sys.stdout.flush()

            if action_str in ("end_turn", "end_turn(skipped_status)") or done:
                rewards.append(reward_val)
                done = obs.done
                break

            if sub_step == 3:
                # Force end_turn if agent exhausted all sub-steps
                obs  = env.step(LogisticsAction(action_type="end_turn"))
                step_global += 1
                rewards.append(round(obs.incremental_reward, 4))
                done = obs.done
                print(
                    f"[STEP] step={step_global} action=end_turn(forced) "
                    f"reward={obs.incremental_reward:.2f} done={str(done).lower()} error=null"
                )
                sys.stdout.flush()
                break

        if done:
            break

    success     = turn >= 1
    total_score = sum(rewards)
    # The hackathon requires 'score' output to be strictly (0, 1) exclusive (no 0.0 or 1.0)
    score       = min(max(total_score / 5.0, 0.001), 0.999)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # -------------------------------------------------------
    # Required stdout: [END] success=X steps=N score=X rewards=r1,r2,...
    # -------------------------------------------------------
    print(f"[END] success={str(success).lower()} steps={step_global} score={score:.3f} rewards={rewards_str}")
    sys.stdout.flush()

    return {
        "task":         task_id,
        "success":      success,
        "steps":        step_global,
        "turns":        turn,
        "rewards":      rewards,
        "total_reward": total_score,
    }


if __name__ == "__main__":
    tasks = [
        ("TASK-EASY",   "Port Backlog Clearance (Easy)"),
        ("TASK-MEDIUM", "Mumbai Crisis Coordination (Medium)"),
        ("TASK-HARD",   "Multi-Port Network Collapse (Hard)"),
    ]

    all_scores = {}

    for tid, task_name in tasks:
        print(f"\n# ====== Running: {task_name} ======")
        result = run_episode(tid)
        all_scores[tid] = result["total_reward"]
        print(f"# Task Score: {result['total_reward']:.4f} | Turns: {result['turns']}")

    print(f"\n# ===== BASELINE SCORES SUMMARY =====")
    for tid, s in all_scores.items():
        print(f"#   {tid}: {s:.4f}")
    if all_scores:
        avg = sum(all_scores.values()) / len(all_scores)
        print(f"#   AVERAGE: {avg:.4f}")
