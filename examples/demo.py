"""
demo.py — Zero-Setup Interactive Demo
======================================
Connects directly to the live Hugging Face Space and plays a full episode
without requiring any local installation beyond: pip install requests

Usage:
    python examples/demo.py
    python examples/demo.py --task TASK-HARD
    python examples/demo.py --url http://localhost:8000
"""

import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests first: pip install requests")
    sys.exit(1)

# ─── ANSI colors ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

DEFAULT_URL  = "https://leavin1611-logistics-hackathon-env.hf.space"
DEFAULT_TASK = "TASK-MEDIUM"


def fmt_status(s):
    colors = {"DELAYED": YELLOW, "CRITICAL": RED, "RESOLVED": GREEN, "IN_TRANSIT": CYAN}
    return f"{colors.get(s, '')}{s}{RESET}"


def print_shipments(shipments):
    print(f"\n  {'ID':<10} {'Status':<14} {'Cargo':<35} {'Delay':>6}  {'SLA Buffer':>12}  {'Route':>5}")
    print("  " + "─" * 85)
    for s in shipments:
        sla = s['sla_buffer_h']
        sla_color = GREEN if sla >= 0 else (YELLOW if sla > -2 else RED)
        print(
            f"  {BOLD}{s['id']:<10}{RESET}"
            f"{fmt_status(s['status']):<23}"
            f"{s['cargo'][:33]:<35}"
            f"{s['delay_h']:>6.1f}h"
            f"  {sla_color}{sla:>+10.1f}h{RESET}"
            f"  {s['route']:>5}"
        )


def play_demo(base_url: str, task_id: str):
    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}🚛  Logistics Shipment RL — Live Demo{RESET}")
    print(f"{DIM}  API  : {base_url}{RESET}")
    print(f"{DIM}  Task : {task_id}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # ── Ping ──────────────────────────────────────────────────────
    print(f"{DIM}Pinging API…{RESET}", end="", flush=True)
    try:
        r = sess.get(f"{base_url}/schema", timeout=10)
        r.raise_for_status()
        print(f"\r{GREEN}✓ API Online{RESET}            ")
    except Exception as e:
        print(f"\r{RED}✗ API Offline: {e}{RESET}")
        print(f"{DIM}(The Space may be sleeping — try again in 30s){RESET}")
        return

    # ── Reset ─────────────────────────────────────────────────────
    print(f"\n{BOLD}[RESET]{RESET} Starting episode…")
    r = sess.post(f"{base_url}/reset", data=json.dumps({"task_id": task_id}), timeout=15)
    if not r.ok:
        print(f"{RED}Reset failed: {r.text[:200]}{RESET}")
        return
    obs = r.json()

    print(f"\n  {BOLD}Task:{RESET}        {obs['task']}")
    print(f"  {BOLD}Max Turns:{RESET}   {obs['max_turns']}")
    print(f"  {BOLD}Disruptions:{RESET}")
    for d in obs.get("disruptions", []):
        print(f"    {RED}⚠{RESET}  {d}")
    print_shipments(obs.get("shipments", []))

    # ── Episode script ────────────────────────────────────────────
    # Pre-defined sensible actions to demonstrate the environment
    actions = [
        {"action_type": "get_network_status"},
        {"action_type": "set_priority", "priority_ids": ["SHIP-001", "SHIP-003"]},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-001",
         "new_route": "R2", "new_carrier": "SpeedLane",
         "reason": "R1 congested by JNPT backlog; R2 has light traffic"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-001",
         "message": "We sincerely apologise for the delay to your pharmaceutical shipment. "
                    "Due to ongoing port congestion at JNPT we have rerouted via the Western Highway. "
                    "Expected arrival by 6:00 PM today. We will continue to monitor your shipment."},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-003",
         "new_route": "R2", "new_carrier": "IndiaFreight",
         "reason": "Server hardware is high-value and SLA is critically breached"},
        {"action_type": "end_turn"},
        {"action_type": "get_network_status"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-003",
         "message": "We apologise for the delay to your server hardware. Due to a carrier strike "
                    "we have arranged an alternative carrier. Your shipment is now in transit and "
                    "expected to arrive by 8:00 PM this evening."},
        {"action_type": "end_turn"},
    ]

    total_reward = 0.0
    step_n = 0

    for action in actions:
        step_n += 1
        atype = action["action_type"]
        print(f"\n{BOLD}[STEP {step_n}]{RESET} → {CYAN}{atype}{RESET}")
        if "shipment_id" in action:
            print(f"          shipment: {action['shipment_id']}")

        r = sess.post(
            f"{base_url}/step",
            data=json.dumps({"action": action}),
            timeout=15,
        )
        if not r.ok:
            print(f"  {RED}Error {r.status_code}: {r.text[:300]}{RESET}")
            if r.status_code == 422:
                try:
                    errs = r.json().get("detail", [])
                    for e in errs:
                        print(f"  {RED}  • {e['loc'][-1]}: {e['msg']}{RESET}")
                except Exception:
                    pass
            continue

        data = r.json()
        step_obs  = data.get("observation", {})
        reward    = data.get("reward", 0) or 0
        done      = data.get("done", False)
        total_reward += reward

        print(f"  {GREEN if reward >= 0 else RED}reward = {reward:+.4f}{RESET}   "
              f"cumulative = {step_obs.get('cumulative_reward', 0):.4f}   "
              f"done = {done}")
        if step_obs.get("feedback"):
            print(f"  {DIM}{step_obs['feedback'][:100]}{RESET}")

        if atype in ("end_turn", "get_network_status"):
            print_shipments(step_obs.get("shipments", []))

        if done:
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"{GREEN}{BOLD}🏁  Episode Complete!{RESET}")
            print(f"    Total Reward : {BOLD}{total_reward:.4f}{RESET}")
            print(f"    Steps Taken  : {step_n}")
            print(f"{BOLD}{'='*60}{RESET}\n")
            return
        time.sleep(0.3)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"  {DIM}Demo complete. Total reward so far: {total_reward:.4f}{RESET}")
    print(f"  {DIM}Visit the live dashboard:{RESET}")
    print(f"  {CYAN}{base_url}/docs{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistics RL Demo")
    parser.add_argument("--url",  default=DEFAULT_URL,  help="API base URL")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task ID")
    args = parser.parse_args()
    play_demo(args.url.rstrip("/"), args.task)
