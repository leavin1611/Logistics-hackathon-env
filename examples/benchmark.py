"""
benchmark.py — Multi-Model Comparative Benchmark
==================================================
Runs multiple LLM models against all 3 task difficulties and produces
a formatted comparison table + JSON leaderboard.

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --models gpt-4o-mini,llama-3.1-8b-instant --url http://localhost:8000

Requires: pip install requests tabulate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("pip install requests"); sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Colors ───────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"
B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

DEFAULT_URL   = "https://leavin1611-logistics-hackathon-env.hf.space"
TASKS         = ["TASK-EASY", "TASK-MEDIUM", "TASK-HARD"]

# Pre-defined action sequences for deterministic benchmarking
# (No LLM needed — pure environment stress test)
ACTION_SCRIPTS = {
    "TASK-EASY": [
        {"action_type": "get_network_status"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-001", "new_route": "R2", "new_carrier": "SpeedLane", "reason": "R1 heavy congestion"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-001", "message": "We sincerely apologise for the delay to your vegetable shipment. Due to port congestion at JNPT, we have rerouted your cargo via Western Highway. Expected arrival by 4:00 PM."},
        {"action_type": "end_turn"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-002", "new_route": "R2", "reason": "Clear backlog"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-002", "message": "We apologise for the inconvenience. Your auto parts shipment has been rerouted to avoid JNPT congestion. New ETA: 5:30 PM today."},
        {"action_type": "end_turn"},
        {"action_type": "end_turn"},
    ],
    "TASK-MEDIUM": [
        {"action_type": "get_network_status"},
        {"action_type": "set_priority", "priority_ids": ["SHIP-001", "SHIP-003"]},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-001", "new_route": "R2", "new_carrier": "SpeedLane", "reason": "Reefer stuck on R1 due to carrier strike"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-001", "message": "We sincerely apologise for the delay to your pharmaceutical shipment. Due to the carrier strike at JNPT, we have rerouted via Western Highway with SpeedLane. Expected arrival by 6:00 PM."},
        {"action_type": "end_turn"},
        {"action_type": "get_network_status"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-003", "new_route": "R2", "new_carrier": "IndiaFreight", "reason": "High-value server hardware needs fastest lane"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-003", "message": "We regret the delay to your server hardware shipment. Due to customs blockage, we have arranged alternative carrier IndiaFreight. ETA: 8:00 PM today."},
        {"action_type": "end_turn"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-004", "new_route": "R2", "reason": "Clear hazmat queue"},
        {"action_type": "end_turn"},
        {"action_type": "end_turn"},
        {"action_type": "end_turn"},
    ],
    "TASK-HARD": [
        {"action_type": "get_network_status"},
        {"action_type": "set_priority", "priority_ids": ["SHIP-001", "SHIP-002", "SHIP-006"]},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-001", "new_route": "R2", "new_carrier": "SpeedLane", "reason": "COVID vaccines critical — BlueLine bankrupt"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-001", "message": "URGENT: We deeply apologise for the severe delay to your COVID vaccine shipment. Due to JNPT closure and BlueLine bankruptcy, we have arranged emergency rerouting via Western Highway with SpeedLane. Expected delivery by 4:00 PM. Cold chain integrity is being actively monitored."},
        {"action_type": "end_turn"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-002", "new_route": "R4", "new_carrier": "NorthStar", "reason": "Election ballots critical priority"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-003", "new_route": "R6", "new_carrier": "IndiaFreight", "reason": "Surgical equipment — bypass Chennai congestion"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-002", "message": "We apologise for the delay to your election ballot shipment. Due to carrier suspension, we have arranged NorthStar via Yamuna Expressway. ETA: 3:00 PM."},
        {"action_type": "end_turn"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-004", "new_route": "R2", "reason": "Hazmat clear from blocked R1"},
        {"action_type": "reroute_shipment", "shipment_id": "SHIP-006", "new_route": "R6", "new_carrier": "CoastCargo", "reason": "Blood bank reefer failure bypass"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-006", "message": "We sincerely regret the delay to your blood bank supplies. Due to reefer failure on the primary route, we have rerouted via Bangalore Alt Bypass. Expected arrival by 7:00 PM."},
        {"action_type": "end_turn"},
        {"action_type": "communicate_eta", "shipment_id": "SHIP-003", "message": "We apologise for the delay to your surgical equipment. Due to Chennai port capacity restrictions, we have arranged alternative routing. ETA: 9:00 PM today."},
        {"action_type": "end_turn"},
        {"action_type": "end_turn"},
        {"action_type": "end_turn"},
        {"action_type": "end_turn"},
    ],
}


def run_scripted_episode(base_url: str, task_id: str) -> dict:
    """Run a deterministic scripted episode and return metrics."""
    sess = requests.Session()
    sess.headers["Content-Type"] = "application/json"
    
    t0 = time.time()
    
    # Reset
    r = sess.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=15)
    if not r.ok:
        return {"task": task_id, "error": f"Reset failed: {r.status_code}"}
    
    steps = 0
    total_reward = 0.0
    actions_taken = []
    
    for action in ACTION_SCRIPTS.get(task_id, []):
        r = sess.post(f"{base_url}/step", json={"action": action}, timeout=15)
        steps += 1
        if not r.ok:
            continue
        data = r.json()
        reward = data.get("reward", 0) or 0
        total_reward += reward
        actions_taken.append({
            "step": steps,
            "action": action["action_type"],
            "reward": round(reward, 4),
        })
        if data.get("done"):
            break
    
    elapsed = time.time() - t0
    
    return {
        "task": task_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "normalized_score": round(min(max(total_reward / 5.0, 0.001), 0.999), 4),
        "elapsed_s": round(elapsed, 2),
        "actions": actions_taken,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def main():
    parser = argparse.ArgumentParser(description="Logistics RL Benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--output", default="outputs/benchmark_results.json", help="Output file")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    
    print(f"\n{B}{'='*65}{X}")
    print(f"{B}🏁  Logistics Shipment RL — Benchmark Suite{X}")
    print(f"{D}  API: {base}{X}")
    print(f"{B}{'='*65}{X}\n")
    
    # Ping
    try:
        r = requests.get(f"{base}/schema", timeout=8)
        assert r.ok
        print(f"  {G}✓ API Online{X}\n")
    except Exception:
        print(f"  {R}✗ API Offline — is the space sleeping?{X}\n")
        return
    
    results = []
    for task in TASKS:
        print(f"  {C}▶ Running {task}…{X}", end="", flush=True)
        result = run_scripted_episode(base, task)
        results.append(result)
        
        if "error" in result:
            print(f"  {R}✗ {result['error']}{X}")
        else:
            color = G if result["total_reward"] > 1.0 else Y if result["total_reward"] > 0.3 else R
            print(f"\r  {G}✓{X} {task:<14}  "
                  f"reward={color}{result['total_reward']:+.4f}{X}  "
                  f"score={result['normalized_score']:.4f}  "
                  f"steps={result['steps']}  "
                  f"time={result['elapsed_s']}s")
    
    # Summary table
    print(f"\n{B}{'─'*65}{X}")
    if HAS_TABULATE:
        table = [[r["task"], f"{r.get('total_reward',0):+.4f}",
                   f"{r.get('normalized_score',0):.4f}",
                   r.get("steps", "?"), f"{r.get('elapsed_s', 0):.1f}s"]
                  for r in results]
        print(tabulate(table, headers=["Task", "Reward", "Score", "Steps", "Time"],
                       tablefmt="rounded_outline"))
    
    avg_score = sum(r.get("normalized_score", 0) for r in results) / len(results) if results else 0
    avg_reward = sum(r.get("total_reward", 0) for r in results) / len(results) if results else 0
    print(f"\n  {B}Average Score:  {avg_score:.4f}{X}")
    print(f"  {B}Average Reward: {avg_reward:+.4f}{X}")
    print(f"{B}{'─'*65}{X}\n")
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "outputs", exist_ok=True)
    output = {
        "benchmark": "logistics_shipment_rl",
        "api_url": base,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results,
        "summary": {
            "avg_score": avg_score,
            "avg_reward": avg_reward,
            "tasks_run": len(results),
        }
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  {D}Results saved to {args.output}{X}\n")


if __name__ == "__main__":
    main()
