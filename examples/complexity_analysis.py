"""
complexity_analysis.py — Environment Complexity Report
=======================================================
Generates a formal mathematical analysis of the environment's
state space, action space, and branching factor.

This is the kind of analysis academic reviewers love to see.

Usage:
    python examples/complexity_analysis.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from server.environment import TASKS, ROUTES, CARRIERS, LogisticsAction
except ImportError:
    print("Run from project root: python examples/complexity_analysis.py")
    sys.exit(1)

# ── Colors ───────────────────────────────────────────────────────────────
B = "\033[1m"; D = "\033[2m"; C = "\033[96m"; G = "\033[92m"
Y = "\033[93m"; R = "\033[91m"; M = "\033[95m"; X = "\033[0m"


def factorial(n):
    return math.factorial(n) if n >= 0 else 1


def comb(n, k):
    return math.comb(n, k)


def analyze_task(task_id: str, task_def: dict):
    n_shipments = len(task_def["shipments"])
    n_routes = len(ROUTES)
    n_carriers = len(CARRIERS)
    max_turns = task_def["max_turns"]
    n_disruptions = len(task_def["disruptions"])
    actions_per_turn = 4  # max sub-step budget

    # Action space per step
    n_reroute = n_shipments * (n_routes - 1) * n_carriers  # shipment × route × carrier
    n_priority = sum(comb(n_shipments, k) for k in range(1, min(4, n_shipments + 1)))
    n_communicate = n_shipments  # message content is free-text (infinite)
    n_escalate = n_shipments
    n_status = 1
    n_end_turn = 1

    # Total discrete actions (excluding free-text)
    total_discrete = n_reroute + n_priority + n_communicate + n_escalate + n_status + n_end_turn

    # Episode branching factor
    branching_per_turn = total_discrete ** actions_per_turn
    total_episode_paths = branching_per_turn ** max_turns

    # State space (combinatorial)
    shipment_states = 4  # IN_TRANSIT, DELAYED, CRITICAL, RESOLVED
    state_space = (shipment_states ** n_shipments) * (2 ** n_shipments)  # status × priority
    
    # Information content
    info_bits = math.log2(total_episode_paths) if total_episode_paths > 0 else 0

    print(f"\n  {B}{C}{'='*55}{X}")
    print(f"  {B}{task_id}: {task_def['name']}{X}")
    print(f"  {D}{task_def['description']}{X}")
    print(f"  {B}{C}{'='*55}{X}")
    
    print(f"\n  {M}Scenario Parameters{X}")
    print(f"    Shipments          : {B}{n_shipments}{X}")
    print(f"    Routes available   : {B}{n_routes}{X}")
    print(f"    Carriers           : {B}{n_carriers}{X}")
    print(f"    Disruptions        : {B}{n_disruptions}{X}")
    print(f"    Max turns          : {B}{max_turns}{X}")
    print(f"    Actions per turn   : {B}{actions_per_turn}{X}")
    
    print(f"\n  {M}Action Space (per step){X}")
    print(f"    reroute_shipment   : {B}{n_reroute:>6}{X}  (ship × route × carrier)")
    print(f"    set_priority       : {B}{n_priority:>6}{X}  (C(n,1)+C(n,2)+C(n,3))")
    print(f"    communicate_eta    : {B}{n_communicate:>6}{X}  + ∞ free-text messages")
    print(f"    escalate           : {B}{n_escalate:>6}{X}")
    print(f"    get_network_status : {B}{n_status:>6}{X}")
    print(f"    end_turn           : {B}{n_end_turn:>6}{X}")
    print(f"    {B}Total discrete     : {G}{total_discrete:>6}{X}")
    
    print(f"\n  {M}Episode Complexity{X}")
    print(f"    Branching/turn     : {B}{branching_per_turn:.2e}{X}  ({total_discrete}^{actions_per_turn})")
    print(f"    Total paths        : {B}{total_episode_paths:.2e}{X}  (branch^{max_turns})")
    print(f"    Information        : {B}{info_bits:.1f} bits{X}")
    print(f"    State space        : {B}{state_space:,}{X}  ({shipment_states}^{n_shipments} × 2^{n_shipments})")

    return {
        "task": task_id,
        "shipments": n_shipments,
        "action_space": total_discrete,
        "branching_factor": branching_per_turn,
        "total_paths": total_episode_paths,
        "info_bits": round(info_bits, 1),
        "state_space": state_space,
    }


def main():
    print(f"\n{B}{'━'*60}{X}")
    print(f"{B}📐  Logistics Shipment RL — Complexity Analysis{X}")
    print(f"{B}{'━'*60}{X}")

    all_results = []
    for task_id, task_def in TASKS.items():
        result = analyze_task(task_id, task_def)
        all_results.append(result)

    # Summary comparison
    print(f"\n\n{B}{'━'*60}{X}")
    print(f"{B}📊  Cross-Task Complexity Comparison{X}")
    print(f"{'━'*60}")
    print(f"  {'Task':<14} {'Ships':>5} {'Actions':>8} {'Branch/Turn':>14} {'Total Paths':>16} {'Bits':>6}")
    print(f"  {'─'*14} {'─'*5} {'─'*8} {'─'*14} {'─'*16} {'─'*6}")
    for r in all_results:
        color = G if r["total_paths"] < 1e20 else Y if r["total_paths"] < 1e50 else R
        print(f"  {r['task']:<14} {r['shipments']:>5} {r['action_space']:>8} "
              f"{r['branching_factor']:>14.2e} {color}{r['total_paths']:>16.2e}{X} "
              f"{r['info_bits']:>6.0f}")

    hardest = max(all_results, key=lambda x: x["total_paths"])
    print(f"\n  {B}Most complex: {R}{hardest['task']}{X}")
    print(f"  {D}With {hardest['total_paths']:.2e} possible episode trajectories,{X}")
    print(f"  {D}this environment is non-trivial for brute-force search.{X}")
    print(f"  {D}Effective exploration requires intelligent credit assignment (GRPO).{X}")
    print(f"\n{B}{'━'*60}{X}\n")


if __name__ == "__main__":
    main()
