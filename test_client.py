import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models import LogisticsAction
from client import LogisticsShipmentEnv

async def main():
    print("🚛 Connecting to Logistics Environment...")
    
    # Connects to the local FastAPI server we'll start on port 8000
    async with LogisticsShipmentEnv(base_url="http://127.0.0.1:8000") as env:
        
        # 1. Reset the environment (starts a new 5-hour crisis episode)
        obs = await env.reset()
        print("\n--- 📡 INITIAL SNAPSHOT ---")
        print(obs.network_snapshot)
        print(f"Active Disruptions: {obs.active_disruptions_count}")
        print(f"Delayed Shipments: {obs.delayed_shipments}")
        
        # 2. Make a dummy AI decision
        print("\n--- 🤖 SENDING AI ACTION ---")
        action = LogisticsAction(
            reasoning="The situation looks bad. Let's fast-track shipment SHIP-001 and hope for the best.",
            rerouting_decisions={},
            priority_shipments=["SHIP-001"],
            customer_communications={"SHIP-001": "Hang tight, we are prioritizing your shipment!"},
            escalations=[]
        )
        
        # 3. Step the environment forward 1 hour
        result = await env.step(action)
        
        print("\n--- 🏆 STEP RESULT ---")
        print(f"Feedback: {result.observation.previous_action_feedback}")
        print(f"Reward Score: {result.reward:.3f} / 1.0")
        print(f"Delay Saved (Hours): {result.observation.total_delay_saved_hours}")

if __name__ == "__main__":
    asyncio.run(main())
