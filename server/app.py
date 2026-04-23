"""FastAPI server for the Logistics Shipment RL Environment."""

from fastapi.responses import FileResponse
import os

try:
    from openenv.core.env_server.http_server import create_app
    from .environment import LogisticsShipmentEnvironment, LogisticsAction, LogisticsObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from server.environment import LogisticsShipmentEnvironment, LogisticsAction, LogisticsObservation

app = create_app(
    LogisticsShipmentEnvironment,
    LogisticsAction,
    LogisticsObservation,
    env_name="logistics_shipment_env",
)

# Serve the beautiful dashboard at the root
@app.get("/")
async def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"error": "Dashboard not found", "path": dashboard_path}

@app.get("/health")
async def health():
    return {"status": "ok", "env": "logistics_shipment_env"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
