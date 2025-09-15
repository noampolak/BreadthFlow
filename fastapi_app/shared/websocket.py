from fastapi import WebSocket
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime


class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, metadata: Dict[str, Any] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = metadata or {}
        print(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_pipeline_update(self, pipeline_id: str, status: str):
        message = {
            "type": "pipeline_update",
            "pipeline_id": pipeline_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message)

    async def broadcast_signal_update(self, signal_data: dict):
        message = {"type": "signal_update", "data": signal_data, "timestamp": datetime.now().isoformat()}
        await self.broadcast(message)

    async def broadcast_dashboard_update(self, dashboard_data: dict):
        message = {"type": "dashboard_update", "data": dashboard_data, "timestamp": datetime.now().isoformat()}
        await self.broadcast(message)
