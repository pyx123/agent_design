"""WebSocket support for real-time updates."""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import (
    WebSocketMessage,
    StatusUpdate,
    EvidenceCollected,
    TaskCompleted,
    AnalysisComplete,
)
from src.services import get_db_session, Repository


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        # Maps request_id to set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Maps websocket to set of request_ids it's subscribed to
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.subscriptions[websocket] = set()
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        # Remove from all subscriptions
        if websocket in self.subscriptions:
            for request_id in self.subscriptions[websocket]:
                if request_id in self.active_connections:
                    self.active_connections[request_id].discard(websocket)
                    if not self.active_connections[request_id]:
                        del self.active_connections[request_id]
            del self.subscriptions[websocket]
    
    def subscribe(self, websocket: WebSocket, request_id: str):
        """Subscribe a WebSocket to a request."""
        if request_id not in self.active_connections:
            self.active_connections[request_id] = set()
        self.active_connections[request_id].add(websocket)
        
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(request_id)
    
    def unsubscribe(self, websocket: WebSocket, request_id: str):
        """Unsubscribe a WebSocket from a request."""
        if request_id in self.active_connections:
            self.active_connections[request_id].discard(websocket)
            if not self.active_connections[request_id]:
                del self.active_connections[request_id]
        
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(request_id)
    
    async def send_to_request(self, request_id: str, message: WebSocketMessage):
        """Send a message to all connections subscribed to a request."""
        if request_id in self.active_connections:
            # Send to all connections in parallel
            tasks = []
            for connection in self.active_connections[request_id].copy():
                tasks.append(self._send_safe(connection, message))
            await asyncio.gather(*tasks)
    
    async def _send_safe(self, websocket: WebSocket, message: WebSocketMessage):
        """Send a message safely, handling disconnections."""
        try:
            await websocket.send_json(message.model_dump(mode="json"))
        except Exception:
            # Connection is broken, remove it
            self.disconnect(websocket)
    
    async def broadcast_status_update(self, request_id: str, status: str, progress: float = 0.0):
        """Broadcast a status update."""
        update = StatusUpdate(
            request_id=request_id,
            status=status,
            progress=progress
        )
        message = WebSocketMessage(
            type="status_update",
            data=update.model_dump()
        )
        await self.send_to_request(request_id, message)
    
    async def broadcast_evidence_collected(
        self,
        request_id: str,
        evidence: dict,
        total_evidence: int
    ):
        """Broadcast evidence collection."""
        event = EvidenceCollected(
            request_id=request_id,
            evidence=evidence,
            total_evidence=total_evidence
        )
        message = WebSocketMessage(
            type="evidence_collected",
            data=event.model_dump()
        )
        await self.send_to_request(request_id, message)
    
    async def broadcast_task_completed(
        self,
        request_id: str,
        task_id: str,
        task_type: str,
        result: str = None
    ):
        """Broadcast task completion."""
        event = TaskCompleted(
            request_id=request_id,
            task_id=task_id,
            task_type=task_type,
            result=result
        )
        message = WebSocketMessage(
            type="task_completed",
            data=event.model_dump()
        )
        await self.send_to_request(request_id, message)
    
    async def broadcast_analysis_complete(
        self,
        request_id: str,
        root_cause: dict = None,
        remediation: dict = None,
        confidence: float = 0.0
    ):
        """Broadcast analysis completion."""
        event = AnalysisComplete(
            request_id=request_id,
            root_cause=root_cause,
            remediation=remediation,
            confidence=confidence
        )
        message = WebSocketMessage(
            type="analysis_complete",
            data=event.model_dump()
        )
        await self.send_to_request(request_id, message)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db_session)
):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    repo = Repository(db)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                # Subscribe to a request
                request_id = data.get("data", {}).get("request_id")
                if request_id:
                    # Verify request exists
                    request = await repo.get_request(request_id)
                    if request:
                        manager.subscribe(websocket, request_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "data": {"request_id": request_id}
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": "Request not found"}
                        })
            
            elif data.get("type") == "unsubscribe":
                # Unsubscribe from a request
                request_id = data.get("data", {}).get("request_id")
                if request_id:
                    manager.unsubscribe(websocket, request_id)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "data": {"request_id": request_id}
                    })
            
            elif data.get("type") == "ping":
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "data": {"timestamp": datetime.utcnow().isoformat()}
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })
        manager.disconnect(websocket)


# Event hooks for graph execution
async def on_status_change(request_id: str, status: str):
    """Hook called when request status changes."""
    await manager.broadcast_status_update(request_id, status)


async def on_evidence_collected(request_id: str, evidence: dict, total: int):
    """Hook called when evidence is collected."""
    await manager.broadcast_evidence_collected(request_id, evidence, total)


async def on_task_completed(request_id: str, task: dict):
    """Hook called when a task completes."""
    await manager.broadcast_task_completed(
        request_id,
        task["task_id"],
        task["type"],
        task.get("result_summary")
    )


async def on_analysis_complete(request_id: str, state: dict):
    """Hook called when analysis completes."""
    root_cause = state.get("root_cause")
    remediation = state.get("remediation")
    confidence = root_cause.get("confidence", 0.0) if root_cause else 0.0
    
    await manager.broadcast_analysis_complete(
        request_id,
        root_cause,
        remediation,
        confidence
    )