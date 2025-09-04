"""API module for DevOps Agent."""

from .models import (
    TroubleshootingRequestCreate,
    TroubleshootingRequestResponse,
    TroubleshootingSummary,
    TroubleshootingResult,
    UpdateTroubleshootingRequest,
    UpdateRequestAction,
    EvidenceFilter,
    ReportFormat,
    ErrorResponse,
    HealthStatus,
    WebSocketMessage,
    StatusUpdate,
    EvidenceCollected,
    TaskCompleted,
    AnalysisComplete,
)
from .routes import router
from .websocket import manager, websocket_endpoint

__all__ = [
    # Models
    "TroubleshootingRequestCreate",
    "TroubleshootingRequestResponse",
    "TroubleshootingSummary",
    "TroubleshootingResult",
    "UpdateTroubleshootingRequest",
    "UpdateRequestAction",
    "EvidenceFilter",
    "ReportFormat",
    "ErrorResponse",
    "HealthStatus",
    "WebSocketMessage",
    "StatusUpdate",
    "EvidenceCollected",
    "TaskCompleted",
    "AnalysisComplete",
    # Routes
    "router",
    # WebSocket
    "manager",
    "websocket_endpoint",
]