"""API request/response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from src.graph.state import (
    Severity, Environment, ExecutionStatus,
    TimeRange, TroubleshootingRequest as BaseRequest
)


class TroubleshootingRequestCreate(BaseModel):
    """API model for creating a troubleshooting request."""
    title: str = Field(..., description="Issue title")
    description: Optional[str] = Field(None, description="Detailed description")
    service: str = Field(..., description="Affected service name")
    environment: Environment = Field(Environment.PROD, description="Environment")
    severity: Severity = Field(Severity.MEDIUM, description="Issue severity")
    time_range: Optional[TimeRange] = Field(None, description="Time range to investigate")
    artifacts_hints: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    mode: str = Field("async", description="Execution mode: sync or async")
    
    model_config = ConfigDict(use_enum_values=True)


class TroubleshootingRequestResponse(BaseModel):
    """Response after creating a request."""
    id: str
    status: ExecutionStatus
    mode: str
    created_at: datetime


class TroubleshootingSummary(BaseModel):
    """Summary of a troubleshooting request for list views."""
    id: str
    title: str
    service: str
    environment: str
    severity: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    evidence_count: int = 0
    has_root_cause: bool = False


class TroubleshootingResult(BaseModel):
    """Complete troubleshooting result."""
    id: str
    status: ExecutionStatus
    request: Dict[str, Any]
    plan: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    root_cause: Optional[Dict[str, Any]] = None
    remediation: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    model_config = ConfigDict(use_enum_values=True)


class UpdateRequestAction(str, Enum):
    """Actions for updating a request."""
    CANCEL = "cancel"
    RETRY = "retry"
    CONTINUE = "continue"


class UpdateTroubleshootingRequest(BaseModel):
    """Model for updating a troubleshooting request."""
    action: UpdateRequestAction
    additional_context: Optional[str] = None


class EvidenceFilter(BaseModel):
    """Filter for evidence queries."""
    source: Optional[str] = None


class ReportFormat(str, Enum):
    """Report output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, str]
    timestamp: datetime


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatusUpdate(BaseModel):
    """Status update for WebSocket."""
    request_id: str
    status: ExecutionStatus
    progress: float = 0.0
    message: Optional[str] = None


class EvidenceCollected(BaseModel):
    """Evidence collection notification."""
    request_id: str
    evidence: Dict[str, Any]
    total_evidence: int


class TaskCompleted(BaseModel):
    """Task completion notification."""
    request_id: str
    task_id: str
    task_type: str
    result: Optional[str] = None


class AnalysisComplete(BaseModel):
    """Analysis completion notification."""
    request_id: str
    root_cause: Optional[Dict[str, Any]] = None
    remediation: Optional[Dict[str, Any]] = None
    confidence: float = 0.0