"""GraphState and related data models for the DevOps Agent system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class ExecutionStatus(str, Enum):
    """Execution status for troubleshooting requests."""
    INIT = "init"
    PLANNING = "planning"
    INVESTIGATING = "investigating"
    ANALYZING = "analyzing"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of investigation tasks."""
    LOG = "log"
    ALARM = "alarm"
    KPI = "kpi"
    KNOWLEDGE = "knowledge"
    CHANGE = "change"


class TaskStatus(str, Enum):
    """Status of individual tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Severity(str, Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Environment(str, Enum):
    """Deployment environments."""
    PROD = "prod"
    STAGING = "staging"
    DEV = "dev"


class ErrorType(str, Enum):
    """Types of errors that can occur."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"


# Pydantic models for structured data

class TimeRange(BaseModel):
    """Time range for investigation."""
    from_time: datetime = Field(alias="from")
    to_time: datetime = Field(alias="to")
    
    model_config = ConfigDict(populate_by_name=True)


class TroubleshootingRequest(BaseModel):
    """User's troubleshooting request."""
    id: str = Field(default_factory=lambda: f"req_{uuid4().hex[:12]}")
    tenant_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    service: str
    environment: Environment = Environment.PROD
    severity: Severity = Severity.MEDIUM
    time_range: Optional[TimeRange] = None
    artifacts_hints: Optional[Dict[str, Any]] = None
    mode: str = "async"  # sync or async
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InvestigationTask(BaseModel):
    """Individual investigation task."""
    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    type: TaskType
    inputs: Dict[str, Any]
    hypotheses: List[str] = Field(default_factory=list)
    priority: int = 1
    timeout_s: int = 30
    status: TaskStatus = TaskStatus.PENDING
    result_summary: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class InvestigationPlan(BaseModel):
    """Investigation plan created by planner."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")
    created_by: str = "planner"
    goals: List[str]
    tasks: List[InvestigationTask]
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Evidence(BaseModel):
    """Evidence collected during investigation."""
    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    source: TaskType
    summary: str
    raw_ref: Optional[Dict[str, Any]] = None  # Query info, server ref, etc.
    time_window: Optional[TimeRange] = None
    quality_score: float = 0.5
    task_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Finding(BaseModel):
    """Findings derived from evidence."""
    finding_id: str = Field(default_factory=lambda: f"find_{uuid4().hex[:12]}")
    hypothesis_ref: str
    confidence: float
    impact_scope: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)  # evidence_ids
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RootCause(BaseModel):
    """Root cause analysis result."""
    hypothesis: str
    confidence: float
    affected_components: List[str]
    time_correlation: Optional[Dict[str, Any]] = None
    change_correlation: Optional[Dict[str, Any]] = None


class RemediationAction(BaseModel):
    """Individual remediation action."""
    description: str
    command: Optional[str] = None
    risk_level: str = "medium"
    estimated_duration: Optional[str] = None


class Remediation(BaseModel):
    """Remediation plan."""
    actions: List[RemediationAction]
    risk: str = "medium"
    required_approvals: List[str] = Field(default_factory=list)
    validation_steps: List[str] = Field(default_factory=list)
    rollback_steps: List[str] = Field(default_factory=list)


class AgentError(BaseModel):
    """Error from agent execution."""
    agent: str
    error_type: ErrorType
    message: str
    retriable: bool = True
    attempt: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Message(BaseModel):
    """Conversation message."""
    role: str  # user, assistant, system
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TokenUsage(BaseModel):
    """Token usage tracking."""
    agent: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    wall_time_ms: int = 0


class BranchDecision(BaseModel):
    """Record of a branching decision in the graph."""
    node: str
    condition: str
    result: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# LangGraph State Definition

class GraphState(TypedDict, total=False):
    """
    Complete state for a troubleshooting session.
    This is the state that flows through the LangGraph nodes.
    """
    # Basic information
    request: dict  # TroubleshootingRequest as dict
    session_id: str
    tenant_id: Optional[str]
    
    # Execution plan
    plan: Optional[dict]  # InvestigationPlan as dict
    tasks: List[dict]  # List of InvestigationTask as dict
    current_task_id: Optional[str]
    
    # Evidence collection
    evidence: List[dict]  # List of Evidence as dict
    findings: List[dict]  # List of Finding as dict
    evidence_graph: Dict[str, List[str]]  # Evidence relationship graph
    
    # Execution control
    next_actions: List[str]  # Actions like ["query_logs", "query_kpis", "summarize"]
    execution_path: List[str]  # Path of nodes executed
    branch_history: List[dict]  # List of BranchDecision as dict
    
    # Results
    root_cause: Optional[dict]  # RootCause as dict
    remediation: Optional[dict]  # Remediation as dict
    confidence_scores: Dict[str, float]  # Hypothesis -> confidence mapping
    
    # Context management
    history: List[dict]  # List of Message as dict
    context_window: int
    memory_summary: Optional[str]
    
    # Resource management
    concurrency_limit: int
    token_usage: List[dict]  # List of TokenUsage as dict
    
    # Error handling
    errors: List[dict]  # List of AgentError as dict
    retry_attempts: Dict[str, int]  # agent -> retry count
    
    # Status flags
    status: str  # ExecutionStatus
    done: bool
    terminated_reason: Optional[str]
    
    # Metadata
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime


# State transition management

class StateTransitions:
    """Valid state transitions for the system."""
    
    VALID_TRANSITIONS = {
        ExecutionStatus.INIT: [ExecutionStatus.PLANNING],
        ExecutionStatus.PLANNING: [ExecutionStatus.INVESTIGATING, ExecutionStatus.FAILED],
        ExecutionStatus.INVESTIGATING: [
            ExecutionStatus.ANALYZING,
            ExecutionStatus.PLANNING,  # Re-planning
            ExecutionStatus.FAILED
        ],
        ExecutionStatus.ANALYZING: [ExecutionStatus.SUMMARIZING, ExecutionStatus.INVESTIGATING],
        ExecutionStatus.SUMMARIZING: [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED],
        ExecutionStatus.FAILED: [ExecutionStatus.PLANNING],  # Allow retry
        ExecutionStatus.COMPLETED: [],  # Terminal state
        ExecutionStatus.CANCELLED: []  # Terminal state
    }
    
    @classmethod
    def is_valid_transition(cls, from_status: ExecutionStatus, to_status: ExecutionStatus) -> bool:
        """Check if a state transition is valid."""
        return to_status in cls.VALID_TRANSITIONS.get(from_status, [])
    
    @classmethod
    def get_valid_next_states(cls, current_status: ExecutionStatus) -> List[ExecutionStatus]:
        """Get list of valid next states from current state."""
        return cls.VALID_TRANSITIONS.get(current_status, [])


# Helper functions for state management

def create_initial_state(request: TroubleshootingRequest) -> GraphState:
    """Create initial GraphState from a troubleshooting request."""
    now = datetime.utcnow().isoformat()
    return GraphState(
        request=request.model_dump(),
        session_id=f"session_{uuid4().hex[:12]}",
        tenant_id=request.tenant_id,
        plan=None,
        tasks=[],
        current_task_id=None,
        evidence=[],
        findings=[],
        evidence_graph={},
        next_actions=[],
        execution_path=[],
        branch_history=[],
        root_cause=None,
        remediation=None,
        confidence_scores={},
        history=[],
        context_window=10000,
        memory_summary=None,
        concurrency_limit=3,
        token_usage=[],
        errors=[],
        retry_attempts={},
        status=ExecutionStatus.INIT.value,
        done=False,
        terminated_reason=None,
        created_at=now,
        updated_at=now
    )


def update_state_timestamp(state: GraphState) -> GraphState:
    """Update the state's updated_at timestamp."""
    state["updated_at"] = datetime.utcnow().isoformat()
    return state


def add_execution_path(state: GraphState, node: str) -> GraphState:
    """Add a node to the execution path."""
    if "execution_path" not in state:
        state["execution_path"] = []
    state["execution_path"].append(node)
    return update_state_timestamp(state)


def add_error(state: GraphState, error: AgentError) -> GraphState:
    """Add an error to the state."""
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(error.model_dump())
    return update_state_timestamp(state)


def is_terminal_state(state: GraphState) -> bool:
    """Check if the state is in a terminal state."""
    return state.get("status") in [
        ExecutionStatus.COMPLETED.value,
        ExecutionStatus.CANCELLED.value
    ] or state.get("done", False)