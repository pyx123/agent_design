"""Unit tests for state management."""

import pytest
from datetime import datetime

from src.graph.state import (
    ExecutionStatus, TaskType, TaskStatus, Severity,
    TroubleshootingRequest, InvestigationPlan, InvestigationTask,
    Evidence, Finding, RootCause, Remediation, RemediationAction,
    StateTransitions, create_initial_state, is_terminal_state,
    add_execution_path, add_error, AgentError, ErrorType
)


class TestDataModels:
    """Test data model classes."""
    
    def test_troubleshooting_request_creation(self):
        """Test creating a troubleshooting request."""
        request = TroubleshootingRequest(
            title="Test issue",
            service="test-service",
            description="Test description",
            severity=Severity.HIGH
        )
        
        assert request.title == "Test issue"
        assert request.service == "test-service"
        assert request.severity == Severity.HIGH
        assert request.environment.value == "prod"  # Default
        assert request.id.startswith("req_")
    
    def test_investigation_plan_creation(self):
        """Test creating an investigation plan."""
        task = InvestigationTask(
            type=TaskType.LOG,
            inputs={"service": "test"},
            hypotheses=["Test hypothesis"]
        )
        
        plan = InvestigationPlan(
            goals=["Test goal"],
            tasks=[task]
        )
        
        assert len(plan.goals) == 1
        assert len(plan.tasks) == 1
        assert plan.tasks[0].type == TaskType.LOG
        assert plan.version == 1
    
    def test_evidence_creation(self):
        """Test creating evidence."""
        evidence = Evidence(
            source=TaskType.KPI,
            summary="High CPU usage detected",
            quality_score=0.9
        )
        
        assert evidence.source == TaskType.KPI
        assert evidence.quality_score == 0.9
        assert evidence.evidence_id.startswith("ev_")
    
    def test_root_cause_creation(self):
        """Test creating root cause."""
        root_cause = RootCause(
            hypothesis="Database connection pool exhaustion",
            confidence=0.85,
            affected_components=["payment-service", "database"]
        )
        
        assert root_cause.confidence == 0.85
        assert len(root_cause.affected_components) == 2
    
    def test_remediation_creation(self):
        """Test creating remediation."""
        action = RemediationAction(
            description="Increase connection pool size",
            command="kubectl set env deployment/payment-service DB_POOL_SIZE=100",
            risk_level="low"
        )
        
        remediation = Remediation(
            actions=[action],
            risk="low",
            validation_steps=["Check connection metrics"]
        )
        
        assert len(remediation.actions) == 1
        assert remediation.actions[0].risk_level == "low"
        assert len(remediation.validation_steps) == 1


class TestStateTransitions:
    """Test state transition logic."""
    
    def test_valid_transitions(self):
        """Test valid state transitions."""
        assert StateTransitions.is_valid_transition(
            ExecutionStatus.INIT, ExecutionStatus.PLANNING
        )
        assert StateTransitions.is_valid_transition(
            ExecutionStatus.PLANNING, ExecutionStatus.INVESTIGATING
        )
        assert StateTransitions.is_valid_transition(
            ExecutionStatus.INVESTIGATING, ExecutionStatus.ANALYZING
        )
        assert StateTransitions.is_valid_transition(
            ExecutionStatus.SUMMARIZING, ExecutionStatus.COMPLETED
        )
    
    def test_invalid_transitions(self):
        """Test invalid state transitions."""
        assert not StateTransitions.is_valid_transition(
            ExecutionStatus.INIT, ExecutionStatus.COMPLETED
        )
        assert not StateTransitions.is_valid_transition(
            ExecutionStatus.COMPLETED, ExecutionStatus.PLANNING
        )
    
    def test_terminal_states(self):
        """Test terminal states have no valid transitions."""
        assert StateTransitions.get_valid_next_states(ExecutionStatus.COMPLETED) == []
        assert StateTransitions.get_valid_next_states(ExecutionStatus.CANCELLED) == []


class TestStateHelpers:
    """Test state helper functions."""
    
    def test_create_initial_state(self, sample_request):
        """Test initial state creation."""
        state = create_initial_state(sample_request)
        
        assert state["request"]["id"] == sample_request.id
        assert state["status"] == ExecutionStatus.INIT.value
        assert state["done"] is False
        assert state["evidence"] == []
        assert state["tasks"] == []
        assert state["session_id"].startswith("session_")
    
    def test_is_terminal_state(self):
        """Test terminal state detection."""
        state = {"status": ExecutionStatus.COMPLETED.value}
        assert is_terminal_state(state)
        
        state = {"status": ExecutionStatus.CANCELLED.value}
        assert is_terminal_state(state)
        
        state = {"status": ExecutionStatus.INVESTIGATING.value, "done": True}
        assert is_terminal_state(state)
        
        state = {"status": ExecutionStatus.PLANNING.value}
        assert not is_terminal_state(state)
    
    def test_add_execution_path(self):
        """Test adding to execution path."""
        state = {}
        state = add_execution_path(state, "planner")
        
        assert "execution_path" in state
        assert state["execution_path"] == ["planner"]
        
        state = add_execution_path(state, "log_agent")
        assert state["execution_path"] == ["planner", "log_agent"]
    
    def test_add_error(self):
        """Test adding errors to state."""
        state = {}
        error = AgentError(
            agent="test_agent",
            error_type=ErrorType.TIMEOUT,
            message="Request timed out"
        )
        
        state = add_error(state, error)
        
        assert "errors" in state
        assert len(state["errors"]) == 1
        assert state["errors"][0]["agent"] == "test_agent"
        assert state["errors"][0]["error_type"] == ErrorType.TIMEOUT.value