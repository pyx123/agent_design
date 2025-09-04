"""Unit tests for repository layer."""

import pytest
import json
from datetime import datetime

from src.services.repository import Repository
from src.graph.state import (
    TroubleshootingRequest, InvestigationPlan, InvestigationTask,
    Evidence, Finding, RootCause, Remediation, RemediationAction,
    AgentError, Message, TokenUsage,
    ExecutionStatus, TaskType, TaskStatus, ErrorType
)


@pytest.mark.asyncio
class TestRepository:
    """Test repository operations."""
    
    async def test_create_and_get_request(self, db_session):
        """Test creating and retrieving a request."""
        repo = Repository(db_session)
        
        # Create request
        request = TroubleshootingRequest(
            title="Test issue",
            service="test-service",
            severity="high"
        )
        
        request_id = await repo.create_request(request)
        assert request_id == request.id
        
        # Get request
        retrieved = await repo.get_request(request_id)
        assert retrieved is not None
        assert retrieved["title"] == "Test issue"
        assert retrieved["service"] == "test-service"
    
    async def test_update_request_status(self, db_session):
        """Test updating request status."""
        repo = Repository(db_session)
        
        # Create request
        request = TroubleshootingRequest(title="Test", service="test")
        request_id = await repo.create_request(request)
        
        # Update status
        await repo.update_request_status(request_id, ExecutionStatus.INVESTIGATING)
        
        # Verify
        retrieved = await repo.get_request(request_id)
        assert retrieved["status"] == ExecutionStatus.INVESTIGATING.value
    
    async def test_create_plan(self, db_session, sample_request):
        """Test creating an investigation plan."""
        repo = Repository(db_session)
        
        # Create request first
        request_id = await repo.create_request(sample_request)
        
        # Create plan
        task = InvestigationTask(
            type=TaskType.LOG,
            inputs={"service": "test"}
        )
        plan = InvestigationPlan(
            goals=["Test goal"],
            tasks=[task]
        )
        
        plan_id = await repo.create_plan(request_id, plan)
        assert plan_id == plan.plan_id
        
        # Get latest plan
        retrieved = await repo.get_latest_plan(request_id)
        assert retrieved is not None
        assert retrieved["goals"] == ["Test goal"]
    
    async def test_create_and_update_task(self, db_session, sample_request):
        """Test task operations."""
        repo = Repository(db_session)
        
        # Create request
        request_id = await repo.create_request(sample_request)
        
        # Create task
        task = InvestigationTask(
            type=TaskType.KPI,
            inputs={"metrics": ["cpu", "memory"]}
        )
        
        task_id = await repo.create_task(request_id, task)
        assert task_id == task.task_id
        
        # Update task status
        await repo.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            "Found high CPU usage"
        )
        
        # Get tasks
        tasks = await repo.get_tasks(request_id)
        assert len(tasks) == 1
        assert tasks[0]["status"] == TaskStatus.COMPLETED.value
        assert tasks[0]["result_summary"] == "Found high CPU usage"
    
    async def test_create_evidence(self, db_session, sample_request):
        """Test evidence creation."""
        repo = Repository(db_session)
        
        # Create request
        request_id = await repo.create_request(sample_request)
        
        # Create evidence
        evidence = Evidence(
            source=TaskType.LOG,
            summary="Error spike detected",
            quality_score=0.8
        )
        
        evidence_id = await repo.create_evidence(request_id, evidence)
        assert evidence_id == evidence.evidence_id
        
        # Get evidence
        evidence_list = await repo.get_evidence(request_id)
        assert len(evidence_list) == 1
        assert evidence_list[0]["summary"] == "Error spike detected"
    
    async def test_save_root_cause_and_remediation(self, db_session, sample_request):
        """Test saving analysis results."""
        repo = Repository(db_session)
        
        # Create request
        request_id = await repo.create_request(sample_request)
        
        # Save root cause
        root_cause = RootCause(
            hypothesis="Connection pool exhaustion",
            confidence=0.9,
            affected_components=["api", "database"]
        )
        
        rc_id = await repo.save_root_cause(request_id, root_cause)
        assert rc_id is not None
        
        # Save remediation
        action = RemediationAction(
            description="Increase pool size",
            risk_level="low"
        )
        remediation = Remediation(
            actions=[action],
            risk="low"
        )
        
        rem_id = await repo.save_remediation(request_id, remediation)
        assert rem_id is not None
    
    async def test_log_error(self, db_session, sample_request):
        """Test error logging."""
        repo = Repository(db_session)
        
        # Create request
        request_id = await repo.create_request(sample_request)
        
        # Log error
        error = AgentError(
            agent="log_agent",
            error_type=ErrorType.TIMEOUT,
            message="Query timeout",
            retriable=True
        )
        
        await repo.log_error(request_id, error)
        
        # Note: No retrieval method for errors in current implementation
        # This test just ensures no exceptions
    
    async def test_token_usage(self, db_session, sample_request):
        """Test token usage tracking."""
        repo = Repository(db_session)
        
        # Create request
        request_id = await repo.create_request(sample_request)
        
        # Log usage
        usage = TokenUsage(
            agent="planner",
            prompt_tokens=100,
            completion_tokens=50,
            tool_calls=2,
            wall_time_ms=1500
        )
        
        await repo.log_token_usage(request_id, usage)
        
        # Get summary
        summary = await repo.get_token_usage_summary(request_id)
        assert summary["total_prompt_tokens"] == 100
        assert summary["total_completion_tokens"] == 50
        assert summary["total_calls"] == 1