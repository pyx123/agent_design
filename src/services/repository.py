"""Repository layer for database operations."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

import aiosqlite
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.graph.state import (
    TroubleshootingRequest, InvestigationPlan, InvestigationTask,
    Evidence, Finding, RootCause, Remediation, AgentError,
    Message, TokenUsage, ExecutionStatus, TaskStatus
)


class Repository:
    """Repository for all database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # Troubleshooting Requests
    
    async def create_request(self, request: TroubleshootingRequest) -> str:
        """Create a new troubleshooting request."""
        sql = """
        INSERT INTO troubleshooting_requests 
        (id, tenant_id, title, description, service, environment, severity, 
         time_from, time_to, artifacts_hints, mode, status, created_at)
        VALUES 
        (:id, :tenant_id, :title, :description, :service, :environment, :severity,
         :time_from, :time_to, :artifacts_hints, :mode, :status, :created_at)
        """
        
        params = {
            "id": request.id,
            "tenant_id": request.tenant_id,
            "title": request.title,
            "description": request.description,
            "service": request.service,
            "environment": request.environment.value,
            "severity": request.severity.value,
            "time_from": request.time_range.from_time.isoformat() if request.time_range else None,
            "time_to": request.time_range.to_time.isoformat() if request.time_range else None,
            "artifacts_hints": json.dumps(request.artifacts_hints) if request.artifacts_hints else None,
            "mode": request.mode,
            "status": ExecutionStatus.INIT.value,
            "created_at": request.created_at.isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return request.id
    
    async def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a troubleshooting request by ID."""
        sql = "SELECT * FROM troubleshooting_requests WHERE id = :id"
        result = await self.session.execute(text(sql), {"id": request_id})
        row = result.first()
        
        if row:
            data = dict(row._mapping)
            # Parse JSON fields
            if data.get("artifacts_hints"):
                data["artifacts_hints"] = json.loads(data["artifacts_hints"])
            return data
        return None
    
    async def update_request_status(self, request_id: str, status: ExecutionStatus) -> None:
        """Update request status."""
        sql = """
        UPDATE troubleshooting_requests 
        SET status = :status, updated_at = :updated_at
        WHERE id = :id
        """
        await self.session.execute(text(sql), {
            "id": request_id,
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        })
        await self.session.commit()
    
    async def list_requests(
        self, 
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List troubleshooting requests with filters."""
        sql = "SELECT * FROM troubleshooting_requests WHERE 1=1"
        params = {"limit": limit, "offset": offset}
        
        if tenant_id:
            sql += " AND tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id
        
        if status:
            sql += " AND status = :status"
            params["status"] = status
        
        if service:
            sql += " AND service = :service"
            params["service"] = service
        
        sql += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        
        result = await self.session.execute(text(sql), params)
        rows = result.fetchall()
        
        requests = []
        for row in rows:
            data = dict(row._mapping)
            if data.get("artifacts_hints"):
                data["artifacts_hints"] = json.loads(data["artifacts_hints"])
            requests.append(data)
        
        return requests
    
    # Investigation Plans
    
    async def create_plan(self, request_id: str, plan: InvestigationPlan) -> str:
        """Create an investigation plan."""
        sql = """
        INSERT INTO investigation_plans
        (plan_id, request_id, created_by, goals_json, plan_json, version, created_at)
        VALUES
        (:plan_id, :request_id, :created_by, :goals_json, :plan_json, :version, :created_at)
        """
        
        params = {
            "plan_id": plan.plan_id,
            "request_id": request_id,
            "created_by": plan.created_by,
            "goals_json": json.dumps(plan.goals),
            "plan_json": json.dumps(plan.model_dump()),
            "version": plan.version,
            "created_at": plan.created_at.isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return plan.plan_id
    
    async def get_latest_plan(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest plan for a request."""
        sql = """
        SELECT * FROM investigation_plans 
        WHERE request_id = :request_id 
        ORDER BY version DESC, created_at DESC 
        LIMIT 1
        """
        result = await self.session.execute(text(sql), {"request_id": request_id})
        row = result.first()
        
        if row:
            data = dict(row._mapping)
            data["goals"] = json.loads(data["goals_json"])
            data["plan"] = json.loads(data["plan_json"])
            return data
        return None
    
    # Investigation Tasks
    
    async def create_task(self, request_id: str, task: InvestigationTask, plan_id: Optional[str] = None) -> str:
        """Create an investigation task."""
        sql = """
        INSERT INTO investigation_tasks
        (task_id, request_id, plan_id, type, inputs_json, hypotheses_json, 
         priority, timeout_s, status, created_at)
        VALUES
        (:task_id, :request_id, :plan_id, :type, :inputs_json, :hypotheses_json,
         :priority, :timeout_s, :status, :created_at)
        """
        
        params = {
            "task_id": task.task_id,
            "request_id": request_id,
            "plan_id": plan_id,
            "type": task.type.value,
            "inputs_json": json.dumps(task.inputs),
            "hypotheses_json": json.dumps(task.hypotheses),
            "priority": task.priority,
            "timeout_s": task.timeout_s,
            "status": task.status.value,
            "created_at": datetime.utcnow().isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return task.task_id
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus,
        result_summary: Optional[str] = None
    ) -> None:
        """Update task status."""
        sql = """
        UPDATE investigation_tasks
        SET status = :status, result_summary = :result_summary, 
            updated_at = :updated_at
        """
        
        params = {
            "task_id": task_id,
            "status": status.value,
            "result_summary": result_summary,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == TaskStatus.RUNNING:
            sql += ", started_at = :started_at"
            params["started_at"] = datetime.utcnow().isoformat()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            sql += ", completed_at = :completed_at"
            params["completed_at"] = datetime.utcnow().isoformat()
        
        sql += " WHERE task_id = :task_id"
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
    
    async def get_tasks(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a request."""
        sql = """
        SELECT * FROM investigation_tasks 
        WHERE request_id = :request_id 
        ORDER BY created_at ASC
        """
        result = await self.session.execute(text(sql), {"request_id": request_id})
        rows = result.fetchall()
        
        tasks = []
        for row in rows:
            data = dict(row._mapping)
            data["inputs"] = json.loads(data["inputs_json"])
            data["hypotheses"] = json.loads(data["hypotheses_json"])
            tasks.append(data)
        
        return tasks
    
    # Evidence
    
    async def create_evidence(self, request_id: str, evidence: Evidence) -> str:
        """Create evidence record."""
        sql = """
        INSERT INTO evidences
        (evidence_id, request_id, task_id, source, summary, raw_ref_json,
         time_from, time_to, quality_score, created_at)
        VALUES
        (:evidence_id, :request_id, :task_id, :source, :summary, :raw_ref_json,
         :time_from, :time_to, :quality_score, :created_at)
        """
        
        params = {
            "evidence_id": evidence.evidence_id,
            "request_id": request_id,
            "task_id": evidence.task_id,
            "source": evidence.source.value,
            "summary": evidence.summary,
            "raw_ref_json": json.dumps(evidence.raw_ref) if evidence.raw_ref else None,
            "time_from": evidence.time_window.from_time.isoformat() if evidence.time_window else None,
            "time_to": evidence.time_window.to_time.isoformat() if evidence.time_window else None,
            "quality_score": evidence.quality_score,
            "created_at": evidence.created_at.isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return evidence.evidence_id
    
    async def get_evidence(self, request_id: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evidence for a request."""
        sql = "SELECT * FROM evidences WHERE request_id = :request_id"
        params = {"request_id": request_id}
        
        if source:
            sql += " AND source = :source"
            params["source"] = source
        
        sql += " ORDER BY created_at DESC"
        
        result = await self.session.execute(text(sql), params)
        rows = result.fetchall()
        
        evidence_list = []
        for row in rows:
            data = dict(row._mapping)
            if data.get("raw_ref_json"):
                data["raw_ref"] = json.loads(data["raw_ref_json"])
            evidence_list.append(data)
        
        return evidence_list
    
    # Findings
    
    async def create_finding(self, request_id: str, finding: Finding) -> str:
        """Create a finding."""
        sql = """
        INSERT INTO findings
        (finding_id, request_id, hypothesis_ref, confidence, 
         impact_scope_json, supporting_evidence_json, created_at)
        VALUES
        (:finding_id, :request_id, :hypothesis_ref, :confidence,
         :impact_scope_json, :supporting_evidence_json, :created_at)
        """
        
        params = {
            "finding_id": finding.finding_id,
            "request_id": request_id,
            "hypothesis_ref": finding.hypothesis_ref,
            "confidence": finding.confidence,
            "impact_scope_json": json.dumps(finding.impact_scope),
            "supporting_evidence_json": json.dumps(finding.supporting_evidence),
            "created_at": finding.created_at.isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return finding.finding_id
    
    # Root Cause & Remediation
    
    async def save_root_cause(self, request_id: str, root_cause: RootCause) -> str:
        """Save root cause analysis."""
        sql = """
        INSERT OR REPLACE INTO root_causes
        (id, request_id, hypothesis, confidence, affected_components_json,
         time_correlation_json, change_correlation_json, created_at)
        VALUES
        (:id, :request_id, :hypothesis, :confidence, :affected_components_json,
         :time_correlation_json, :change_correlation_json, :created_at)
        """
        
        params = {
            "id": f"rc_{uuid4().hex[:12]}",
            "request_id": request_id,
            "hypothesis": root_cause.hypothesis,
            "confidence": root_cause.confidence,
            "affected_components_json": json.dumps(root_cause.affected_components),
            "time_correlation_json": json.dumps(root_cause.time_correlation) if root_cause.time_correlation else None,
            "change_correlation_json": json.dumps(root_cause.change_correlation) if root_cause.change_correlation else None,
            "created_at": datetime.utcnow().isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return params["id"]
    
    async def save_remediation(self, request_id: str, remediation: Remediation) -> str:
        """Save remediation plan."""
        sql = """
        INSERT OR REPLACE INTO remediations
        (id, request_id, actions_json, risk_level, required_approvals_json,
         validation_steps_json, created_at)
        VALUES
        (:id, :request_id, :actions_json, :risk_level, :required_approvals_json,
         :validation_steps_json, :created_at)
        """
        
        params = {
            "id": f"rem_{uuid4().hex[:12]}",
            "request_id": request_id,
            "actions_json": json.dumps([a.model_dump() for a in remediation.actions]),
            "risk_level": remediation.risk,
            "required_approvals_json": json.dumps(remediation.required_approvals),
            "validation_steps_json": json.dumps(remediation.validation_steps),
            "created_at": datetime.utcnow().isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
        return params["id"]
    
    # Errors
    
    async def log_error(self, request_id: str, error: AgentError) -> None:
        """Log an agent error."""
        sql = """
        INSERT INTO agent_errors
        (id, request_id, agent, error_type, message, retriable, attempt, created_at)
        VALUES
        (:id, :request_id, :agent, :error_type, :message, :retriable, :attempt, :created_at)
        """
        
        params = {
            "id": f"err_{uuid4().hex[:12]}",
            "request_id": request_id,
            "agent": error.agent,
            "error_type": error.error_type.value,
            "message": error.message,
            "retriable": 1 if error.retriable else 0,
            "attempt": error.attempt,
            "created_at": error.timestamp.isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
    
    # Messages
    
    async def save_message(self, request_id: str, message: Message) -> None:
        """Save a conversation message."""
        sql = """
        INSERT INTO messages
        (id, request_id, role, content, metadata_json, created_at)
        VALUES
        (:id, :request_id, :role, :content, :metadata_json, :created_at)
        """
        
        params = {
            "id": f"msg_{uuid4().hex[:12]}",
            "request_id": request_id,
            "role": message.role,
            "content": message.content,
            "metadata_json": json.dumps(message.metadata) if message.metadata else None,
            "created_at": message.timestamp.isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
    
    # Token Usage
    
    async def log_token_usage(self, request_id: str, usage: TokenUsage) -> None:
        """Log token usage."""
        sql = """
        INSERT INTO token_usage
        (id, request_id, agent, prompt_tokens, completion_tokens, 
         tool_calls, wall_time_ms, created_at)
        VALUES
        (:id, :request_id, :agent, :prompt_tokens, :completion_tokens,
         :tool_calls, :wall_time_ms, :created_at)
        """
        
        params = {
            "id": f"tok_{uuid4().hex[:12]}",
            "request_id": request_id,
            "agent": usage.agent,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tool_calls": usage.tool_calls,
            "wall_time_ms": usage.wall_time_ms,
            "created_at": datetime.utcnow().isoformat()
        }
        
        await self.session.execute(text(sql), params)
        await self.session.commit()
    
    async def get_token_usage_summary(self, request_id: str) -> Dict[str, Any]:
        """Get token usage summary for a request."""
        sql = """
        SELECT 
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(tool_calls) as total_tool_calls,
            SUM(wall_time_ms) as total_wall_time_ms,
            COUNT(*) as total_calls
        FROM token_usage
        WHERE request_id = :request_id
        """
        
        result = await self.session.execute(text(sql), {"request_id": request_id})
        row = result.first()
        
        if row:
            return dict(row._mapping)
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tool_calls": 0,
            "total_wall_time_ms": 0,
            "total_calls": 0
        }


# Helper function to get repository instance
async def get_repository(session: AsyncSession) -> Repository:
    """Get repository instance with session."""
    return Repository(session)