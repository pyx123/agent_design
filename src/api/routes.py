"""API routes for the DevOps Agent."""

import asyncio
import json
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import (
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
)
from src.graph import (
    TroubleshootingRequest,
    ExecutionStatus,
    run_investigation,
    run_investigation_with_parallel_execution,
)
from src.services import get_db_session, Repository
from src.config import settings


# Create router
router = APIRouter(prefix="/api/v1", tags=["troubleshooting"])

# Background task queue
background_tasks = {}


async def run_investigation_background(request_id: str, request_dict: dict):
    """Run investigation in background."""
    try:
        # Run the investigation
        if settings.limits.max_concurrent_agents > 1:
            result = await run_investigation_with_parallel_execution(request_dict)
        else:
            result = await run_investigation(request_dict)
        
        # Store result
        background_tasks[request_id] = {
            "status": "completed",
            "result": result,
            "completed_at": datetime.utcnow()
        }
        
        # Update database
        async with get_db_session() as session:
            repo = Repository(session)
            await repo.update_request_status(request_id, ExecutionStatus.COMPLETED)
            
            # Save root cause and remediation if present
            if result.get("root_cause"):
                from src.graph.state import RootCause
                root_cause = RootCause(**result["root_cause"])
                await repo.save_root_cause(request_id, root_cause)
            
            if result.get("remediation"):
                from src.graph.state import Remediation
                remediation = Remediation(**result["remediation"])
                await repo.save_remediation(request_id, remediation)
                
    except Exception as e:
        background_tasks[request_id] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow()
        }
        
        # Update database
        async with get_db_session() as session:
            repo = Repository(session)
            await repo.update_request_status(request_id, ExecutionStatus.FAILED)


@router.post("/troubleshoot", response_model=TroubleshootingRequestResponse)
async def create_troubleshooting_request(
    request: TroubleshootingRequestCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
) -> TroubleshootingRequestResponse:
    """Create a new troubleshooting request."""
    
    # Create request object
    troubleshooting_request = TroubleshootingRequest(
        **request.model_dump(),
        tenant_id=None  # TODO: Extract from auth context
    )
    
    # Save to database
    repo = Repository(db)
    request_id = await repo.create_request(troubleshooting_request)
    
    # Handle sync vs async mode
    if request.mode == "sync":
        # Run synchronously (with timeout)
        try:
            result = await asyncio.wait_for(
                run_investigation(troubleshooting_request.model_dump()),
                timeout=300  # 5 minute timeout
            )
            
            # Update status
            await repo.update_request_status(request_id, ExecutionStatus(result["status"]))
            
            return TroubleshootingRequestResponse(
                id=request_id,
                status=ExecutionStatus(result["status"]),
                mode="sync",
                created_at=troubleshooting_request.created_at
            )
            
        except asyncio.TimeoutError:
            await repo.update_request_status(request_id, ExecutionStatus.FAILED)
            raise HTTPException(status_code=504, detail="Investigation timed out")
            
    else:
        # Run asynchronously in background
        background_tasks.add_task(
            run_investigation_background,
            request_id,
            troubleshooting_request.model_dump()
        )
        
        return TroubleshootingRequestResponse(
            id=request_id,
            status=ExecutionStatus.INIT,
            mode="async",
            created_at=troubleshooting_request.created_at
        )


@router.get("/troubleshoot", response_model=List[TroubleshootingSummary])
async def list_troubleshooting_requests(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    service: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
) -> List[TroubleshootingSummary]:
    """List troubleshooting requests with pagination."""
    
    repo = Repository(db)
    offset = (page - 1) * size
    
    requests = await repo.list_requests(
        tenant_id=None,  # TODO: Extract from auth
        status=status,
        service=service,
        limit=size,
        offset=offset
    )
    
    # Convert to summary format
    summaries = []
    for req in requests:
        # Get evidence count
        evidence = await repo.get_evidence(req["id"])
        
        summary = TroubleshootingSummary(
            id=req["id"],
            title=req["title"],
            service=req["service"],
            environment=req["environment"],
            severity=req["severity"],
            status=req["status"],
            created_at=datetime.fromisoformat(req["created_at"]),
            updated_at=datetime.fromisoformat(req["updated_at"]) if req.get("updated_at") else None,
            evidence_count=len(evidence),
            has_root_cause=False  # TODO: Check if root cause exists
        )
        summaries.append(summary)
    
    return summaries


@router.get("/troubleshoot/{request_id}", response_model=TroubleshootingResult)
async def get_troubleshooting_request(
    request_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> TroubleshootingResult:
    """Get troubleshooting request details."""
    
    repo = Repository(db)
    
    # Get request
    request = await repo.get_request(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Get related data
    plan = await repo.get_latest_plan(request_id)
    evidence = await repo.get_evidence(request_id)
    tasks = await repo.get_tasks(request_id)
    
    # Get token usage summary
    token_usage = await repo.get_token_usage_summary(request_id)
    
    # Check if running in background
    if request_id in background_tasks:
        task_info = background_tasks[request_id]
        if task_info["status"] == "completed":
            result_state = task_info["result"]
            # Update from state
            return TroubleshootingResult(
                id=request_id,
                status=ExecutionStatus(result_state.get("status", "running")),
                request=request,
                plan=result_state.get("plan"),
                evidence=result_state.get("evidence", []),
                findings=result_state.get("findings", []),
                root_cause=result_state.get("root_cause"),
                remediation=result_state.get("remediation"),
                errors=result_state.get("errors", []),
                token_usage=token_usage,
                created_at=datetime.fromisoformat(request["created_at"]),
                completed_at=task_info.get("completed_at")
            )
    
    # Return current state from database
    return TroubleshootingResult(
        id=request_id,
        status=ExecutionStatus(request["status"]),
        request=request,
        plan=plan.get("plan") if plan else None,
        evidence=evidence,
        findings=[],  # TODO: Load from DB
        root_cause=None,  # TODO: Load from DB
        remediation=None,  # TODO: Load from DB
        errors=[],  # TODO: Load from DB
        token_usage=token_usage,
        created_at=datetime.fromisoformat(request["created_at"]),
        completed_at=None
    )


@router.patch("/troubleshoot/{request_id}")
async def update_troubleshooting_request(
    request_id: str,
    update: UpdateTroubleshootingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
) -> dict:
    """Update a troubleshooting request (cancel, retry, continue)."""
    
    repo = Repository(db)
    
    # Get request
    request = await repo.get_request(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    current_status = ExecutionStatus(request["status"])
    
    if update.action == UpdateRequestAction.CANCEL:
        # Cancel the request
        if current_status in [ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed request")
        
        await repo.update_request_status(request_id, ExecutionStatus.CANCELLED)
        
        # TODO: Cancel background task if running
        
        return {"status": "cancelled"}
    
    elif update.action == UpdateRequestAction.RETRY:
        # Retry the request
        if current_status not in [ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Can only retry failed requests")
        
        # Reset status and run again
        await repo.update_request_status(request_id, ExecutionStatus.INIT)
        
        # Run in background
        background_tasks.add_task(
            run_investigation_background,
            request_id,
            request
        )
        
        return {"status": "retrying"}
    
    elif update.action == UpdateRequestAction.CONTINUE:
        # Continue with additional context
        if current_status != ExecutionStatus.INVESTIGATING:
            raise HTTPException(
                status_code=400,
                detail="Can only continue requests in investigating state"
            )
        
        # TODO: Add additional context to state and continue
        
        return {"status": "continuing"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


@router.get("/troubleshoot/{request_id}/evidence")
async def get_evidence(
    request_id: str,
    source: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
) -> List[dict]:
    """Get evidence collected for a request."""
    
    repo = Repository(db)
    
    # Check request exists
    request = await repo.get_request(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Get evidence
    evidence = await repo.get_evidence(request_id, source)
    
    return evidence


@router.get("/troubleshoot/{request_id}/report")
async def get_report(
    request_id: str,
    format: ReportFormat = Query(ReportFormat.MARKDOWN),
    db: AsyncSession = Depends(get_db_session)
) -> Response:
    """Get analysis report in specified format."""
    
    # Get full result
    result = await get_troubleshooting_request(request_id, db)
    
    # Generate report based on format
    if format == ReportFormat.MARKDOWN:
        # Check if report was generated
        if request_id in background_tasks:
            task_info = background_tasks[request_id]
            if task_info.get("status") == "completed":
                result_state = task_info["result"]
                report = result_state.get("metadata", {}).get("incident_report", "")
                if report:
                    return Response(
                        content=report,
                        media_type="text/markdown",
                        headers={"Content-Disposition": f"inline; filename=report_{request_id}.md"}
                    )
        
        # Generate basic report
        report = _generate_markdown_report(result)
        return Response(
            content=report,
            media_type="text/markdown",
            headers={"Content-Disposition": f"inline; filename=report_{request_id}.md"}
        )
    
    elif format == ReportFormat.HTML:
        # TODO: Implement HTML generation
        raise HTTPException(status_code=501, detail="HTML format not implemented")
    
    elif format == ReportFormat.PDF:
        # TODO: Implement PDF generation
        raise HTTPException(status_code=501, detail="PDF format not implemented")


def _generate_markdown_report(result: TroubleshootingResult) -> str:
    """Generate a basic markdown report."""
    lines = [
        f"# Troubleshooting Report",
        f"\n## Request Information",
        f"- **ID**: {result.id}",
        f"- **Title**: {result.request['title']}",
        f"- **Service**: {result.request['service']}",
        f"- **Status**: {result.status}",
        f"- **Created**: {result.created_at}",
    ]
    
    if result.root_cause:
        lines.extend([
            f"\n## Root Cause Analysis",
            f"**Hypothesis**: {result.root_cause['hypothesis']}",
            f"**Confidence**: {result.root_cause['confidence']:.2%}",
        ])
    
    if result.remediation:
        lines.extend([
            f"\n## Remediation Steps",
        ])
        for i, action in enumerate(result.remediation.get("actions", []), 1):
            lines.append(f"{i}. {action['description']}")
    
    if result.evidence:
        lines.extend([
            f"\n## Evidence Summary",
            f"Total evidence collected: {len(result.evidence)}",
        ])
    
    return "\n".join(lines)


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Health check endpoint."""
    
    services = {
        "api": "healthy",
        "database": "healthy",  # TODO: Check DB connection
        "mcp": "healthy",  # TODO: Check MCP servers
    }
    
    return HealthStatus(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        version=settings.environment,
        services=services,
        timestamp=datetime.utcnow()
    )