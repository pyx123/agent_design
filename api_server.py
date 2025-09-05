from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from database import get_db, SessionLocal
from models import (
    TaskExecution, Interaction, ToolCall, PerformanceMetric,
    TaskExecutionCreate, InteractionCreate, ToolCallCreate, PerformanceMetricCreate
)

app = FastAPI(title="Cline Recorder API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Cline Recorder API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Task Execution endpoints
@app.post("/tasks/", response_model=Dict[str, Any])
async def create_task(task: TaskExecutionCreate, db: Session = Depends(get_db)):
    """Create a new task execution"""
    try:
        # Check if task already exists
        existing_task = db.query(TaskExecution).filter(TaskExecution.task_id == task.task_id).first()
        if existing_task:
            raise HTTPException(status_code=400, detail=f"Task {task.task_id} already exists")
        
        # Create new task execution
        db_task = TaskExecution(
            task_id=task.task_id,
            task_name=task.task_name,
            metadata=task.metadata
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        
        return {
            "id": db_task.id,
            "task_id": db_task.task_id,
            "task_name": db_task.task_name,
            "status": db_task.status,
            "started_at": db_task.started_at.isoformat() if db_task.started_at else None,
            "message": "Task execution created successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/", response_model=List[Dict[str, Any]])
async def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all task executions with optional filtering"""
    query = db.query(TaskExecution)
    
    if status:
        query = query.filter(TaskExecution.status == status)
    
    tasks = query.offset(skip).limit(limit).all()
    
    return [
        {
            "id": task.id,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "total_duration": task.total_duration,
            "metadata": task.metadata
        }
        for task in tasks
    ]

@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get a specific task execution by task_id"""
    task = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Get related data counts
    interaction_count = db.query(Interaction).filter(Interaction.task_execution_id == task.id).count()
    tool_call_count = db.query(ToolCall).filter(ToolCall.task_execution_id == task.id).count()
    performance_metric_count = db.query(PerformanceMetric).filter(PerformanceMetric.task_execution_id == task.id).count()
    
    return {
        "id": task.id,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "status": task.status,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "total_duration": task.total_duration,
        "metadata": task.metadata,
        "interaction_count": interaction_count,
        "tool_call_count": tool_call_count,
        "performance_metric_count": performance_metric_count
    }

@app.put("/tasks/{task_id}", response_model=Dict[str, Any])
async def update_task(task_id: str, task_update: TaskExecutionUpdate, db: Session = Depends(get_db)):
    """Update a task execution"""
    db_task = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    try:
        if task_update.status is not None:
            db_task.status = task_update.status
        if task_update.completed_at is not None:
            db_task.completed_at = task_update.completed_at
        if task_update.total_duration is not None:
            db_task.total_duration = task_update.total_duration
        if task_update.metadata is not None:
            db_task.metadata = task_update.metadata
        
        db.commit()
        db.refresh(db_task)
        
        return {
            "message": "Task updated successfully",
            "task_id": db_task.task_id,
            "status": db_task.status
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Interaction endpoints
@app.post("/interactions/", response_model=Dict[str, Any])
async def create_interaction(interaction: InteractionCreate, db: Session = Depends(get_db)):
    """Create a new interaction record"""
    try:
        # Verify task exists
        task = db.query(TaskExecution).filter(TaskExecution.id == interaction.task_execution_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task execution not found")
        
        # Create interaction
        db_interaction = Interaction(
            task_execution_id=interaction.task_execution_id,
            interaction_order=interaction.interaction_order,
            prompt=interaction.prompt,
            response=interaction.response,
            model_name=interaction.model_name,
            tokens_used=interaction.tokens_used,
            cost=interaction.cost,
            metadata=interaction.metadata
        )
        db.add(db_interaction)
        db.commit()
        db.refresh(db_interaction)
        
        return {
            "id": db_interaction.id,
            "message": "Interaction recorded successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}/interactions/", response_model=List[Dict[str, Any]])
async def get_task_interactions(
    task_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all interactions for a specific task"""
    task = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    interactions = db.query(Interaction).filter(
        Interaction.task_execution_id == task.id
    ).order_by(Interaction.interaction_order).offset(skip).limit(limit).all()
    
    return [
        {
            "id": interaction.id,
            "interaction_order": interaction.interaction_order,
            "prompt": interaction.prompt,
            "response": interaction.response,
            "model_name": interaction.model_name,
            "tokens_used": interaction.tokens_used,
            "cost": interaction.cost,
            "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None,
            "metadata": interaction.metadata
        }
        for interaction in interactions
    ]

# Tool Call endpoints
@app.post("/tool-calls/", response_model=Dict[str, Any])
async def create_tool_call(tool_call: ToolCallCreate, db: Session = Depends(get_db)):
    """Create a new tool call record"""
    try:
        # Verify task exists
        task = db.query(TaskExecution).filter(TaskExecution.id == tool_call.task_execution_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task execution not found")
        
        # Create tool call
        db_tool_call = ToolCall(
            task_execution_id=tool_call.task_execution_id,
            tool_name=tool_call.tool_name,
            tool_parameters=tool_call.tool_parameters,
            tool_result=tool_call.tool_result,
            success=tool_call.success,
            error_message=tool_call.error_message,
            execution_time=tool_call.execution_time,
            metadata=tool_call.metadata
        )
        db.add(db_tool_call)
        db.commit()
        db.refresh(db_tool_call)
        
        return {
            "id": db_tool_call.id,
            "message": "Tool call recorded successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}/tool-calls/", response_model=List[Dict[str, Any]])
async def get_task_tool_calls(
    task_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all tool calls for a specific task"""
    task = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    tool_calls = db.query(ToolCall).filter(
        ToolCall.task_execution_id == task.id
    ).order_by(ToolCall.timestamp).offset(skip).limit(limit).all()
    
    return [
        {
            "id": tool_call.id,
            "tool_name": tool_call.tool_name,
            "tool_parameters": tool_call.tool_parameters,
            "tool_result": tool_call.tool_result,
            "success": tool_call.success,
            "error_message": tool_call.error_message,
            "execution_time": tool_call.execution_time,
            "timestamp": tool_call.timestamp.isoformat() if tool_call.timestamp else None,
            "metadata": tool_call.metadata
        }
        for tool_call in tool_calls
    ]

# Performance Metric endpoints
@app.post("/performance-metrics/", response_model=Dict[str, Any])
async def create_performance_metric(metric: PerformanceMetricCreate, db: Session = Depends(get_db)):
    """Create a new performance metric record"""
    try:
        # Verify task exists
        task = db.query(TaskExecution).filter(TaskExecution.id == metric.task_execution_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task execution not found")
        
        # Create metric
        db_metric = PerformanceMetric(
            task_execution_id=metric.task_execution_id,
            metric_name=metric.metric_name,
            metric_value=metric.metric_value,
            metric_unit=metric.metric_unit
        )
        db.add(db_metric)
        db.commit()
        db.refresh(db_metric)
        
        return {
            "id": db_metric.id,
            "message": "Performance metric recorded successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}/performance-metrics/", response_model=List[Dict[str, Any]])
async def get_task_performance_metrics(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Get all performance metrics for a specific task"""
    task = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    metrics = db.query(PerformanceMetric).filter(
        PerformanceMetric.task_execution_id == task.id
    ).order_by(PerformanceMetric.timestamp).all()
    
    return [
        {
            "id": metric.id,
            "metric_name": metric.metric_name,
            "metric_value": metric.metric_value,
            "metric_unit": metric.metric_unit,
            "timestamp": metric.timestamp.isoformat() if metric.timestamp else None
        }
        for metric in metrics
    ]

# Analytics endpoints
@app.get("/analytics/summary")
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Get overall analytics summary"""
    total_tasks = db.query(TaskExecution).count()
    completed_tasks = db.query(TaskExecution).filter(TaskExecution.status == "completed").count()
    running_tasks = db.query(TaskExecution).filter(TaskExecution.status == "running").count()
    failed_tasks = db.query(TaskExecution).filter(TaskExecution.status == "failed").count()
    
    total_interactions = db.query(Interaction).count()
    total_tool_calls = db.query(ToolCall).count()
    
    # Calculate average task duration
    completed_tasks_with_duration = db.query(TaskExecution).filter(
        TaskExecution.status == "completed",
        TaskExecution.total_duration.isnot(None)
    ).all()
    
    avg_duration = 0
    if completed_tasks_with_duration:
        avg_duration = sum(task.total_duration for task in completed_tasks_with_duration) / len(completed_tasks_with_duration)
    
    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "running_tasks": running_tasks,
        "failed_tasks": failed_tasks,
        "total_interactions": total_interactions,
        "total_tool_calls": total_tool_calls,
        "average_task_duration_seconds": round(avg_duration, 2),
        "completion_rate": round(completed_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0
    }

@app.get("/analytics/tool-usage")
async def get_tool_usage_analytics(db: Session = Depends(get_db)):
    """Get tool usage analytics"""
    from sqlalchemy import func
    
    # Get tool usage counts
    tool_usage = db.query(
        ToolCall.tool_name,
        func.count(ToolCall.id).label('usage_count'),
        func.avg(ToolCall.execution_time).label('avg_execution_time'),
        func.sum(ToolCall.execution_time).label('total_execution_time')
    ).filter(
        ToolCall.success == True,
        ToolCall.execution_time.isnot(None)
    ).group_by(ToolCall.tool_name).all()
    
    return [
        {
            "tool_name": usage.tool_name,
            "usage_count": usage.usage_count,
            "avg_execution_time": round(usage.avg_execution_time, 3) if usage.avg_execution_time else None,
            "total_execution_time": round(usage.total_execution_time, 3) if usage.total_execution_time else None
        }
        for usage in tool_usage
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)