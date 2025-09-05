from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

Base = declarative_base()

class TaskExecution(Base):
    """Represents a complete task execution session"""
    __tablename__ = "task_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, index=True, nullable=False)
    task_name = Column(String(255), nullable=True)
    status = Column(String(50), default="running")  # running, completed, failed
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_duration = Column(Float, nullable=True)  # in seconds
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="task_execution")
    tool_calls = relationship("ToolCall", back_populates="task_execution")
    performance_metrics = relationship("PerformanceMetric", back_populates="task_execution")

class Interaction(Base):
    """Represents an LLM interaction (prompt + response)"""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    task_execution_id = Column(Integer, ForeignKey("task_executions.id"), nullable=False)
    interaction_order = Column(Integer, nullable=False)  # Order within the task
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_name = Column(String(255), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    task_execution = relationship("TaskExecution", back_populates="interactions")

class ToolCall(Base):
    """Represents a tool call execution"""
    __tablename__ = "tool_calls"
    
    id = Column(Integer, primary_key=True, index=True)
    task_execution_id = Column(Integer, ForeignKey("task_executions.id"), nullable=False)
    tool_name = Column(String(255), nullable=False)
    tool_parameters = Column(JSON, nullable=True)
    tool_result = Column(JSON, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)  # in seconds
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    task_execution = relationship("TaskExecution", back_populates="tool_calls")

class PerformanceMetric(Base):
    """Represents performance metrics for task execution"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    task_execution_id = Column(Integer, ForeignKey("task_executions.id"), nullable=False)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    task_execution = relationship("TaskExecution", back_populates="performance_metrics")

# Pydantic models for API requests/responses
class TaskExecutionCreate(BaseModel):
    task_id: str
    task_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskExecutionUpdate(BaseModel):
    status: Optional[str] = None
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class InteractionCreate(BaseModel):
    task_execution_id: int
    interaction_order: int
    prompt: str
    response: str
    model_name: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ToolCallCreate(BaseModel):
    task_execution_id: int
    tool_name: str
    tool_parameters: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class PerformanceMetricCreate(BaseModel):
    task_execution_id: int
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None