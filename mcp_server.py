import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    LogMessage,
    ShowMessageRequest,
    ShowMessageResult,
    ShowMessageParams,
    MessageType,
)
from database import get_db, SessionLocal
from models import (
    TaskExecution, Interaction, ToolCall, PerformanceMetric,
    TaskExecutionCreate, InteractionCreate, ToolCallCreate, PerformanceMetricCreate
)
from sqlalchemy.orm import Session
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClineRecorderServer:
    def __init__(self):
        self.server = Server("cline-recorder")
        self.setup_tools()
    
    def setup_tools(self):
        """Setup MCP tools for recording interactions and tool calls"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="start_task_execution",
                        description="Start recording a new task execution session",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Unique identifier for the task"},
                                "task_name": {"type": "string", "description": "Name or description of the task"},
                                "metadata": {"type": "object", "description": "Additional metadata for the task"}
                            },
                            "required": ["task_id"]
                        }
                    ),
                    Tool(
                        name="record_interaction",
                        description="Record an LLM interaction (prompt + response)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task ID to associate with"},
                                "interaction_order": {"type": "integer", "description": "Order of interaction within task"},
                                "prompt": {"type": "string", "description": "The prompt sent to the LLM"},
                                "response": {"type": "string", "description": "The response from the LLM"},
                                "model_name": {"type": "string", "description": "Name of the LLM model used"},
                                "tokens_used": {"type": "integer", "description": "Number of tokens used"},
                                "cost": {"type": "number", "description": "Cost of the interaction"},
                                "metadata": {"type": "object", "description": "Additional metadata"}
                            },
                            "required": ["task_id", "interaction_order", "prompt", "response"]
                        }
                    ),
                    Tool(
                        name="record_tool_call",
                        description="Record a tool call execution",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task ID to associate with"},
                                "tool_name": {"type": "string", "description": "Name of the tool called"},
                                "tool_parameters": {"type": "object", "description": "Parameters passed to the tool"},
                                "tool_result": {"type": "object", "description": "Result returned by the tool"},
                                "success": {"type": "boolean", "description": "Whether the tool call succeeded"},
                                "error_message": {"type": "string", "description": "Error message if failed"},
                                "execution_time": {"type": "number", "description": "Execution time in seconds"},
                                "metadata": {"type": "object", "description": "Additional metadata"}
                            },
                            "required": ["task_id", "tool_name"]
                        }
                    ),
                    Tool(
                        name="record_performance_metric",
                        description="Record a performance metric for task execution",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task ID to associate with"},
                                "metric_name": {"type": "string", "description": "Name of the metric"},
                                "metric_value": {"type": "number", "description": "Value of the metric"},
                                "metric_unit": {"type": "string", "description": "Unit of measurement"}
                            },
                            "required": ["task_id", "metric_name", "metric_value"]
                        }
                    ),
                    Tool(
                        name="complete_task_execution",
                        description="Mark a task execution as completed",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task ID to mark as completed"},
                                "status": {"type": "string", "description": "Final status (completed, failed)"},
                                "total_duration": {"type": "number", "description": "Total duration in seconds"},
                                "metadata": {"type": "object", "description": "Additional metadata"}
                            },
                            "required": ["task_id"]
                        }
                    ),
                    Tool(
                        name="get_task_summary",
                        description="Get a summary of a task execution",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task ID to get summary for"}
                            },
                            "required": ["task_id"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "start_task_execution":
                    return await self._start_task_execution(arguments)
                elif name == "record_interaction":
                    return await self._record_interaction(arguments)
                elif name == "record_tool_call":
                    return await self._record_tool_call(arguments)
                elif name == "record_performance_metric":
                    return await self._record_performance_metric(arguments)
                elif name == "complete_task_execution":
                    return await self._complete_task_execution(arguments)
                elif name == "get_task_summary":
                    return await self._get_task_summary(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                        isError=True
                    )
            except Exception as e:
                logger.error(f"Error in tool call {name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True
                )
    
    async def _start_task_execution(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Start recording a new task execution"""
        task_id = arguments["task_id"]
        task_name = arguments.get("task_name")
        metadata = arguments.get("metadata")
        
        db = SessionLocal()
        try:
            # Check if task already exists
            existing_task = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
            if existing_task:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Task {task_id} already exists")],
                    isError=True
                )
            
            # Create new task execution
            task_execution = TaskExecution(
                task_id=task_id,
                task_name=task_name,
                metadata=metadata
            )
            db.add(task_execution)
            db.commit()
            db.refresh(task_execution)
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Task execution started with ID: {task_execution.id}")]
            )
        finally:
            db.close()
    
    async def _record_interaction(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Record an LLM interaction"""
        task_id = arguments["task_id"]
        interaction_order = arguments["interaction_order"]
        prompt = arguments["prompt"]
        response = arguments["response"]
        model_name = arguments.get("model_name")
        tokens_used = arguments.get("tokens_used")
        cost = arguments.get("cost")
        metadata = arguments.get("metadata")
        
        db = SessionLocal()
        try:
            # Find the task execution
            task_execution = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
            if not task_execution:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Task {task_id} not found")],
                    isError=True
                )
            
            # Create interaction record
            interaction = Interaction(
                task_execution_id=task_execution.id,
                interaction_order=interaction_order,
                prompt=prompt,
                response=response,
                model_name=model_name,
                tokens_used=tokens_used,
                cost=cost,
                metadata=metadata
            )
            db.add(interaction)
            db.commit()
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Interaction recorded with ID: {interaction.id}")]
            )
        finally:
            db.close()
    
    async def _record_tool_call(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Record a tool call execution"""
        task_id = arguments["task_id"]
        tool_name = arguments["tool_name"]
        tool_parameters = arguments.get("tool_parameters")
        tool_result = arguments.get("tool_result")
        success = arguments.get("success", True)
        error_message = arguments.get("error_message")
        execution_time = arguments.get("execution_time")
        metadata = arguments.get("metadata")
        
        db = SessionLocal()
        try:
            # Find the task execution
            task_execution = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
            if not task_execution:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Task {task_id} not found")],
                    isError=True
                )
            
            # Create tool call record
            tool_call = ToolCall(
                task_execution_id=task_execution.id,
                tool_name=tool_name,
                tool_parameters=tool_parameters,
                tool_result=tool_result,
                success=success,
                error_message=error_message,
                execution_time=execution_time,
                metadata=metadata
            )
            db.add(tool_call)
            db.commit()
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Tool call recorded with ID: {tool_call.id}")]
            )
        finally:
            db.close()
    
    async def _record_performance_metric(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Record a performance metric"""
        task_id = arguments["task_id"]
        metric_name = arguments["metric_name"]
        metric_value = arguments["metric_value"]
        metric_unit = arguments.get("metric_unit")
        
        db = SessionLocal()
        try:
            # Find the task execution
            task_execution = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
            if not task_execution:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Task {task_id} not found")],
                    isError=True
                )
            
            # Create performance metric record
            metric = PerformanceMetric(
                task_execution_id=task_execution.id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit
            )
            db.add(metric)
            db.commit()
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Performance metric recorded with ID: {metric.id}")]
            )
        finally:
            db.close()
    
    async def _complete_task_execution(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Mark a task execution as completed"""
        task_id = arguments["task_id"]
        status = arguments.get("status", "completed")
        total_duration = arguments.get("total_duration")
        metadata = arguments.get("metadata")
        
        db = SessionLocal()
        try:
            # Find the task execution
            task_execution = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
            if not task_execution:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Task {task_id} not found")],
                    isError=True
                )
            
            # Update task execution
            task_execution.status = status
            task_execution.completed_at = datetime.utcnow()
            if total_duration is not None:
                task_execution.total_duration = total_duration
            if metadata is not None:
                task_execution.metadata = metadata
            
            db.commit()
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Task {task_id} marked as {status}")]
            )
        finally:
            db.close()
    
    async def _get_task_summary(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get a summary of a task execution"""
        task_id = arguments["task_id"]
        
        db = SessionLocal()
        try:
            # Find the task execution with related data
            task_execution = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).first()
            if not task_execution:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Task {task_id} not found")],
                    isError=True
                )
            
            # Count related records
            interaction_count = len(task_execution.interactions)
            tool_call_count = len(task_execution.tool_calls)
            performance_metric_count = len(task_execution.performance_metrics)
            
            summary = {
                "task_id": task_execution.task_id,
                "task_name": task_execution.task_name,
                "status": task_execution.status,
                "started_at": task_execution.started_at.isoformat() if task_execution.started_at else None,
                "completed_at": task_execution.completed_at.isoformat() if task_execution.completed_at else None,
                "total_duration": task_execution.total_duration,
                "interaction_count": interaction_count,
                "tool_call_count": tool_call_count,
                "performance_metric_count": performance_metric_count,
                "metadata": task_execution.metadata
            }
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(summary, indent=2))]
            )
        finally:
            db.close()

async def main():
    """Main entry point for the MCP server"""
    server = ClineRecorderServer()
    
    # Initialize database
    from database import init_db
    init_db()
    
    # Start the server
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cline-recorder",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())