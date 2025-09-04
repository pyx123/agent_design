"""Log analysis agent implementation."""

import json
from datetime import datetime
from typing import Any, Dict, List

from src.graph.state import (
    GraphState, Evidence, TaskType, TaskStatus,
    update_state_timestamp
)
from src.graph.nodes.base import BaseAgent


class LogAgent(BaseAgent):
    """Agent for analyzing logs from external systems."""
    
    def __init__(self):
        super().__init__(name="log_agent", agent_type="log")
    
    def get_system_prompt(self, state: GraphState) -> str:
        """Get log agent system prompt."""
        # Load from file
        prompt_path = "docs/prompts/log.md"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except:
            # Fallback prompt
            return """
You are the Log Agent. Your responsibility is to:
- Query log systems for relevant information
- Analyze log patterns and anomalies
- Extract evidence from logs
- Identify error patterns, timeouts, and system issues

You have access to MCP tools for querying logs. Use the mcp_tool_invoker to call:
- log.search: Search logs with filters
- log.tail: Get recent logs
- log.summary: Get log summaries

Output only valid JSON matching the specified schema.
"""
    
    def get_user_prompt(self, state: GraphState) -> str:
        """Build prompt for log analysis."""
        request = state.get("request", {})
        
        # Find relevant log tasks
        tasks = state.get("tasks", [])
        log_tasks = [t for t in tasks if t.get("type") == "log" and t.get("status") != "completed"]
        
        if not log_tasks:
            # Create a default task based on request
            service = request.get("service", "unknown")
            time_range = request.get("time_range", {})
            
            context = f"""
Analyze logs for the following issue:
- Service: {service}
- Environment: {request.get('environment', 'prod')}
- Issue: {request.get('title', 'Unknown issue')}
- Time Range: {time_range.get('from', 'recent')} to {time_range.get('to', 'now')}

Focus on:
1. Error messages (5xx, timeouts, connection errors)
2. Warning messages indicating degradation
3. Restart or crash events
4. Performance anomalies

Use the available MCP tools to query the logs system.
"""
        else:
            # Use the first pending task
            task = log_tasks[0]
            inputs = task.get("inputs", {})
            
            context = f"""
Execute log analysis task:
- Task ID: {task.get('task_id')}
- Service: {inputs.get('service', 'unknown')}
- Environment: {inputs.get('environment', 'prod')}
- Time Range: {inputs.get('time_range', {}).get('from', 'recent')} to {inputs.get('time_range', {}).get('to', 'now')}
- Filters: {json.dumps(inputs.get('filters', {}))}
- Search hints: {', '.join(inputs.get('hints', []))}
- Hypotheses: {', '.join(task.get('hypotheses', []))}

Use the mcp_tool_invoker to query logs and extract relevant evidence.
"""
        
        return context
    
    def process_response(self, response: Dict[str, Any], state: GraphState) -> GraphState:
        """Process log agent response."""
        # Check for errors
        if "error" in response:
            print(f"Log agent error: {response['error']}")
            return state
        
        # Extract evidence from response
        evidence_list = response.get("evidence", [])
        findings_list = response.get("findings", [])
        
        # Add evidence to state
        if "evidence" not in state:
            state["evidence"] = []
        
        for ev_data in evidence_list:
            evidence = Evidence(
                source=TaskType.LOG,
                summary=ev_data.get("summary", ""),
                raw_ref=ev_data.get("raw_ref"),
                quality_score=ev_data.get("quality_score", 0.7),
                task_id=self._get_current_task_id(state)
            )
            state["evidence"].append(evidence.model_dump())
        
        # Add findings to state
        if findings_list and "findings" not in state:
            state["findings"] = []
        
        for finding in findings_list:
            state["findings"].append(finding)
        
        # Update task status
        task_id = self._get_current_task_id(state)
        if task_id:
            self._update_task_status(state, task_id, TaskStatus.COMPLETED)
        
        # Clear next_actions to return to planner
        state["next_actions"] = []
        
        return update_state_timestamp(state)
    
    def _get_current_task_id(self, state: GraphState) -> str:
        """Get current log task ID."""
        tasks = state.get("tasks", [])
        log_tasks = [t for t in tasks if t.get("type") == "log" and t.get("status") != "completed"]
        
        if log_tasks:
            return log_tasks[0].get("task_id", "")
        return ""
    
    def _update_task_status(self, state: GraphState, task_id: str, status: TaskStatus):
        """Update task status in state."""
        tasks = state.get("tasks", [])
        for task in tasks:
            if task.get("task_id") == task_id:
                task["status"] = status.value
                task["updated_at"] = datetime.utcnow().isoformat()
                if status == TaskStatus.COMPLETED:
                    task["completed_at"] = datetime.utcnow().isoformat()
                break


def log_agent_node(state: GraphState) -> GraphState:
    """Log agent node function for LangGraph."""
    agent = LogAgent()
    # Initialize is handled in __call__ if needed
    return agent(state)