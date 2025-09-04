"""Alarm/Alert analysis agent implementation."""

import json
from datetime import datetime
from typing import Any, Dict, List

from src.graph.state import (
    GraphState, Evidence, TaskType, TaskStatus,
    update_state_timestamp
)
from src.graph.nodes.base import BaseAgent


class AlarmAgent(BaseAgent):
    """Agent for analyzing alarms/alerts from monitoring systems."""
    
    def __init__(self):
        super().__init__(name="alarm_agent", agent_type="alarm")
    
    def get_system_prompt(self, state: GraphState) -> str:
        """Get alarm agent system prompt."""
        # Load from file
        prompt_path = "docs/prompts/alarm.md"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except:
            # Fallback prompt
            return """
You are the Alarm Agent. Your responsibility is to:
- Query alerting systems for active and recent alerts
- Analyze alert patterns and correlations
- Identify critical alerts related to the issue
- Group and deduplicate alerts

You have access to MCP tools for querying alerts. Use the mcp_tool_invoker to call:
- alert.list: List alerts with filters
- alert.get: Get alert details
- alert.summary: Get alert summaries

Output only valid JSON matching the specified schema.
"""
    
    def get_user_prompt(self, state: GraphState) -> str:
        """Build prompt for alarm analysis."""
        request = state.get("request", {})
        
        # Find relevant alarm tasks
        tasks = state.get("tasks", [])
        alarm_tasks = [t for t in tasks if t.get("type") == "alarm" and t.get("status") != "completed"]
        
        if not alarm_tasks:
            # Create a default context
            service = request.get("service", "unknown")
            time_range = request.get("time_range", {})
            
            context = f"""
Analyze alerts/alarms for the following issue:
- Service: {service}
- Environment: {request.get('environment', 'prod')}
- Issue: {request.get('title', 'Unknown issue')}
- Time Range: {time_range.get('from', 'recent')} to {time_range.get('to', 'now')}
- Severity: {request.get('severity', 'medium')}

Focus on:
1. Critical and high severity alerts
2. Alerts related to the service or its dependencies
3. Alert patterns and correlations
4. Alert storm detection

Use the available MCP tools to query the alerting system.
"""
        else:
            # Use the first pending task
            task = alarm_tasks[0]
            inputs = task.get("inputs", {})
            
            context = f"""
Execute alarm analysis task:
- Task ID: {task.get('task_id')}
- Service: {inputs.get('service', 'unknown')}
- Environment: {inputs.get('environment', 'prod')}
- Time Range: {inputs.get('time_range', {}).get('from', 'recent')} to {inputs.get('time_range', {}).get('to', 'now')}
- Severity Filter: {inputs.get('filters', {}).get('severity', 'all')}
- Search hints: {', '.join(inputs.get('hints', []))}
- Hypotheses: {', '.join(task.get('hypotheses', []))}

Use the mcp_tool_invoker to query alerts and extract relevant evidence.
Group related alerts and identify patterns.
"""
        
        return context
    
    def process_response(self, response: Dict[str, Any], state: GraphState) -> GraphState:
        """Process alarm agent response."""
        # Check for errors
        if "error" in response:
            print(f"Alarm agent error: {response['error']}")
            return state
        
        # Extract evidence from response
        evidence_list = response.get("evidence", [])
        findings_list = response.get("findings", [])
        
        # Add evidence to state
        if "evidence" not in state:
            state["evidence"] = []
        
        for ev_data in evidence_list:
            evidence = Evidence(
                source=TaskType.ALARM,
                summary=ev_data.get("summary", ""),
                raw_ref=ev_data.get("raw_ref"),
                quality_score=ev_data.get("quality_score", 0.8),
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
        """Get current alarm task ID."""
        tasks = state.get("tasks", [])
        alarm_tasks = [t for t in tasks if t.get("type") == "alarm" and t.get("status") != "completed"]
        
        if alarm_tasks:
            return alarm_tasks[0].get("task_id", "")
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


async def alarm_agent_node(state: GraphState) -> GraphState:
    """Alarm agent node function for LangGraph."""
    agent = AlarmAgent()
    await agent.initialize()  # Initialize tools
    return await agent(state)