"""KPI/Metrics analysis agent implementation."""

import json
from datetime import datetime
from typing import Any, Dict, List

from src.graph.state import (
    GraphState, Evidence, TaskType, TaskStatus,
    update_state_timestamp
)
from src.graph.nodes.base import BaseAgent


class KPIAgent(BaseAgent):
    """Agent for analyzing KPIs and metrics."""
    
    def __init__(self):
        super().__init__(name="kpi_agent", agent_type="kpi")
    
    def get_system_prompt(self, state: GraphState) -> str:
        """Get KPI agent system prompt."""
        # Load from file
        prompt_path = "docs/prompts/kpi.md"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except:
            # Fallback prompt
            return """
You are the KPI Agent. Your responsibility is to:
- Query metrics systems for relevant KPIs
- Analyze metric trends and anomalies
- Identify performance degradation patterns
- Correlate metrics with the reported issue

You have access to MCP tools for querying metrics. Use the mcp_tool_invoker to call:
- metric.query: Query metrics with PromQL or similar
- metric.range_query: Get metric time series data
- metric.baseline: Compare with historical baselines

Output only valid JSON matching the specified schema.
"""
    
    def get_user_prompt(self, state: GraphState) -> str:
        """Build prompt for KPI analysis."""
        request = state.get("request", {})
        
        # Find relevant KPI tasks
        tasks = state.get("tasks", [])
        kpi_tasks = [t for t in tasks if t.get("type") == "kpi" and t.get("status") != "completed"]
        
        if not kpi_tasks:
            # Create a default context
            service = request.get("service", "unknown")
            time_range = request.get("time_range", {})
            
            context = f"""
Analyze KPIs/metrics for the following issue:
- Service: {service}
- Environment: {request.get('environment', 'prod')}
- Issue: {request.get('title', 'Unknown issue')}
- Time Range: {time_range.get('from', 'recent')} to {time_range.get('to', 'now')}

Key metrics to analyze:
1. Latency metrics (p50, p90, p99)
2. Error rates and status codes
3. Request rates (QPS)
4. Resource utilization (CPU, memory, connections)
5. Custom business metrics

Use the available MCP tools to query the metrics system.
Look for anomalies, spikes, or degradation patterns.
"""
        else:
            # Use the first pending task
            task = kpi_tasks[0]
            inputs = task.get("inputs", {})
            
            context = f"""
Execute KPI analysis task:
- Task ID: {task.get('task_id')}
- Service: {inputs.get('service', 'unknown')}
- Environment: {inputs.get('environment', 'prod')}
- Time Range: {inputs.get('time_range', {}).get('from', 'recent')} to {inputs.get('time_range', {}).get('to', 'now')}
- Metric hints: {', '.join(inputs.get('hints', []))}
- Hypotheses: {', '.join(task.get('hypotheses', []))}

Use the mcp_tool_invoker to query metrics and identify:
1. Anomalies or significant changes
2. Correlation with the reported issue timeline
3. Comparison with normal baselines
4. Related metric degradation
"""
        
        return context
    
    def process_response(self, response: Dict[str, Any], state: GraphState) -> GraphState:
        """Process KPI agent response."""
        # Check for errors
        if "error" in response:
            print(f"KPI agent error: {response['error']}")
            return state
        
        # Extract evidence from response
        evidence_list = response.get("evidence", [])
        findings_list = response.get("findings", [])
        
        # Add evidence to state
        if "evidence" not in state:
            state["evidence"] = []
        
        for ev_data in evidence_list:
            evidence = Evidence(
                source=TaskType.KPI,
                summary=ev_data.get("summary", ""),
                raw_ref=ev_data.get("raw_ref"),
                quality_score=ev_data.get("quality_score", 0.9),  # Metrics typically have high quality
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
        """Get current KPI task ID."""
        tasks = state.get("tasks", [])
        kpi_tasks = [t for t in tasks if t.get("type") == "kpi" and t.get("status") != "completed"]
        
        if kpi_tasks:
            return kpi_tasks[0].get("task_id", "")
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


async def kpi_agent_node(state: GraphState) -> GraphState:
    """KPI agent node function for LangGraph."""
    agent = KPIAgent()
    await agent.initialize()  # Initialize tools
    return await agent(state)