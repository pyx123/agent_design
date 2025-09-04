"""Planner agent node implementation."""

import json
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from src.graph.state import (
    GraphState, InvestigationPlan, InvestigationTask, TaskType,
    ExecutionStatus, update_state_timestamp
)
from src.graph.nodes.base import BaseAgent


class PlannerAgent(BaseAgent):
    """Planner agent that creates and updates investigation plans."""
    
    def __init__(self):
        super().__init__(name="planner", agent_type="planner")
    
    def get_system_prompt(self, state: GraphState) -> str:
        """Get planner system prompt."""
        # Load from file
        prompt_path = "docs/prompts/planner.md"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except:
            # Fallback prompt
            return """
You are the Planner Agent. Your responsibility is to:
- Decompose the user's troubleshooting request into an executable investigation plan
- Orchestrate the work of other agents (Log/Alarm/KPI/Summary)
- Update the plan based on new evidence and adjust priorities

Output only valid JSON matching the specified schema.
"""
    
    def get_user_prompt(self, state: GraphState) -> str:
        """Build user prompt with current state context."""
        request = state.get("request", {})
        current_plan = state.get("plan")
        evidence = state.get("evidence", [])
        findings = state.get("findings", [])
        errors = state.get("errors", [])
        
        # Build context
        context_parts = [
            f"Request ID: {request.get('id', 'unknown')}",
            f"Title: {request.get('title', 'No title')}",
            f"Description: {request.get('description', 'No description')}",
            f"Service: {request.get('service', 'unknown')}",
            f"Environment: {request.get('environment', 'prod')}",
            f"Severity: {request.get('severity', 'medium')}",
        ]
        
        # Add time range if available
        if request.get("time_range"):
            time_range = request["time_range"]
            context_parts.append(f"Time Range: {time_range.get('from')} to {time_range.get('to')}")
        
        # Add current state info
        context_parts.append(f"\nCurrent State:")
        context_parts.append(f"- Evidence collected: {len(evidence)}")
        context_parts.append(f"- Findings: {len(findings)}")
        context_parts.append(f"- Errors: {len(errors)}")
        
        # Add previous plan summary if exists
        if current_plan:
            context_parts.append(f"\nPrevious Plan:")
            context_parts.append(f"- Goals: {', '.join(current_plan.get('goals', []))}")
            context_parts.append(f"- Tasks completed: {len([t for t in state.get('tasks', []) if t.get('status') == 'completed'])}")
        
        # Add recent evidence summary
        if evidence:
            context_parts.append(f"\nRecent Evidence:")
            for ev in evidence[-3:]:  # Last 3 pieces of evidence
                context_parts.append(f"- [{ev.get('source')}] {ev.get('summary', '')[:100]}...")
        
        # Add any recent errors
        if errors:
            context_parts.append(f"\nRecent Errors:")
            for err in errors[-2:]:  # Last 2 errors
                context_parts.append(f"- [{err.get('agent')}] {err.get('message', '')[:100]}...")
        
        context_parts.append("\nBased on the above context, generate or update the investigation plan.")
        
        return "\n".join(context_parts)
    
    def process_response(self, response: Dict[str, Any], state: GraphState) -> GraphState:
        """Process planner response and update state."""
        # Check for errors in response
        if "error" in response:
            print(f"Planner error: {response['error']}")
            state["status"] = ExecutionStatus.FAILED.value
            return state
        
        # Extract plan and next actions
        plan_data = response.get("plan", {})
        next_actions = response.get("next_actions", [])
        
        # Create InvestigationPlan object
        if plan_data:
            # Parse tasks
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task = InvestigationTask(
                    task_id=task_data.get("task_id", f"task_{uuid4().hex[:12]}"),
                    type=TaskType(task_data.get("type", "log")),
                    inputs=task_data.get("inputs", {}),
                    hypotheses=task_data.get("hypotheses", []),
                    priority=task_data.get("priority", 1),
                    timeout_s=task_data.get("timeout_s", 30)
                )
                tasks.append(task)
            
            # Create plan
            plan = InvestigationPlan(
                goals=plan_data.get("goals", ["Investigate the issue"]),
                tasks=tasks,
                version=state.get("plan", {}).get("version", 0) + 1 if state.get("plan") else 1
            )
            
            # Update state
            state["plan"] = plan.model_dump()
            
            # Add new tasks to state task list
            if "tasks" not in state:
                state["tasks"] = []
            
            for task in tasks:
                # Check if task already exists
                existing_ids = {t["task_id"] for t in state["tasks"]}
                if task.task_id not in existing_ids:
                    state["tasks"].append(task.model_dump())
        
        # Update next actions
        state["next_actions"] = next_actions
        
        # Update status based on next actions
        if "summarize" in next_actions:
            state["status"] = ExecutionStatus.SUMMARIZING.value
        elif any(action in next_actions for action in ["query_logs", "query_alarms", "query_kpis"]):
            state["status"] = ExecutionStatus.INVESTIGATING.value
        else:
            state["status"] = ExecutionStatus.PLANNING.value
        
        # Check termination conditions
        if not next_actions or (len(next_actions) == 1 and "summarize" in next_actions):
            # Ready to summarize or no more actions
            if state.get("evidence") and len(state["evidence"]) > 0:
                state["status"] = ExecutionStatus.SUMMARIZING.value
            else:
                # No evidence collected yet, need to investigate
                state["status"] = ExecutionStatus.INVESTIGATING.value
                state["next_actions"] = ["query_logs", "query_kpis"]
        
        return update_state_timestamp(state)
    
    def _extract_time_range(self, request: Dict[str, Any]) -> Dict[str, str]:
        """Extract time range from request or use default."""
        if request.get("time_range"):
            return request["time_range"]
        
        # Default to last 30 minutes
        now = datetime.utcnow()
        from_time = now.replace(minute=now.minute - 30)
        
        return {
            "from": from_time.isoformat() + "Z",
            "to": now.isoformat() + "Z"
        }


def planner_node(state: GraphState) -> GraphState:
    """Planner node function for LangGraph."""
    agent = PlannerAgent()
    return agent(state)