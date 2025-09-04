"""Summary agent implementation for root cause analysis and remediation."""

import json
from typing import Any, Dict, List

from src.graph.state import (
    GraphState, RootCause, Remediation, RemediationAction,
    ExecutionStatus, update_state_timestamp
)
from src.graph.nodes.base import BaseAgent


class SummaryAgent(BaseAgent):
    """Agent for summarizing findings and providing root cause analysis."""
    
    def __init__(self):
        super().__init__(name="summary_agent", agent_type="summary")
    
    def get_system_prompt(self, state: GraphState) -> str:
        """Get summary agent system prompt."""
        # Load from file
        prompt_path = "docs/prompts/summary.md"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except:
            # Fallback prompt
            return """
You are the Summary Agent. Your responsibility is to:
- Analyze all collected evidence and findings
- Determine the most likely root cause
- Provide actionable remediation steps
- Generate a comprehensive incident report

Based on the evidence from logs, alarms, and KPIs, synthesize:
1. Root cause hypothesis with confidence score
2. Affected components and services
3. Timeline correlation
4. Remediation actions with risk assessment
5. Validation and rollback procedures

Output only valid JSON matching the specified schema.
"""
    
    def get_user_prompt(self, state: GraphState) -> str:
        """Build prompt for summary generation."""
        request = state.get("request", {})
        plan = state.get("plan", {})
        evidence = state.get("evidence", [])
        findings = state.get("findings", [])
        
        # Build comprehensive context
        context_parts = [
            "## Troubleshooting Summary Request",
            f"\n### Original Issue:",
            f"- Title: {request.get('title', 'Unknown')}",
            f"- Service: {request.get('service', 'Unknown')}",
            f"- Environment: {request.get('environment', 'prod')}",
            f"- Severity: {request.get('severity', 'medium')}",
            f"- Description: {request.get('description', 'No description')}",
        ]
        
        # Add investigation goals
        if plan:
            context_parts.append(f"\n### Investigation Goals:")
            for goal in plan.get("goals", []):
                context_parts.append(f"- {goal}")
        
        # Add evidence summary
        context_parts.append(f"\n### Evidence Collected ({len(evidence)} items):")
        
        # Group evidence by source
        evidence_by_source = {}
        for ev in evidence:
            source = ev.get("source", "unknown")
            if source not in evidence_by_source:
                evidence_by_source[source] = []
            evidence_by_source[source].append(ev)
        
        for source, items in evidence_by_source.items():
            context_parts.append(f"\n#### From {source} ({len(items)} items):")
            for ev in items[:5]:  # Limit to 5 per source
                context_parts.append(f"- {ev.get('summary', 'No summary')}")
                if ev.get('quality_score', 0) > 0.8:
                    context_parts.append(f"  (High confidence: {ev.get('quality_score')})")
        
        # Add findings
        if findings:
            context_parts.append(f"\n### Key Findings ({len(findings)} items):")
            for finding in findings[:5]:
                context_parts.append(f"- {finding.get('hypothesis_ref', 'Unknown')}")
                context_parts.append(f"  Confidence: {finding.get('confidence', 0)}")
                if finding.get('impact_scope'):
                    context_parts.append(f"  Impact: {', '.join(finding['impact_scope'])}")
        
        context_parts.append("\n### Task:")
        context_parts.append("Based on all the evidence above, provide:")
        context_parts.append("1. Root cause analysis with confidence score")
        context_parts.append("2. Affected components list")
        context_parts.append("3. Remediation steps with risk assessment")
        context_parts.append("4. Validation procedures")
        context_parts.append("5. Markdown-formatted incident report")
        
        return "\n".join(context_parts)
    
    def process_response(self, response: Dict[str, Any], state: GraphState) -> GraphState:
        """Process summary response and update state."""
        # Check for errors
        if "error" in response:
            print(f"Summary agent error: {response['error']}")
            state["status"] = ExecutionStatus.FAILED.value
            return state
        
        # Extract root cause
        root_cause_data = response.get("root_cause")
        if root_cause_data:
            root_cause = RootCause(
                hypothesis=root_cause_data.get("hypothesis", "Unknown"),
                confidence=root_cause_data.get("confidence", 0.5),
                affected_components=root_cause_data.get("affected_components", []),
                time_correlation=root_cause_data.get("time_correlation"),
                change_correlation=root_cause_data.get("change_correlation")
            )
            state["root_cause"] = root_cause.model_dump()
        
        # Extract remediation
        remediation_data = response.get("remediation")
        if remediation_data:
            # Parse actions
            actions = []
            for action_data in remediation_data.get("actions", []):
                action = RemediationAction(
                    description=action_data.get("description", ""),
                    command=action_data.get("command"),
                    risk_level=action_data.get("risk_level", "medium"),
                    estimated_duration=action_data.get("estimated_duration")
                )
                actions.append(action)
            
            remediation = Remediation(
                actions=actions,
                risk=remediation_data.get("risk", "medium"),
                required_approvals=remediation_data.get("required_approvals", []),
                validation_steps=remediation_data.get("validation_steps", []),
                rollback_steps=remediation_data.get("rollback_steps", [])
            )
            state["remediation"] = remediation.model_dump()
        
        # Store the markdown report if provided
        if "report_md" in response:
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["incident_report"] = response["report_md"]
        
        # Update confidence scores
        if "confidence_scores" in response:
            state["confidence_scores"] = response["confidence_scores"]
        
        # Mark as completed
        state["status"] = ExecutionStatus.COMPLETED.value
        state["done"] = True
        state["next_actions"] = []
        
        return update_state_timestamp(state)


async def summary_agent_node(state: GraphState) -> GraphState:
    """Summary agent node function for LangGraph."""
    agent = SummaryAgent()
    return await agent(state)