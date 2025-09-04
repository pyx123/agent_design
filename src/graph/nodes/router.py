"""Router node for directing flow based on next_actions."""

from typing import List

from src.graph.state import GraphState, update_state_timestamp


def router_node(state: GraphState) -> GraphState:
    """
    Router node that doesn't modify state, just used for conditional edges.
    The actual routing logic is in the edge conditions.
    """
    # Simply pass through the state
    return update_state_timestamp(state)


def should_query_logs(state: GraphState) -> bool:
    """Check if we should query logs."""
    next_actions = state.get("next_actions", [])
    return "query_logs" in next_actions


def should_query_alarms(state: GraphState) -> bool:
    """Check if we should query alarms."""
    next_actions = state.get("next_actions", [])
    return "query_alarms" in next_actions


def should_query_kpis(state: GraphState) -> bool:
    """Check if we should query KPIs."""
    next_actions = state.get("next_actions", [])
    return "query_kpis" in next_actions


def should_summarize(state: GraphState) -> bool:
    """Check if we should summarize."""
    next_actions = state.get("next_actions", [])
    
    # Summarize if explicitly requested or if we have enough evidence
    if "summarize" in next_actions:
        return True
    
    # Also summarize if no more actions and we have evidence
    if not next_actions and state.get("evidence"):
        return True
    
    return False


def should_continue_investigating(state: GraphState) -> bool:
    """Check if we should continue investigating."""
    next_actions = state.get("next_actions", [])
    
    # Continue if we have investigation actions
    investigation_actions = {"query_logs", "query_alarms", "query_kpis"}
    return bool(investigation_actions.intersection(set(next_actions)))


def get_next_node(state: GraphState) -> str:
    """
    Determine the next node based on state.
    Used for conditional routing in the graph.
    """
    next_actions = state.get("next_actions", [])
    
    # Priority order for actions
    if "summarize" in next_actions:
        return "summary_agent"
    elif "query_logs" in next_actions:
        return "log_agent"
    elif "query_alarms" in next_actions:
        return "alarm_agent"
    elif "query_kpis" in next_actions:
        return "kpi_agent"
    else:
        # Default back to planner if no clear action
        return "planner"