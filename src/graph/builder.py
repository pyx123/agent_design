"""LangGraph workflow builder."""

import asyncio
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
try:
    # 新版本的导入路径
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # 旧版本或简化版本的导入路径
    try:
        from langgraph.checkpoint import SqliteSaver, MemorySaver
    except ImportError:
        # 如果都失败，使用简单的字典实现
        class MemorySaver:
            """简单的内存检查点保存器"""
            def __init__(self):
                self.checkpoints = {}
            
            def get(self, config):
                thread_id = config.get("configurable", {}).get("thread_id")
                return self.checkpoints.get(thread_id)
            
            def put(self, config, checkpoint, metadata=None):
                thread_id = config.get("configurable", {}).get("thread_id")
                self.checkpoints[thread_id] = checkpoint
                return config
            
            def get_tuple(self, config):
                checkpoint = self.get(config)
                if checkpoint:
                    return {"checkpoint": checkpoint, "metadata": {}}
                return None
            
            def list(self, config):
                return []
            
            @classmethod
            def from_conn_string(cls, conn_string):
                """兼容性方法"""
                return cls()
        
        # SQLite保存器降级为内存保存器
        SqliteSaver = MemorySaver

from src.config import settings
from src.graph.state import GraphState, ExecutionStatus, create_initial_state
from src.graph.nodes import (
    planner_node,
    log_agent_node,
    alarm_agent_node,
    kpi_agent_node,
    summary_agent_node,
    router_node,
    get_next_node,
)


class GraphBuilder:
    """Builds and configures the LangGraph workflow."""
    
    def __init__(self, memory_type: str = "sqlite", checkpoint_db_path: str = None):
        self.memory_type = memory_type
        self.checkpoint_db_path = checkpoint_db_path or ".langgraph.db"
        self.graph = None
        self.compiled_graph = None
    
    def build(self):
        """Build the investigation graph."""
        # Create graph
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("planner", planner_node)
        graph.add_node("router", router_node)
        graph.add_node("log_agent", log_agent_node)
        graph.add_node("alarm_agent", alarm_agent_node)
        graph.add_node("kpi_agent", kpi_agent_node)
        graph.add_node("summary_agent", summary_agent_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Add edges
        # From planner to router
        graph.add_edge("planner", "router")
        
        # From router to agents or summary based on next_actions
        graph.add_conditional_edges(
            "router",
            get_next_node,
            {
                "log_agent": "log_agent",
                "alarm_agent": "alarm_agent",
                "kpi_agent": "kpi_agent",
                "summary_agent": "summary_agent",
                "planner": "planner",  # Loop back if needed
            }
        )
        
        # From agents back to planner (for re-planning)
        graph.add_edge("log_agent", "planner")
        graph.add_edge("alarm_agent", "planner")
        graph.add_edge("kpi_agent", "planner")
        
        # From summary to END
        graph.add_edge("summary_agent", END)
        
        self.graph = graph
        return self
    
    def compile(self):
        """Compile the graph with checkpointing."""
        if not self.graph:
            raise RuntimeError("Graph not built. Call build() first.")
        
        # Create checkpointer based on type
        if self.memory_type == "sqlite":
            checkpointer = SqliteSaver.from_conn_string(
                f"sqlite:///{self.checkpoint_db_path}"
            )
        else:
            checkpointer = MemorySaver()
        
        # Compile graph
        self.compiled_graph = self.graph.compile(
            checkpointer=checkpointer,
            debug=settings.is_development
        )
        
        return self.compiled_graph
    
    def get_graph(self):
        """Get the compiled graph, building if necessary."""
        if not self.compiled_graph:
            self.build()
            self.compile()
        return self.compiled_graph


class ParallelExecutor:
    """Executes multiple agent tasks in parallel."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or settings.limits.max_concurrent_agents
        self.semaphore = asyncio.Semaphore(self.max_workers)
    
    async def execute_parallel_agents(
        self,
        state: GraphState,
        agents_to_run: List[str]
    ) -> Dict[str, GraphState]:
        """Execute multiple agents in parallel and collect results."""
        tasks = []
        
        async def run_agent_with_limit(agent_name: str) -> tuple[str, GraphState]:
            async with self.semaphore:
                # Import and run the appropriate agent
                if agent_name == "log_agent":
                    result = await log_agent_node(state.copy())
                elif agent_name == "alarm_agent":
                    result = await alarm_agent_node(state.copy())
                elif agent_name == "kpi_agent":
                    result = await kpi_agent_node(state.copy())
                else:
                    raise ValueError(f"Unknown agent: {agent_name}")
                
                return agent_name, result
        
        # Create tasks for all agents
        for agent in agents_to_run:
            task = asyncio.create_task(run_agent_with_limit(agent))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_results = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Agent execution failed: {result}")
                continue
            
            agent_name, agent_state = result
            agent_results[agent_name] = agent_state
        
        return agent_results
    
    def merge_agent_results(
        self,
        base_state: GraphState,
        agent_results: Dict[str, GraphState]
    ) -> GraphState:
        """Merge results from multiple agents into base state."""
        merged_state = base_state.copy()
        
        # Merge evidence
        if "evidence" not in merged_state:
            merged_state["evidence"] = []
        
        for agent_name, agent_state in agent_results.items():
            # Add evidence from each agent
            for evidence in agent_state.get("evidence", []):
                merged_state["evidence"].append(evidence)
            
            # Merge findings
            if "findings" not in merged_state:
                merged_state["findings"] = []
            
            for finding in agent_state.get("findings", []):
                merged_state["findings"].append(finding)
            
            # Update task statuses
            for task in agent_state.get("tasks", []):
                # Find corresponding task in merged state
                for merged_task in merged_state.get("tasks", []):
                    if merged_task["task_id"] == task["task_id"]:
                        merged_task["status"] = task["status"]
                        merged_task["updated_at"] = task.get("updated_at")
                        merged_task["completed_at"] = task.get("completed_at")
                        break
            
            # Merge errors
            if "errors" not in merged_state:
                merged_state["errors"] = []
            
            for error in agent_state.get("errors", []):
                merged_state["errors"].append(error)
        
        return merged_state


# Factory function
def build_graph() -> StateGraph:
    """Build and return the compiled investigation graph."""
    builder = GraphBuilder(
        memory_type=settings.langgraph_memory_type,
    )
    return builder.get_graph()


# Async workflow runner
async def run_investigation(request_dict: dict) -> dict:
    """
    Run a complete investigation workflow.
    
    Args:
        request_dict: Dictionary containing the troubleshooting request
        
    Returns:
        Final state dictionary with results
    """
    # Get compiled graph
    graph = build_graph()
    
    # Create initial state
    from src.graph.state import TroubleshootingRequest
    request = TroubleshootingRequest(**request_dict)
    initial_state = create_initial_state(request)
    
    # Run the graph
    config = {
        "configurable": {
            "thread_id": request.id,
            "checkpoint_ns": "investigation"
        }
    }
    
    # Execute the graph
    final_state = await graph.ainvoke(initial_state, config)
    
    return final_state


# Parallel execution example
async def run_investigation_with_parallel_execution(request_dict: dict) -> dict:
    """
    Run investigation with parallel agent execution where possible.
    
    This is an example of how to implement parallel execution
    outside of the main graph for more control.
    """
    from src.graph.state import TroubleshootingRequest
    
    # Create request and initial state
    request = TroubleshootingRequest(**request_dict)
    state = create_initial_state(request)
    
    # Run planner first
    state = await planner_node(state)
    
    # Check if we have multiple investigation actions
    next_actions = state.get("next_actions", [])
    investigation_actions = [
        action for action in next_actions 
        if action in ["query_logs", "query_alarms", "query_kpis"]
    ]
    
    if len(investigation_actions) > 1:
        # Execute agents in parallel
        executor = ParallelExecutor()
        
        # Map actions to agent names
        agent_map = {
            "query_logs": "log_agent",
            "query_alarms": "alarm_agent",
            "query_kpis": "kpi_agent"
        }
        
        agents_to_run = [agent_map[action] for action in investigation_actions]
        
        # Run agents in parallel
        agent_results = await executor.execute_parallel_agents(state, agents_to_run)
        
        # Merge results
        state = executor.merge_agent_results(state, agent_results)
        
        # Clear next_actions and run planner again
        state["next_actions"] = []
        state = await planner_node(state)
    
    # Check if we should summarize
    if "summarize" in state.get("next_actions", []):
        state = await summary_agent_node(state)
    
    return state