"""Base agent class for all nodes."""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_llm_client, settings
from src.graph.state import (
    GraphState, AgentError, ErrorType, TokenUsage,
    add_error, update_state_timestamp, add_execution_path
)
from src.tools import get_agent_tools


class BaseAgent(ABC):
    """Base class for all agent nodes."""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.llm: BaseChatModel = get_llm_client()
        self.tools: List[StructuredTool] = []
        self.config = settings.get_agent_config(agent_type)
        self.max_retries = self.config.max_retries
        self.timeout_seconds = self.config.timeout_seconds
    
    async def initialize(self):
        """Initialize agent with tools."""
        self.tools = get_agent_tools(self.agent_type)
    
    @abstractmethod
    def get_system_prompt(self, state: GraphState) -> str:
        """Get system prompt for the agent."""
        pass
    
    @abstractmethod
    def get_user_prompt(self, state: GraphState) -> str:
        """Get user prompt based on current state."""
        pass
    
    @abstractmethod
    def process_response(self, response: Dict[str, Any], state: GraphState) -> GraphState:
        """Process LLM response and update state."""
        pass
    
    async def __call__(self, state: GraphState) -> GraphState:
        """Execute the agent node."""
        # Add to execution path
        state = add_execution_path(state, self.name)
        
        start_time = time.time()
        attempt = 0
        
        try:
            # Get prompts
            system_prompt = self.get_system_prompt(state)
            user_prompt = self.get_user_prompt(state)
            
            # Call LLM with retry
            response = await self._call_llm_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                state=state
            )
            
            # Process response
            state = self.process_response(response, state)
            
            # Log token usage
            wall_time_ms = int((time.time() - start_time) * 1000)
            self._log_token_usage(state, wall_time_ms)
            
            return update_state_timestamp(state)
            
        except Exception as e:
            # Log error
            error = AgentError(
                agent=self.name,
                error_type=self._classify_error(e),
                message=str(e),
                retriable=attempt < self.max_retries,
                attempt=attempt + 1
            )
            state = add_error(state, error)
            
            # Re-raise if not retriable
            if not error.retriable:
                raise
            
            return update_state_timestamp(state)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        state: GraphState
    ) -> Dict[str, Any]:
        """Call LLM with retry logic."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Bind tools if available
        if self.tools:
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self.llm.ainvoke(messages)
        
        # Extract JSON from response
        content = response.content
        
        # Try to parse as JSON
        try:
            # Handle case where LLM returns markdown code block
            if "```json" in content:
                json_start = content.index("```json") + 7
                json_end = content.index("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                # Generic code block
                json_start = content.index("```") + 3
                json_end = content.index("```", json_start)
                content = content[json_start:json_end].strip()
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            # Log the parsing error but try to extract useful information
            print(f"Warning: Failed to parse JSON from {self.name}: {e}")
            print(f"Raw content: {content}")
            
            # Return a basic error response
            return {
                "error": {
                    "type": "parse_error",
                    "message": f"Failed to parse JSON: {str(e)}",
                    "raw_content": content[:500]  # First 500 chars for debugging
                }
            }
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "rate" in error_str and "limit" in error_str:
            return ErrorType.RATE_LIMIT
        elif "permission" in error_str or "forbidden" in error_str:
            return ErrorType.PERMANENT
        elif "validation" in error_str or "invalid" in error_str:
            return ErrorType.VALIDATION
        else:
            return ErrorType.TRANSIENT
    
    def _log_token_usage(self, state: GraphState, wall_time_ms: int):
        """Log token usage to state."""
        # Note: Actual token counting would require response metadata
        # This is a placeholder for now
        usage = TokenUsage(
            agent=self.name,
            prompt_tokens=0,  # Would get from response metadata
            completion_tokens=0,  # Would get from response metadata
            tool_calls=len(self.tools),
            wall_time_ms=wall_time_ms
        )
        
        if "token_usage" not in state:
            state["token_usage"] = []
        state["token_usage"].append(usage.model_dump())
    
    def _extract_evidence_from_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract evidence from tool call results."""
        evidence = []
        
        for call in tool_calls:
            if "result" in call and call.get("success", False):
                # Structure evidence from tool result
                ev = {
                    "source": self.agent_type,
                    "summary": call.get("result", {}).get("summary", ""),
                    "raw_ref": {
                        "tool": call.get("tool_name"),
                        "query": call.get("arguments"),
                        "server": call.get("server_id")
                    },
                    "quality_score": 0.8  # Default score
                }
                evidence.append(ev)
        
        return evidence
    
    def _create_context_summary(self, state: GraphState) -> str:
        """Create a summary of current context for the prompt."""
        context_parts = []
        
        # Add request info
        request = state.get("request", {})
        context_parts.append(f"Issue: {request.get('title', 'Unknown')}")
        context_parts.append(f"Service: {request.get('service', 'Unknown')}")
        
        # Add current evidence count
        evidence_count = len(state.get("evidence", []))
        if evidence_count > 0:
            context_parts.append(f"Evidence collected: {evidence_count} items")
        
        # Add current findings
        findings = state.get("findings", [])
        if findings:
            context_parts.append(f"Findings: {len(findings)} hypotheses")
        
        # Add errors if any
        errors = state.get("errors", [])
        if errors:
            context_parts.append(f"Errors encountered: {len(errors)}")
        
        return "\n".join(context_parts)