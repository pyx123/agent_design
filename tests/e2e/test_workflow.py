"""End-to-end tests for complete workflows."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from src.graph import (
    run_investigation,
    TroubleshootingRequest,
    ExecutionStatus
)


@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """Test complete investigation workflows."""
    
    @patch('src.graph.nodes.base.get_llm_client')
    @patch('src.tools.mcp_bridge.mcp_manager')
    async def test_simple_investigation_flow(self, mock_mcp, mock_llm):
        """Test a simple investigation from start to finish."""
        
        # Mock LLM responses
        mock_llm_instance = AsyncMock()
        mock_llm.return_value = mock_llm_instance
        
        # Mock planner response
        mock_llm_instance.ainvoke.side_effect = [
            AsyncMock(content='''
            {
                "plan": {
                    "goals": ["Find root cause of latency"],
                    "tasks": [{
                        "task_id": "t1",
                        "type": "log",
                        "inputs": {"service": "test"},
                        "hypotheses": ["Connection issues"],
                        "priority": 1,
                        "timeout_s": 30
                    }]
                },
                "next_actions": ["query_logs"]
            }
            '''),
            # Log agent response
            AsyncMock(content='''
            {
                "evidence": [{
                    "summary": "Database connection timeouts",
                    "source": "log",
                    "quality_score": 0.9
                }],
                "findings": []
            }
            '''),
            # Planner response after evidence
            AsyncMock(content='''
            {
                "plan": {"goals": ["Summarize findings"], "tasks": []},
                "next_actions": ["summarize"]
            }
            '''),
            # Summary agent response
            AsyncMock(content='''
            {
                "root_cause": {
                    "hypothesis": "Database connection pool exhausted",
                    "confidence": 0.85,
                    "affected_components": ["api", "database"]
                },
                "remediation": {
                    "actions": [{
                        "description": "Increase connection pool size",
                        "risk_level": "low"
                    }],
                    "risk": "low",
                    "validation_steps": ["Monitor connection metrics"]
                },
                "report_md": "# Investigation Summary\\n\\nRoot cause identified."
            }
            ''')
        ]
        
        # Mock MCP tool calls
        mock_mcp.call_tool = AsyncMock(return_value=AsyncMock(
            success=True,
            result={"logs": ["error messages"]},
            error=None
        ))
        
        # Create request
        request = TroubleshootingRequest(
            title="Test investigation",
            service="test-service",
            description="E2E test"
        )
        
        # Run investigation
        result = await run_investigation(request.model_dump())
        
        # Verify results
        assert result["status"] == ExecutionStatus.COMPLETED.value
        assert result["done"] is True
        assert result["root_cause"] is not None
        assert result["root_cause"]["confidence"] == 0.85
        assert result["remediation"] is not None
        assert len(result["evidence"]) > 0
    
    @patch('src.graph.nodes.base.get_llm_client')
    async def test_investigation_with_error_handling(self, mock_llm):
        """Test investigation with error scenarios."""
        
        # Mock LLM to simulate an error
        mock_llm_instance = AsyncMock()
        mock_llm.return_value = mock_llm_instance
        
        # First call succeeds (planner)
        mock_llm_instance.ainvoke.side_effect = [
            AsyncMock(content='''
            {
                "plan": {
                    "goals": ["Investigate"],
                    "tasks": [{
                        "task_id": "t1",
                        "type": "log",
                        "inputs": {"service": "test"},
                        "priority": 1
                    }]
                },
                "next_actions": ["query_logs"]
            }
            '''),
            # Log agent fails
            Exception("Network error"),
            # Planner handles error
            AsyncMock(content='''
            {
                "plan": {"goals": ["Complete with limited data"], "tasks": []},
                "next_actions": ["summarize"]
            }
            '''),
            # Summary completes
            AsyncMock(content='''
            {
                "root_cause": {
                    "hypothesis": "Unable to fully investigate due to errors",
                    "confidence": 0.3,
                    "affected_components": ["unknown"]
                },
                "remediation": {
                    "actions": [{
                        "description": "Retry investigation when systems are available",
                        "risk_level": "low"
                    }],
                    "risk": "low"
                }
            }
            ''')
        ]
        
        # Create request
        request = TroubleshootingRequest(
            title="Error test",
            service="test-service"
        )
        
        # Run investigation
        result = await run_investigation(request.model_dump())
        
        # Should complete despite errors
        assert result["status"] == ExecutionStatus.COMPLETED.value
        assert len(result.get("errors", [])) > 0
        assert result["root_cause"]["confidence"] < 0.5  # Low confidence due to errors