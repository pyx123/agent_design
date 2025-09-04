"""Bridge between LangGraph tools and MCP."""

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import Tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from pydantic import ConfigDict

from src.tools.mcp_client import mcp_manager, MCPResponse


class MCPToolWrapper:
    """Wraps MCP tools as LangChain tools."""
    
    @staticmethod
    def create_tool_schema(tool_name: str, parameters: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic schema from MCP tool parameters."""
        fields = {}
        
        # Convert MCP parameter schema to Pydantic fields
        for param_name, param_info in parameters.get("properties", {}).items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            param_required = param_name in parameters.get("required", [])
            
            # Map JSON schema types to Python types
            type_mapping = {
                "string": str,
                "number": float,
                "integer": int,
                "boolean": bool,
                "array": List[Any],
                "object": Dict[str, Any],
            }
            
            python_type = type_mapping.get(param_type, Any)
            
            # Create field with proper default
            if param_required:
                fields[param_name] = (python_type, Field(description=param_desc))
            else:
                default_value = param_info.get("default", None)
                fields[param_name] = (python_type, Field(default=default_value, description=param_desc))
        
        # Create dynamic model
        model_name = f"{tool_name.replace('.', '_')}Schema"
        return create_model(model_name, **fields)
    
    @staticmethod
    def create_langchain_tool(
        server_id: str,
        tool_name: str,
        tool_description: str,
        tool_parameters: Dict[str, Any]
    ) -> StructuredTool:
        """Create a LangChain tool from MCP tool definition."""
        
        # Create schema
        schema = MCPToolWrapper.create_tool_schema(tool_name, tool_parameters)
        
        # Create tool function
        async def tool_func(**kwargs) -> str:
            """Execute MCP tool and return result."""
            response = await mcp_manager.call_tool(
                server_id=server_id,
                tool_name=tool_name,
                arguments=kwargs
            )
            
            if response.success:
                # Return result as JSON string for LLM to parse
                return json.dumps(response.result, indent=2)
            else:
                # Return error message
                return f"Error: {response.error}"
        
        # Create structured tool
        return StructuredTool(
            name=tool_name.replace(".", "_"),  # LangChain doesn't like dots in names
            description=tool_description,
            func=tool_func,
            coroutine=tool_func,
            args_schema=schema,
        )


class MCPToolRegistry:
    """Registry for MCP tools accessible to agents."""
    
    def __init__(self):
        self._tools: Dict[str, StructuredTool] = {}
        self._tool_mapping: Dict[str, str] = {}  # tool_name -> server_id
    
    async def initialize(self):
        """Initialize registry with all available MCP tools."""
        # Ensure MCP manager is initialized
        await mcp_manager.initialize()
        
        # Get all tools from all servers
        all_tools = mcp_manager.get_all_tools()
        
        for server_id, tools in all_tools.items():
            for tool in tools:
                # Create LangChain tool
                lc_tool = MCPToolWrapper.create_langchain_tool(
                    server_id=server_id,
                    tool_name=tool.name,
                    tool_description=tool.description,
                    tool_parameters=tool.parameters
                )
                
                # Register tool
                tool_key = tool.name.replace(".", "_")
                self._tools[tool_key] = lc_tool
                self._tool_mapping[tool_key] = server_id
    
    def get_tools_for_agent(self, agent_type: str) -> List[StructuredTool]:
        """Get tools available for a specific agent type."""
        # Get mapping from configuration
        mapping = settings.mcp_mapping.get(agent_type, {})
        allowed_tools = mapping.get("tools", [])
        
        # Filter tools based on mapping
        tools = []
        for tool_name in allowed_tools:
            tool_key = tool_name.replace(".", "_")
            if tool_key in self._tools:
                tools.append(self._tools[tool_key])
        
        return tools
    
    def get_all_tools(self) -> List[StructuredTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tool(self, tool_name: str) -> Optional[StructuredTool]:
        """Get a specific tool by name."""
        tool_key = tool_name.replace(".", "_")
        return self._tools.get(tool_key)


# Global tool registry
tool_registry = MCPToolRegistry()


# Special MCP tool for direct invocation (used by agents)
class MCPToolInvoker(BaseModel):
    """Schema for direct MCP tool invocation."""
    server_id: str = Field(description="MCP server ID (e.g., 'logs', 'metrics', 'alerts')")
    tool_name: str = Field(description="Tool name to invoke (e.g., 'log.search', 'metric.query')")
    arguments: Dict[str, Any] = Field(description="Tool arguments as JSON object")


async def mcp_tool_invoker(
    server_id: str,
    tool_name: str,
    arguments: Dict[str, Any]
) -> str:
    """
    Direct MCP tool invoker for agents.
    
    This is a fallback tool that allows agents to call any MCP tool directly.
    """
    response = await mcp_manager.call_tool(
        server_id=server_id,
        tool_name=tool_name,
        arguments=arguments
    )
    
    if response.success:
        return json.dumps({
            "success": True,
            "result": response.result,
            "metadata": response.metadata
        }, indent=2)
    else:
        return json.dumps({
            "success": False,
            "error": response.error
        }, indent=2)


# Create the universal MCP tool
mcp_universal_tool = StructuredTool(
    name="mcp_tool_invoker",
    description=(
        "Invoke any MCP tool by specifying server_id, tool_name, and arguments. "
        "Use this to call external tools for logs, metrics, alerts, etc."
    ),
    func=mcp_tool_invoker,
    coroutine=mcp_tool_invoker,
    args_schema=MCPToolInvoker,
)


# Helper functions
async def init_tool_registry():
    """Initialize the tool registry."""
    await tool_registry.initialize()


def get_agent_tools(agent_type: str) -> List[StructuredTool]:
    """Get tools for a specific agent type."""
    # Get specific tools for the agent
    tools = tool_registry.get_tools_for_agent(agent_type)
    
    # Always include the universal MCP tool as fallback
    tools.append(mcp_universal_tool)
    
    return tools