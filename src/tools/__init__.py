"""Tools module for MCP integration."""

from .mcp_client import (
    MCPClient,
    MCPClientManager,
    MCPRequest,
    MCPResponse,
    MCPTool,
    mcp_manager,
    init_mcp_clients,
    close_mcp_clients,
)
from .mcp_bridge import (
    MCPToolWrapper,
    MCPToolRegistry,
    tool_registry,
    mcp_universal_tool,
    init_tool_registry,
    get_agent_tools,
)

__all__ = [
    # MCP Client
    "MCPClient",
    "MCPClientManager", 
    "MCPRequest",
    "MCPResponse",
    "MCPTool",
    "mcp_manager",
    "init_mcp_clients",
    "close_mcp_clients",
    # MCP Bridge
    "MCPToolWrapper",
    "MCPToolRegistry",
    "tool_registry",
    "mcp_universal_tool",
    "init_tool_registry",
    "get_agent_tools",
]