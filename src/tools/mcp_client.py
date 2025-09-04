"""MCP (Model Context Protocol) client implementation."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

from src.config import settings, MCPServerConfig


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]


class MCPRequest(BaseModel):
    """MCP request format."""
    tool: str
    arguments: Dict[str, Any]
    request_id: Optional[str] = Field(default_factory=lambda: f"mcp_{datetime.utcnow().timestamp()}")


class MCPResponse(BaseModel):
    """MCP response format."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, server_config: MCPServerConfig):
        self.server_id = server_config.server_id
        self.endpoint = server_config.endpoint
        self.token = server_config.token
        self.timeout_seconds = server_config.timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._tools_cache: Dict[str, MCPTool] = {}
    
    @property
    async def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "DevOps-Agent/1.0",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_seconds),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def discover_tools(self) -> List[MCPTool]:
        """Discover available tools from the MCP server."""
        client = await self.client
        
        try:
            response = await client.get("/tools")
            response.raise_for_status()
            
            tools_data = response.json()
            tools = []
            
            for tool_data in tools_data.get("tools", []):
                tool = MCPTool(**tool_data)
                tools.append(tool)
                self._tools_cache[tool.name] = tool
            
            return tools
            
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to discover tools from {self.server_id}: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> MCPResponse:
        """Call a tool on the MCP server."""
        client = await self.client
        
        # Validate tool exists
        if tool_name not in self._tools_cache:
            # Try to refresh tools cache
            await self.discover_tools()
            if tool_name not in self._tools_cache:
                return MCPResponse(
                    success=False,
                    error=f"Tool '{tool_name}' not found on server '{self.server_id}'"
                )
        
        request = MCPRequest(
            tool=tool_name,
            arguments=arguments,
            request_id=request_id
        )
        
        try:
            response = await client.post(
                "/tools/invoke",
                json=request.model_dump()
            )
            response.raise_for_status()
            
            result_data = response.json()
            return MCPResponse(**result_data)
            
        except httpx.HTTPError as e:
            return MCPResponse(
                success=False,
                error=f"HTTP error calling tool '{tool_name}': {str(e)}"
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                error=f"Unexpected error calling tool '{tool_name}': {str(e)}"
            )
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get tool definition from cache."""
        return self._tools_cache.get(tool_name)


class MCPClientManager:
    """Manages multiple MCP clients."""
    
    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize all MCP clients from configuration."""
        if self._initialized:
            return
        
        for server_id, server_config in settings.mcp_servers.items():
            client = MCPClient(server_config)
            self._clients[server_id] = client
            
            # Discover tools on initialization
            try:
                await client.discover_tools()
            except Exception as e:
                print(f"Warning: Failed to discover tools for {server_id}: {e}")
        
        self._initialized = True
    
    async def close_all(self):
        """Close all MCP clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._initialized = False
    
    def get_client(self, server_id: str) -> Optional[MCPClient]:
        """Get a specific MCP client."""
        return self._clients.get(server_id)
    
    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> MCPResponse:
        """Call a tool on a specific server."""
        client = self.get_client(server_id)
        if not client:
            return MCPResponse(
                success=False,
                error=f"MCP server '{server_id}' not configured"
            )
        
        return await client.call_tool(tool_name, arguments, request_id)
    
    def get_all_tools(self) -> Dict[str, List[MCPTool]]:
        """Get all tools from all servers."""
        tools_by_server = {}
        for server_id, client in self._clients.items():
            tools_by_server[server_id] = list(client._tools_cache.values())
        return tools_by_server
    
    def find_tool_server(self, tool_name: str) -> Optional[str]:
        """Find which server provides a specific tool."""
        # Check mapping configuration first
        for agent_type, mapping in settings.mcp_mapping.items():
            if tool_name in mapping.get("tools", []):
                return mapping.get("server")
        
        # Otherwise search all clients
        for server_id, client in self._clients.items():
            if tool_name in client._tools_cache:
                return server_id
        
        return None


# Global MCP client manager
mcp_manager = MCPClientManager()


# Helper functions
async def init_mcp_clients():
    """Initialize MCP clients."""
    await mcp_manager.initialize()


async def close_mcp_clients():
    """Close all MCP clients."""
    await mcp_manager.close_all()


def get_mcp_manager() -> MCPClientManager:
    """Get the global MCP manager instance."""
    return mcp_manager