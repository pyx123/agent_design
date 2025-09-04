"""Test configuration to override production settings."""

import os

# Set test environment
os.environ["API_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LOG_LEVEL"] = "ERROR"

# Disable MCP clients for testing
os.environ["MCP_LOGS_SERVER_URL"] = ""
os.environ["MCP_METRICS_SERVER_URL"] = ""
os.environ["MCP_ALERTS_SERVER_URL"] = ""