#!/usr/bin/env python
"""Run tests with proper environment setup."""

import os
import sys
import subprocess

# Set test environment variables
os.environ["API_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["MCP_LOGS_SERVER_URL"] = ""
os.environ["MCP_METRICS_SERVER_URL"] = ""
os.environ["MCP_ALERTS_SERVER_URL"] = ""

# Add src to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run pytest
if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/unit", "-v"]
    
    print("Running tests with environment:")
    print(f"  API_ENV: {os.environ.get('API_ENV')}")
    print(f"  DATABASE_URL: {os.environ.get('DATABASE_URL')}")
    print(f"  Python Path: {sys.path[0]}")
    print("")
    
    result = subprocess.run(["pytest"] + args, cwd=os.path.dirname(__file__))
    sys.exit(result.returncode)