"""Pytest configuration and fixtures."""

import asyncio
import os
import pytest
import pytest_asyncio
from datetime import datetime
from typing import AsyncGenerator

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.app import app
from src.config import settings
from src.services.database import DatabaseManager
from src.graph.state import TroubleshootingRequest, TimeRange


# Set test environment
os.environ["API_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    # Create test database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    
    # Initialize schema
    db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db_manager.initialize()
    
    # Create session
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_request() -> TroubleshootingRequest:
    """Create a sample troubleshooting request."""
    return TroubleshootingRequest(
        title="High latency in payment service",
        description="Users reporting slow payment processing",
        service="payment-service",
        environment="prod",
        severity="high",
        time_range=TimeRange(
            from_time=datetime(2025, 1, 1, 10, 0),
            to_time=datetime(2025, 1, 1, 10, 30)
        )
    )


@pytest.fixture
def mock_mcp_response():
    """Mock MCP response for testing."""
    return {
        "success": True,
        "result": {
            "logs": [
                {
                    "timestamp": "2025-01-01T10:15:00Z",
                    "level": "ERROR",
                    "message": "Connection timeout to database",
                    "service": "payment-service"
                }
            ]
        }
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "plan": {
            "goals": ["Identify root cause of payment latency"],
            "tasks": [
                {
                    "task_id": "task_001",
                    "type": "log",
                    "inputs": {
                        "service": "payment-service",
                        "time_range": {
                            "from": "2025-01-01T10:00:00Z",
                            "to": "2025-01-01T10:30:00Z"
                        }
                    },
                    "hypotheses": ["Database connection issues"],
                    "priority": 1,
                    "timeout_s": 30
                }
            ]
        },
        "next_actions": ["query_logs"]
    }