"""Integration tests for API endpoints."""

import pytest
import asyncio
from datetime import datetime

from fastapi import status

from src.graph.state import ExecutionStatus


@pytest.mark.asyncio
class TestTroubleshootingAPI:
    """Test troubleshooting API endpoints."""
    
    async def test_create_request(self, async_client):
        """Test creating a troubleshooting request."""
        request_data = {
            "title": "API test issue",
            "service": "test-service",
            "description": "Integration test",
            "severity": "medium",
            "mode": "async"
        }
        
        response = await async_client.post(
            "/api/v1/troubleshoot",
            json=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"].startswith("req_")
        assert data["status"] == "init"
        assert data["mode"] == "async"
    
    async def test_list_requests(self, async_client):
        """Test listing troubleshooting requests."""
        # Create a request first
        await async_client.post(
            "/api/v1/troubleshoot",
            json={
                "title": "List test",
                "service": "test-service"
            }
        )
        
        # List requests
        response = await async_client.get("/api/v1/troubleshoot")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "title" in data[0]
        assert "service" in data[0]
    
    async def test_get_request_details(self, async_client):
        """Test getting request details."""
        # Create a request
        create_response = await async_client.post(
            "/api/v1/troubleshoot",
            json={
                "title": "Detail test",
                "service": "test-service"
            }
        )
        request_id = create_response.json()["id"]
        
        # Get details
        response = await async_client.get(f"/api/v1/troubleshoot/{request_id}")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == request_id
        assert data["request"]["title"] == "Detail test"
        assert "evidence" in data
        assert "plan" in data
    
    async def test_get_nonexistent_request(self, async_client):
        """Test getting a non-existent request."""
        response = await async_client.get("/api/v1/troubleshoot/invalid_id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    async def test_update_request_cancel(self, async_client):
        """Test cancelling a request."""
        # Create a request
        create_response = await async_client.post(
            "/api/v1/troubleshoot",
            json={
                "title": "Cancel test",
                "service": "test-service"
            }
        )
        request_id = create_response.json()["id"]
        
        # Cancel it
        response = await async_client.patch(
            f"/api/v1/troubleshoot/{request_id}",
            json={"action": "cancel"}
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "cancelled"
    
    async def test_get_evidence(self, async_client):
        """Test getting evidence for a request."""
        # Create a request
        create_response = await async_client.post(
            "/api/v1/troubleshoot",
            json={
                "title": "Evidence test",
                "service": "test-service"
            }
        )
        request_id = create_response.json()["id"]
        
        # Get evidence (should be empty initially)
        response = await async_client.get(
            f"/api/v1/troubleshoot/{request_id}/evidence"
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []
    
    async def test_get_report(self, async_client):
        """Test getting a report."""
        # Create a request
        create_response = await async_client.post(
            "/api/v1/troubleshoot",
            json={
                "title": "Report test",
                "service": "test-service"
            }
        )
        request_id = create_response.json()["id"]
        
        # Get markdown report
        response = await async_client.get(
            f"/api/v1/troubleshoot/{request_id}/report",
            params={"format": "markdown"}
        )
        assert response.status_code == status.HTTP_200_OK
        assert "# Troubleshooting Report" in response.text
    
    async def test_health_check(self, async_client):
        """Test health check endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "services" in data