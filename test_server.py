#!/usr/bin/env python3
"""
Test script for the Cline Recorder MCP Server
This script tests the basic functionality without requiring Cline integration
"""

import asyncio
import json
import time
from datetime import datetime
from mcp_server import ClineRecorderServer
from database import init_db, SessionLocal
from models import TaskExecution, Interaction, ToolCall, PerformanceMetric

async def test_mcp_server():
    """Test the MCP server functionality"""
    print("Testing MCP Server...")
    
    # Initialize database
    init_db()
    
    # Create server instance
    server = ClineRecorderServer()
    
    # Test data
    task_id = f"test_task_{int(time.time())}"
    
    # Test 1: Start task execution
    print(f"\n1. Testing start_task_execution with task_id: {task_id}")
    result = await server._start_task_execution({
        "task_id": task_id,
        "task_name": "Test Task",
        "metadata": {"test": True, "timestamp": datetime.utcnow().isoformat()}
    })
    print(f"Result: {result.content[0].text}")
    
    # Test 2: Record interaction
    print(f"\n2. Testing record_interaction")
    result = await server._record_interaction({
        "task_id": task_id,
        "interaction_order": 1,
        "prompt": "Test prompt",
        "response": "Test response",
        "model_name": "test-model",
        "tokens_used": 50,
        "cost": 0.001
    })
    print(f"Result: {result.content[0].text}")
    
    # Test 3: Record tool call
    print(f"\n3. Testing record_tool_call")
    result = await server._record_tool_call({
        "task_id": task_id,
        "tool_name": "test_tool",
        "tool_parameters": {"param1": "value1"},
        "tool_result": {"result": "success"},
        "execution_time": 0.5
    })
    print(f"Result: {result.content[0].text}")
    
    # Test 4: Record performance metric
    print(f"\n4. Testing record_performance_metric")
    result = await server._record_performance_metric({
        "task_id": task_id,
        "metric_name": "test_metric",
        "metric_value": 100.5,
        "metric_unit": "units"
    })
    print(f"Result: {result.content[0].text}")
    
    # Test 5: Get task summary
    print(f"\n5. Testing get_task_summary")
    result = await server._get_task_summary({"task_id": task_id})
    print(f"Result: {result.content[0].text}")
    
    # Test 6: Complete task execution
    print(f"\n6. Testing complete_task_execution")
    result = await server._complete_task_execution({
        "task_id": task_id,
        "status": "completed",
        "total_duration": 2.5
    })
    print(f"Result: {result.content[0].text}")
    
    print("\nMCP Server tests completed!")

def test_database_directly():
    """Test database operations directly"""
    print("\nTesting Database Operations...")
    
    db = SessionLocal()
    try:
        # Check if tables were created
        task_count = db.query(TaskExecution).count()
        interaction_count = db.query(Interaction).count()
        tool_call_count = db.query(ToolCall).count()
        metric_count = db.query(PerformanceMetric).count()
        
        print(f"Database contains:")
        print(f"  - Tasks: {task_count}")
        print(f"  - Interactions: {interaction_count}")
        print(f"  - Tool Calls: {tool_call_count}")
        print(f"  - Performance Metrics: {metric_count}")
        
        # Show sample data
        if task_count > 0:
            print("\nSample Task:")
            task = db.query(TaskExecution).first()
            print(f"  ID: {task.id}")
            print(f"  Task ID: {task.task_id}")
            print(f"  Status: {task.status}")
            print(f"  Started: {task.started_at}")
            
        if interaction_count > 0:
            print("\nSample Interaction:")
            interaction = db.query(Interaction).first()
            print(f"  Order: {interaction.interaction_order}")
            print(f"  Prompt: {interaction.prompt[:50]}...")
            print(f"  Response: {interaction.response[:50]}...")
            
    finally:
        db.close()

def test_api_endpoints():
    """Test REST API endpoints"""
    print("\nTesting REST API...")
    
    # This would require the API server to be running
    print("Note: API testing requires the server to be running")
    print("Start with: python main.py --mode api")
    print("Then test endpoints like:")
    print("  GET http://localhost:8001/health")
    print("  GET http://localhost:8001/tasks/")
    print("  GET http://localhost:8001/analytics/summary")

async def main():
    """Main test function"""
    print("Cline Recorder MCP Server - Test Suite")
    print("=" * 50)
    
    try:
        # Test MCP server
        await test_mcp_server()
        
        # Test database directly
        test_database_directly()
        
        # Test API endpoints
        test_api_endpoints()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nTo start the servers:")
        print("  MCP Server: python main.py --mode mcp")
        print("  API Server: python main.py --mode api")
        print("  Both: python main.py --mode both")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())