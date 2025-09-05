# Cline Integration Guide

This guide explains how to integrate the Cline Recorder MCP Server with Cline to automatically record interaction records and tool call records during task execution.

## Overview

The MCP server provides tools that Cline can call at each step of task execution to record:
- Task execution sessions
- LLM interactions (prompts and responses)
- Tool calls and their results
- Performance metrics
- Task completion status

## Integration Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Database

The server uses SQLite in-memory mode by default, which requires no setup. The database is created automatically when the server starts.

For persistent storage, you can use a file-based SQLite database:

```bash
# Set environment variable for file-based SQLite
export DATABASE_URL="sqlite:///cline_records.db"
```

Or use Docker Compose:

```bash
docker-compose up -d
```

### 3. Initialize Database Tables

```bash
python main.py --init-db
```

Note: With SQLite in-memory mode, tables are created automatically when the server starts, but you can still run this command to verify the setup.

### 4. Start the MCP Server

```bash
python main.py --mode mcp
```

### 5. Configure Cline

Add the MCP server configuration to your Cline settings:

```json
{
  "mcpServers": {
    "cline-recorder": {
      "command": "python",
      "args": ["/path/to/your/workspace/main.py", "--mode", "mcp"],
      "env": {
        "DATABASE_URL": "sqlite:///:memory:"
      }
    }
  }
}
```

## Usage Patterns

### Starting a Task

At the beginning of each Cline task, call:

```python
# Start recording a new task
start_task_execution(
    task_id="unique_task_identifier",
    task_name="Description of the task",
    metadata={"user": "username", "project": "project_name"}
)
```

### Recording LLM Interactions

After each LLM interaction:

```python
# Record the interaction
record_interaction(
    task_id="unique_task_identifier",
    interaction_order=1,  # Sequential order within the task
    prompt="The prompt sent to the LLM",
    response="The response from the LLM",
    model_name="gpt-4",
    tokens_used=150,
    cost=0.003,
    metadata={"temperature": 0.7}
)
```

### Recording Tool Calls

After each tool execution:

```python
# Record successful tool call
record_tool_call(
    task_id="unique_task_identifier",
    tool_name="file_search",
    tool_parameters={"query": "config.json"},
    tool_result={"files": ["src/config.json"]},
    execution_time=0.5,
    metadata={"tool_version": "1.0.0"}
)

# Record failed tool call
record_tool_call(
    task_id="unique_task_identifier",
    tool_name="read_file",
    tool_parameters={"target_file": "nonexistent.txt"},
    success=False,
    error_message="File not found",
    execution_time=0.1
)
```

### Recording Performance Metrics

Track performance throughout the task:

```python
# Record timing metrics
record_performance_metric(
    task_id="unique_task_identifier",
    metric_name="total_tokens",
    metric_value=1250,
    metric_unit="tokens"
)

record_performance_metric(
    task_id="unique_task_identifier",
    metric_name="total_cost",
    metric_value=0.025,
    metric_unit="USD"
)
```

### Completing a Task

At the end of the task:

```python
# Mark task as completed
complete_task_execution(
    task_id="unique_task_identifier",
    status="completed",
    total_duration=45.2,  # seconds
    metadata={"final_result": "success", "files_created": 3}
)
```

## Example Cline Task Integration

Here's how a typical Cline task might look with recording:

```python
import time

# Start recording
start_time = time.time()
task_id = f"task_{int(time.time())}"

start_task_execution(
    task_id=task_id,
    task_name="Create a web application",
    metadata={"user": "developer", "project": "webapp"}
)

try:
    # Step 1: Generate code
    prompt = "Create a simple Flask web application"
    response = llm.generate(prompt)
    
    record_interaction(
        task_id=task_id,
        interaction_order=1,
        prompt=prompt,
        response=response,
        model_name="gpt-4"
    )
    
    # Step 2: Create files
    tool_result = create_file("app.py", response)
    
    record_tool_call(
        task_id=task_id,
        tool_name="create_file",
        tool_parameters={"filename": "app.py", "content": response},
        tool_result=tool_result,
        execution_time=0.3
    )
    
    # Record success metrics
    record_performance_metric(
        task_id=task_id,
        metric_name="files_created",
        metric_value=1
    )
    
    # Mark as completed
    total_duration = time.time() - start_time
    complete_task_execution(
        task_id=task_id,
        status="completed",
        total_duration=total_duration
    )
    
except Exception as e:
    # Mark as failed
    total_duration = time.time() - start_time
    complete_task_execution(
        task_id=task_id,
        status="failed",
        total_duration=total_duration,
        metadata={"error": str(e)}
    )
    raise
```

## Data Retrieval

### REST API Endpoints

The server also provides REST API endpoints for data retrieval:

- `GET /tasks/` - List all tasks
- `GET /tasks/{task_id}` - Get specific task details
- `GET /tasks/{task_id}/interactions/` - Get task interactions
- `GET /tasks/{task_id}/tool-calls/` - Get task tool calls
- `GET /analytics/summary` - Get overall analytics

### Example API Usage

```bash
# Get all tasks
curl http://localhost:8001/tasks/

# Get specific task
curl http://localhost:8001/tasks/task_1234567890

# Get task interactions
curl http://localhost:8001/tasks/task_1234567890/interactions/

# Get analytics summary
curl http://localhost:8001/analytics/summary
```

## Monitoring and Debugging

### Logs

The server logs all operations. Check logs for debugging:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run server
python main.py --mode mcp
```

### Health Check

Check server health:

```bash
curl http://localhost:8001/health
```

### Database Queries

Direct database queries for advanced analysis:

```sql
-- Get all interactions for a task
SELECT * FROM interactions 
WHERE task_execution_id = (
    SELECT id FROM task_executions WHERE task_id = 'your_task_id'
);

-- Get tool usage statistics
SELECT tool_name, COUNT(*) as usage_count, 
       AVG(execution_time) as avg_time
FROM tool_calls 
GROUP BY tool_name;
```

## Best Practices

1. **Unique Task IDs**: Use unique, descriptive task IDs
2. **Consistent Recording**: Record at every step for complete traceability
3. **Error Handling**: Always record failures and errors
4. **Performance Tracking**: Record timing and resource usage metrics
5. **Metadata**: Include relevant context in metadata fields
6. **Regular Cleanup**: Archive old data to maintain performance

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check DATABASE_URL environment variable
   - For file-based SQLite, ensure the directory is writable
   - For in-memory SQLite, no additional setup is required

2. **MCP Server Not Responding**
   - Verify server is running on correct port
   - Check Cline configuration
   - Review server logs

3. **Data Not Being Recorded**
   - Verify tool calls are successful
   - Check database table creation
   - Review error logs

### Getting Help

- Check server logs for error messages
- Verify database connectivity
- Test individual API endpoints
- Review Cline MCP integration documentation