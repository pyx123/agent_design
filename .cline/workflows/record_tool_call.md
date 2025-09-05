# Record Tool Call Workflow

## Purpose
Record tool call executions for comprehensive logging and performance analysis.

## When to Use
- After every tool call execution
- When using file operations, API calls, or other tools
- For both successful and failed tool executions

## Steps

### 1. Prepare Tool Call Data
Gather the following information:
- **Tool Name**: Name of the tool that was called
- **Parameters**: Input parameters passed to the tool
- **Result**: Output or result from the tool
- **Success**: Whether the tool call succeeded
- **Execution Time**: Time taken to execute (if measurable)
- **Error**: Error message (if failed)

### 2. Record the Tool Call
Call the MCP tool to record the execution:
```json
{
  "tool": "record_tool_call",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "tool_name": "write_file",
    "tool_parameters": {
      "file_path": "app.py",
      "content": "from flask import Flask, render_template...",
      "overwrite": true
    },
    "tool_result": {
      "success": true,
      "file_created": "app.py",
      "bytes_written": 1024
    },
    "success": true,
    "execution_time": 0.15,
    "metadata": {
      "tool_version": "1.0.0",
      "file_size": 1024,
      "operation_type": "file_creation",
      "permissions": "644"
    }
  }
}
```

### 3. Handle Failed Tool Calls
For failed executions, include error details:
```json
{
  "tool": "record_tool_call",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "tool_name": "read_file",
    "tool_parameters": {
      "file_path": "nonexistent.py"
    },
    "tool_result": null,
    "success": false,
    "error_message": "File not found: nonexistent.py",
    "execution_time": 0.05,
    "metadata": {
      "error_type": "FileNotFoundError",
      "retry_attempted": false,
      "fallback_used": false
    }
  }
}
```

## Tool Categories
Classify tools in metadata:
- **file_operations**: File read/write/delete operations
- **code_analysis**: Code parsing, linting, analysis
- **api_calls**: External API requests
- **database_operations**: Database queries and updates
- **system_commands**: Shell commands and system operations
- **web_operations**: Web scraping, HTTP requests
- **git_operations**: Version control operations

## Metadata Guidelines
Include relevant context:
- **tool_version**: Version of the tool used
- **operation_type**: Type of operation performed
- **file_size**: Size of files involved
- **permissions**: File permissions set
- **error_type**: Type of error (if failed)
- **retry_attempted**: Whether retry was attempted
- **fallback_used**: Whether fallback method was used
- **performance_impact**: Impact on overall performance

## Example Usage
```
I just used the write_file tool to create the main application file. Let me record this tool call.

[record_tool_call.md]
```

## Performance Tracking
- Record execution times for performance analysis
- Track tool usage patterns
- Monitor success/failure rates
- Identify bottlenecks and optimization opportunities

## Best Practices
- Record tool calls immediately after execution
- Include comprehensive parameter and result data
- Handle both success and failure cases
- Use consistent metadata classification
- Track performance metrics for optimization