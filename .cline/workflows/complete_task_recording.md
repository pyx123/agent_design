# Complete Task Recording Workflow

## Purpose
Finalize task recording and mark the task as completed with comprehensive summary data.

## When to Use
- At the end of any task or project
- When completing a coding session
- After finishing a multi-step operation
- When task is successfully completed or failed

## Steps

### 1. Calculate Final Metrics
Gather final task statistics:
- **Total Duration**: Time from start to completion
- **Total Interactions**: Number of LLM interactions
- **Total Tool Calls**: Number of tool executions
- **Success Rate**: Percentage of successful operations
- **Final Status**: completed, failed, or partial

### 2. Record Final Performance Metrics
Record any remaining performance data:
```json
{
  "tool": "record_performance_metric",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "metric_name": "total_duration",
    "metric_value": 7200,
    "metric_unit": "seconds"
  }
}
```

### 3. Complete Task Recording
Call the MCP tool to finalize recording:
```json
{
  "tool": "complete_task_execution",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "status": "completed",
    "total_duration": 7200,
    "metadata": {
      "final_result": "success",
      "files_created": 8,
      "lines_of_code": 450,
      "test_coverage": 85.5,
      "total_cost": 0.25,
      "completion_percentage": 100,
      "quality_score": 8.5,
      "lessons_learned": "Authentication system implemented successfully",
      "next_steps": "Add user management features",
      "end_time": "2024-01-15T12:30:00Z"
    }
  }
}
```

### 4. Get Final Summary
Retrieve the complete task summary:
```json
{
  "tool": "get_task_summary",
  "parameters": {
    "task_id": "task_1234567890_web_app_development"
  }
}
```

## Status Options
- **completed**: Task finished successfully
- **failed**: Task failed with errors
- **partial**: Task partially completed
- **cancelled**: Task was cancelled by user
- **timeout**: Task exceeded time limits

## Final Metadata Guidelines
Include comprehensive final data:
- **final_result**: Overall result (success/failure/partial)
- **files_created**: Number of files created
- **lines_of_code**: Total lines of code written
- **test_coverage**: Test coverage percentage
- **total_cost**: Total financial cost
- **completion_percentage**: Percentage of task completed
- **quality_score**: Overall quality rating (1-10)
- **lessons_learned**: Key insights from the task
- **next_steps**: Recommended follow-up actions
- **end_time**: ISO timestamp of task completion

## Example Usage
```
I've successfully completed the web application development. Let me finalize the task recording.

[complete_task_recording.md]
```

## Error Handling
For failed tasks, include error details:
```json
{
  "status": "failed",
  "metadata": {
    "error_message": "Database connection failed",
    "error_type": "ConnectionError",
    "retry_attempts": 3,
    "completion_percentage": 60,
    "partial_results": "Authentication system completed, database integration failed"
  }
}
```

## Success Criteria
A task is considered successfully completed when:
- All primary objectives are met
- Code is functional and tested
- Documentation is updated
- No critical errors remain
- Performance meets requirements

## Best Practices
- Always complete task recording, even for failed tasks
- Include comprehensive final metadata
- Record lessons learned for future improvement
- Provide clear next steps for follow-up work
- Use consistent status and metadata formats
- Include both quantitative and qualitative assessments