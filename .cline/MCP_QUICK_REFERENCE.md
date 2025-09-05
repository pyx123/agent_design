# MCP Recording Quick Reference

## Quick Start
For any new task, use the complete workflow:
```
[full_task_with_recording.md]
```

## Individual Workflows

### Task Management
- `[start_task_recording.md]` - Begin recording a new task
- `[complete_task_recording.md]` - Finish and finalize task recording

### Recording Operations
- `[record_interaction.md]` - Log LLM interactions
- `[record_tool_call.md]` - Log tool executions
- `[record_performance_metric.md]` - Track performance metrics

### Analytics
- `[get_task_analytics.md]` - Get task summary and analytics

## MCP Tools Available
1. `start_task_execution` - Initialize task recording
2. `record_interaction` - Log LLM interactions
3. `record_tool_call` - Log tool executions
4. `record_performance_metric` - Track performance data
5. `complete_task_execution` - Finalize task recording
6. `get_task_summary` - Retrieve task analytics

## Common Patterns

### Starting a New Task
```
I'm beginning a new task to [description]. Let me start recording.

[start_task_recording.md]
```

### Recording an Interaction
```
I just received a response from the LLM. Let me record this interaction.

[record_interaction.md]
```

### Recording a Tool Call
```
I just used the [tool_name] tool. Let me record this execution.

[record_tool_call.md]
```

### Tracking Performance
```
I've completed [operation]. Let me record the performance metrics.

[record_performance_metric.md]
```

### Completing a Task
```
The task is complete. Let me finalize the recording.

[complete_task_recording.md]
```

### Getting Analytics
```
Let me check the task analytics to see how it's performing.

[get_task_analytics.md]
```

## Task ID Format
Use descriptive, unique task IDs:
- Format: `task_{timestamp}_{description}`
- Example: `task_1705312345_web_app_development`
- Example: `task_1705312345_user_authentication_system`

## Metadata Examples

### Task Metadata
```json
{
  "user": "developer",
  "project": "webapp",
  "priority": "high",
  "technologies": ["Python", "Flask", "SQLite"],
  "estimated_duration": "2 hours"
}
```

### Interaction Metadata
```json
{
  "temperature": 0.7,
  "interaction_type": "code_generation",
  "complexity": "medium",
  "success": true,
  "quality_score": 8
}
```

### Tool Call Metadata
```json
{
  "tool_version": "1.0.0",
  "operation_type": "file_creation",
  "file_size": 1024,
  "permissions": "644"
}
```

### Performance Metadata
```json
{
  "execution_time": 45.2,
  "memory_usage": 128,
  "total_cost": 0.25,
  "quality_score": 8.5
}
```

## Error Handling
- If recording fails, log the error but continue with the main task
- Never block primary functionality due to recording issues
- Include recording failures in task metadata

## Best Practices
- Record everything consistently
- Use descriptive metadata
- Maintain sequential ordering
- Include both success and failure cases
- Track performance metrics
- Review analytics regularly