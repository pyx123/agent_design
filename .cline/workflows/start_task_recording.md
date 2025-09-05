# Start Task Recording Workflow

## Purpose
Initialize comprehensive recording for a new Cline task execution session.

## When to Use
- At the beginning of any significant task or project
- When starting a new coding session
- Before beginning any multi-step operation

## Steps

### 1. Generate Unique Task ID
```python
import time
from datetime import datetime

# Generate unique task ID
timestamp = int(time.time())
task_id = f"task_{timestamp}_{task_description}"
```

### 2. Start Task Recording
Call the MCP tool to begin recording:
```json
{
  "tool": "start_task_execution",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "task_name": "Create a web application with user authentication",
    "metadata": {
      "user": "developer",
      "project": "webapp",
      "priority": "high",
      "estimated_duration": "2 hours",
      "technologies": ["Python", "Flask", "SQLite"],
      "start_time": "2024-01-15T10:30:00Z"
    }
  }
}
```

### 3. Verify Recording Started
Check that the task recording was initialized successfully:
```json
{
  "tool": "get_task_summary",
  "parameters": {
    "task_id": "task_1234567890_web_app_development"
  }
}
```

## Expected Output
- Confirmation that task recording has started
- Task ID for future reference
- Initial task summary showing status as "running"

## Error Handling
- If recording fails, log the error and continue with the task
- Include recording failure details in task metadata
- Never block main task execution due to recording issues

## Example Usage
```
I'm starting a new task to create a web application. Let me initialize recording first.

[start_task_recording.md]
```

## Metadata Guidelines
Include relevant context in metadata:
- **user**: Current user or developer
- **project**: Project name or identifier
- **priority**: Task priority level
- **technologies**: Technologies being used
- **estimated_duration**: Expected completion time
- **start_time**: ISO timestamp of task start
- **description**: Detailed task description