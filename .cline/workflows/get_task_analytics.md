# Get Task Analytics Workflow

## Purpose
Retrieve and analyze task execution data for insights, debugging, and optimization.

## When to Use
- During task execution for progress monitoring
- After task completion for analysis
- When debugging issues or performance problems
- For generating reports and summaries

## Steps

### 1. Get Task Summary
Retrieve basic task information:
```json
{
  "tool": "get_task_summary",
  "parameters": {
    "task_id": "task_1234567890_web_app_development"
  }
}
```

### 2. Analyze Task Data
Review the returned data for insights:
- **Task Status**: Current status and completion percentage
- **Duration**: Time spent on the task
- **Interactions**: Number of LLM interactions
- **Tool Calls**: Number of tool executions
- **Performance**: Performance metrics recorded

### 3. Generate Analytics Report
Create a comprehensive analysis report based on the data.

## Analytics Categories

### Performance Analytics
- **Execution Time**: Total and breakdown by phase
- **Efficiency**: Interactions per hour, tool calls per hour
- **Resource Usage**: Memory, CPU, network usage
- **Cost Analysis**: Total cost and cost per interaction

### Quality Analytics
- **Success Rate**: Percentage of successful operations
- **Error Analysis**: Types and frequency of errors
- **Code Quality**: Quality scores and metrics
- **Test Coverage**: Testing completeness

### Usage Analytics
- **Tool Usage**: Most frequently used tools
- **Interaction Patterns**: Common interaction types
- **Model Performance**: LLM model effectiveness
- **Workflow Efficiency**: Time spent on different activities

## Example Analytics Report
```
Task Analytics Report: Web Application Development
================================================

Task ID: task_1234567890_web_app_development
Status: completed
Duration: 2 hours (7,200 seconds)
Completion: 100%

Interactions:
- Total: 15 LLM interactions
- Average per hour: 7.5 interactions
- Success rate: 93.3%

Tool Calls:
- Total: 45 tool executions
- Most used: write_file (12), read_file (8), run_command (6)
- Success rate: 95.6%

Performance:
- Total cost: $0.25
- Average cost per interaction: $0.017
- Code generated: 450 lines
- Files created: 8
- Test coverage: 85.5%

Quality Score: 8.5/10
```

## Usage Patterns

### During Task Execution
```json
{
  "tool": "get_task_summary",
  "parameters": {
    "task_id": "current_task_id"
  }
}
```

### After Task Completion
```json
{
  "tool": "get_task_summary",
  "parameters": {
    "task_id": "completed_task_id"
  }
}
```

### For Multiple Tasks
Retrieve summaries for multiple tasks to compare performance and identify patterns.

## Example Usage
```
Let me check the analytics for the web application development task to see how it performed.

[get_task_analytics.md]
```

## Insights and Recommendations
Based on analytics data, provide:
- **Performance Insights**: What worked well and what didn't
- **Optimization Opportunities**: Areas for improvement
- **Best Practices**: Successful patterns to repeat
- **Resource Planning**: Better estimates for future tasks
- **Quality Improvements**: Ways to enhance output quality

## Best Practices
- Review analytics regularly during long tasks
- Use analytics to identify bottlenecks
- Compare performance across similar tasks
- Track improvements over time
- Share insights with team members
- Use data to optimize workflows and processes