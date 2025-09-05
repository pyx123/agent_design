# Record Performance Metric Workflow

## Purpose
Record performance metrics and measurements during task execution for optimization and analysis.

## When to Use
- After completing significant operations
- When measuring execution times
- When tracking resource usage
- At regular intervals during long-running tasks

## Steps

### 1. Identify Metrics to Record
Common metrics include:
- **Execution Time**: Time taken for operations
- **Memory Usage**: Memory consumption
- **Token Usage**: LLM token consumption
- **Cost**: Financial cost of operations
- **File Size**: Size of generated files
- **Code Quality**: Quality metrics
- **Test Coverage**: Test coverage percentages

### 2. Record the Metric
Call the MCP tool to record the metric:
```json
{
  "tool": "record_performance_metric",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "metric_name": "total_execution_time",
    "metric_value": 45.2,
    "metric_unit": "seconds"
  }
}
```

### 3. Record Multiple Metrics
Record related metrics together:
```json
{
  "tool": "record_performance_metric",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "metric_name": "total_tokens_used",
    "metric_value": 1250,
    "metric_unit": "tokens"
  }
}
```

## Metric Categories

### Timing Metrics
- **total_execution_time**: Overall task duration
- **code_generation_time**: Time spent generating code
- **testing_time**: Time spent on testing
- **debugging_time**: Time spent debugging
- **file_operation_time**: Time for file operations

### Resource Metrics
- **memory_usage**: Memory consumption
- **cpu_usage**: CPU utilization
- **disk_usage**: Disk space used
- **network_usage**: Network bandwidth used

### Quality Metrics
- **code_quality_score**: Subjective code quality (1-10)
- **test_coverage**: Test coverage percentage
- **linting_errors**: Number of linting errors
- **complexity_score**: Code complexity rating

### Cost Metrics
- **total_cost**: Total financial cost
- **llm_cost**: Cost of LLM interactions
- **api_cost**: Cost of API calls
- **storage_cost**: Cost of storage usage

## Example Usage
```
I've completed the authentication system implementation. Let me record the performance metrics.

[record_performance_metric.md]
```

## Recording Patterns

### After Code Generation
```json
{
  "metric_name": "lines_of_code_generated",
  "metric_value": 150,
  "metric_unit": "lines"
}
```

### After Testing
```json
{
  "metric_name": "test_coverage",
  "metric_value": 85.5,
  "metric_unit": "percent"
}
```

### After File Operations
```json
{
  "metric_name": "files_created",
  "metric_value": 5,
  "metric_unit": "count"
}
```

## Best Practices
- Record metrics at logical completion points
- Use consistent metric names and units
- Include both quantitative and qualitative metrics
- Track trends over time
- Record baseline metrics for comparison
- Use meaningful unit names (seconds, bytes, count, percent)