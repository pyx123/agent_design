# MCP Recording Rules

## Overview
This project uses a custom MCP (Model Context Protocol) server to record all interactions and tool calls during Cline task execution. This provides comprehensive logging and analytics for debugging, optimization, and audit purposes.

## MCP Server Integration
- **Server Name**: cline-recorder
- **Default Database**: SQLite in-memory mode
- **Recording Scope**: All LLM interactions, tool calls, and performance metrics

## Recording Guidelines

### 1. Task Lifecycle Management
- **ALWAYS** start recording at the beginning of each task using `start_task_execution`
- **ALWAYS** complete recording at the end using `complete_task_execution`
- Use unique, descriptive task IDs (e.g., `task_${timestamp}_${description}`)

### 2. Interaction Recording
- Record **EVERY** LLM interaction (prompt + response) using `record_interaction`
- Include model information, token usage, and cost when available
- Maintain sequential interaction order within each task

### 3. Tool Call Recording
- Record **EVERY** tool call execution using `record_tool_call`
- Include parameters, results, execution time, and success status
- Record both successful and failed tool calls

### 4. Performance Tracking
- Record performance metrics using `record_performance_metric`
- Track timing, resource usage, and custom metrics
- Include relevant units for all measurements

## Required MCP Tools
The following tools must be available and used consistently:

1. `start_task_execution` - Initialize task recording
2. `record_interaction` - Log LLM interactions
3. `record_tool_call` - Log tool executions
4. `record_performance_metric` - Track performance data
5. `complete_task_execution` - Finalize task recording
6. `get_task_summary` - Retrieve task analytics

## Error Handling
- If MCP recording fails, log the error but continue with the main task
- Never let recording failures block primary functionality
- Include error details in task metadata when recording fails

## Data Privacy
- Be mindful of sensitive information in prompts and responses
- Use metadata fields to mark sensitive content appropriately
- Consider data retention policies for recorded information

## Performance Considerations
- Recording should have minimal impact on task execution time
- Use asynchronous recording when possible
- Batch related operations to reduce overhead

## Best Practices
- Use descriptive task names and metadata
- Include context information in metadata fields
- Maintain consistent naming conventions for metrics
- Regularly review recorded data for insights and optimization opportunities