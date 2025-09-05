# Cline MCP Integration Rules

## Overview
This project integrates with a custom MCP (Model Context Protocol) server for comprehensive recording of all Cline interactions, tool calls, and performance metrics. This provides detailed analytics, debugging capabilities, and optimization insights.

## MCP Server Configuration
- **Server Name**: cline-recorder
- **Database**: SQLite in-memory mode (default)
- **Recording Scope**: All LLM interactions, tool calls, and performance metrics
- **Integration**: Seamless integration via MCP protocol

## Core Recording Principles

### 1. Always Record
- **EVERY** significant task must be recorded from start to finish
- **EVERY** LLM interaction must be logged
- **EVERY** tool call must be recorded (success and failure)
- **EVERY** performance metric must be tracked

### 2. Consistent Workflow
Use the established workflows for all recording operations:
- `[start_task_recording.md]` - Initialize task recording
- `[record_interaction.md]` - Log LLM interactions
- `[record_tool_call.md]` - Log tool executions
- `[record_performance_metric.md]` - Track performance data
- `[complete_task_recording.md]` - Finalize task recording
- `[get_task_analytics.md]` - Retrieve analytics
- `[full_task_with_recording.md]` - Complete task with full recording

### 3. Task Lifecycle Management
Every task follows this lifecycle:
1. **Initialize**: Start recording with unique task ID
2. **Execute**: Record all interactions and tool calls
3. **Monitor**: Track progress and performance
4. **Complete**: Finalize recording with comprehensive metadata

## Required MCP Tools
The following tools must be available and used consistently:

1. **start_task_execution** - Initialize task recording
2. **record_interaction** - Log LLM interactions
3. **record_tool_call** - Log tool executions
4. **record_performance_metric** - Track performance data
5. **complete_task_execution** - Finalize task recording
6. **get_task_summary** - Retrieve task analytics

## Recording Standards

### Task Identification
- Use descriptive, unique task IDs
- Format: `task_{timestamp}_{description}`
- Include relevant context in task names
- Maintain consistent naming conventions

### Interaction Recording
- Record immediately after each LLM interaction
- Include complete prompt and response
- Add model information, token usage, and cost
- Maintain sequential interaction ordering
- Classify interaction types in metadata

### Tool Call Recording
- Record every tool execution (success and failure)
- Include parameters, results, and execution time
- Add error details for failed operations
- Classify tools by category in metadata
- Track performance impact

### Performance Tracking
- Record timing for all major operations
- Track resource usage and costs
- Monitor success/failure rates
- Include quality assessments
- Measure efficiency metrics

## Metadata Guidelines

### Task Metadata
Include comprehensive context:
- **user**: Current user or developer
- **project**: Project name or identifier
- **priority**: Task priority level
- **technologies**: Technologies being used
- **estimated_duration**: Expected completion time
- **start_time**: ISO timestamp of task start
- **description**: Detailed task description

### Interaction Metadata
Classify and contextualize interactions:
- **temperature**: LLM temperature setting
- **max_tokens**: Maximum tokens requested
- **interaction_type**: Type of interaction
- **complexity**: Complexity level (low/medium/high)
- **success**: Whether the interaction was successful
- **quality_score**: Subjective quality rating (1-10)

### Tool Call Metadata
Provide execution context:
- **tool_version**: Version of the tool used
- **operation_type**: Type of operation performed
- **file_size**: Size of files involved
- **permissions**: File permissions set
- **error_type**: Type of error (if failed)
- **retry_attempted**: Whether retry was attempted

### Performance Metadata
Track comprehensive metrics:
- **execution_time**: Time taken for operations
- **memory_usage**: Memory consumption
- **token_usage**: LLM token consumption
- **cost**: Financial cost of operations
- **quality_score**: Overall quality rating
- **completion_percentage**: Task completion status

## Error Handling
- If MCP recording fails, log the error but continue with the main task
- Never let recording failures block primary functionality
- Include error details in task metadata when recording fails
- Provide fallback mechanisms for critical recording operations

## Quality Assurance
- Review recorded data regularly for insights
- Use analytics to identify optimization opportunities
- Track improvements over time
- Share insights with team members
- Use data to optimize workflows and processes

## Best Practices
- Use descriptive task names and metadata
- Include context information in metadata fields
- Maintain consistent naming conventions for metrics
- Regularly review recorded data for insights
- Use asynchronous recording when possible
- Batch related operations to reduce overhead
- Consider data privacy and retention policies

## Integration Commands
To use the recording system, invoke the appropriate workflows:
- `[start_task_recording.md]` - Begin a new task
- `[record_interaction.md]` - Log an LLM interaction
- `[record_tool_call.md]` - Log a tool execution
- `[record_performance_metric.md]` - Track performance
- `[complete_task_recording.md]` - Finish a task
- `[get_task_analytics.md]` - Get analytics
- `[full_task_with_recording.md]` - Complete task with full recording

## Success Criteria
A task is considered successfully recorded when:
- Task recording is initialized with unique ID
- All LLM interactions are logged with metadata
- All tool calls are recorded (success and failure)
- Performance metrics are tracked throughout
- Task is completed with comprehensive summary
- Analytics are available for review and analysis