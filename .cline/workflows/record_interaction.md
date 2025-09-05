# Record Interaction Workflow

## Purpose
Record LLM interactions (prompts and responses) during task execution for analysis and debugging.

## When to Use
- After every significant LLM interaction
- When generating code or solutions
- After receiving responses that will be used in subsequent steps

## Steps

### 1. Prepare Interaction Data
Gather the following information:
- **Prompt**: The exact prompt sent to the LLM
- **Response**: The complete response received
- **Model**: Model name/version used
- **Tokens**: Token usage (if available)
- **Cost**: Estimated cost (if available)
- **Context**: Additional context or metadata

### 2. Record the Interaction
Call the MCP tool to record the interaction:
```json
{
  "tool": "record_interaction",
  "parameters": {
    "task_id": "task_1234567890_web_app_development",
    "interaction_order": 1,
    "prompt": "Create a Flask web application with user authentication system",
    "response": "I'll help you create a Flask web application with user authentication...",
    "model_name": "gpt-4",
    "tokens_used": 150,
    "cost": 0.003,
    "metadata": {
      "temperature": 0.7,
      "max_tokens": 2000,
      "interaction_type": "code_generation",
      "complexity": "medium",
      "success": true
    }
  }
}
```

### 3. Update Interaction Counter
Increment the interaction order for the next recording.

## Interaction Types
Classify interactions in metadata:
- **code_generation**: Creating new code
- **code_review**: Reviewing existing code
- **debugging**: Troubleshooting issues
- **planning**: Task planning and architecture
- **documentation**: Creating documentation
- **testing**: Writing or reviewing tests
- **refactoring**: Improving existing code

## Metadata Guidelines
Include relevant context:
- **temperature**: LLM temperature setting
- **max_tokens**: Maximum tokens requested
- **interaction_type**: Type of interaction
- **complexity**: Complexity level (low/medium/high)
- **success**: Whether the interaction was successful
- **follow_up_needed**: Whether follow-up is required
- **quality_score**: Subjective quality rating (1-10)

## Example Usage
```
I just received a response from the LLM about creating the authentication system. Let me record this interaction.

[record_interaction.md]
```

## Best Practices
- Record interactions immediately after receiving responses
- Include sufficient context in metadata for later analysis
- Be consistent with interaction ordering
- Mark sensitive information appropriately
- Include quality assessments for performance tracking