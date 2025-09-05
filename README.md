# Cline Interaction Recorder MCP Server

This MCP (Model Context Protocol) server is designed to record and store interaction records and tool call records during Cline task execution. It provides a database-backed storage solution for tracking LLM interactions and tool usage in Cline workflows.

## Features

- **Interaction Recording**: Captures LLM interactions during task execution
- **Tool Call Logging**: Records all tool calls and their results
- **Database Storage**: SQLite backend with SQLAlchemy ORM (in-memory by default)
- **MCP Integration**: Seamless integration with Cline via MCP protocol
- **REST API**: Additional HTTP endpoints for data retrieval and management

## Architecture

The server consists of:
- MCP server implementation for Cline integration
- Database models for storing interaction records
- REST API for data access and management
- Configuration management with environment variables

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure database connection in `.env` (optional, defaults to in-memory SQLite)
3. Initialize database tables: `python main.py --init-db`
4. Start the MCP server: `python main.py`

## Usage

### MCP Integration
Cline can integrate this server as an MCP tool to automatically record:
- LLM prompts and responses
- Tool calls and their parameters
- Task execution steps and results
- Performance metrics and timing

### REST API
The server also provides HTTP endpoints for:
- Retrieving interaction history
- Analyzing tool usage patterns
- Exporting data for analysis
- Managing stored records

## Database Schema

- **interactions**: LLM conversation records
- **tool_calls**: Tool execution logs
- **task_executions**: Overall task tracking
- **performance_metrics**: Timing and resource usage

## Configuration

Set the following environment variables:
- `DATABASE_URL`: Database connection string (default: `sqlite:///:memory:`)
- `MCP_PORT`: MCP server port (default: 8000)
- `API_PORT`: REST API port (default: 8001)
- `LOG_LEVEL`: Logging level (default: INFO)