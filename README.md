# DevOps Agent - AI-Powered Troubleshooting System

A LangGraph-based multi-agent system for automated DevOps troubleshooting, integrating with logs, metrics, and alerts through the Model Context Protocol (MCP).

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for logs, metrics, alerts, and root cause analysis
- **LangGraph Orchestration**: Intelligent workflow management with state persistence
- **MCP Integration**: Unified interface to external observability tools
- **Real-time Updates**: WebSocket support for live troubleshooting progress
- **RESTful API**: Easy integration with existing DevOps tools
- **Flexible LLM Support**: Works with OpenAI and Anthropic models

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          API Gateway                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                    LangGraph Orchestrator                            │
│  ┌─────────────┐  ┌──────────────┐                                   │
│  │State Manager│  │Flow Controller│                                │
│  └─────────────┘  └──────────────┘                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                        Agent Layer                                   │
│  ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────┐ ┌─────────────────┐  │
│  │ Planner │ │   Log   │ │ Alarm │ │  KPI  │ │    Summary      │  │
│  │  Agent  │ │  Agent  │ │ Agent │ │ Agent │ │     Agent       │  │
│  └────┬────┘ └────┬────┘ └───┬───┘ └───┬───┘ └────────┬────────┘  │
└───────┼───────────┼──────────┼─────────┼──────────────┼────────────┘
        │           │          │         │              │
┌───────┴───────────┴──────────┴─────────┴──────────────┴────────────┐
│                       MCP Client Layer                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                    External MCP Servers                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │  Logs   │  │ Metrics  │  │  Alerts  │  │ Knowledge Base   │    │
│  │ Server  │  │  Server  │  │  Server  │  │    Server        │    │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- OpenAI or Anthropic API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/devops-agent.git
cd devops-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Initialize the database:
```bash
python -c "import asyncio; from src.services import init_database; asyncio.run(init_database())"
```

### Running the Application

#### Using Python directly:
```bash
python -m src.app
```

#### Using the CLI:
```bash
python -m src.cli serve
```

#### Using Docker:
```bash
docker-compose up -d
```

## 📖 Usage

### API Examples

#### Create a troubleshooting request:
```bash
curl -X POST http://localhost:8000/api/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{
    "title": "High latency in payment service",
    "service": "payment-service",
    "environment": "prod",
    "severity": "high",
    "description": "Users reporting slow payment processing"
  }'
```

#### Check request status:
```bash
curl http://localhost:8000/api/v1/troubleshoot/{request_id}
```

#### Get analysis report:
```bash
curl http://localhost:8000/api/v1/troubleshoot/{request_id}/report
```

### CLI Usage

#### Create a troubleshooting request:
```bash
python -m src.cli troubleshoot \
  --title "Database connection issues" \
  --service "api-gateway" \
  --severity high \
  --sync  # Wait for completion
```

#### List recent requests:
```bash
python -m src.cli list --limit 10
```

#### Get request status:
```bash
python -m src.cli status {request_id}
```

#### Generate report:
```bash
python -m src.cli report {request_id} --format markdown --output report.md
```

### WebSocket Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to a request
ws.send(JSON.stringify({
  type: 'subscribe',
  data: { request_id: 'req_123' }
}));

// Handle updates
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Update:', message);
};
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/unit -v
```

Run integration tests:
```bash
pytest tests/integration -v
```

Run all tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `API_HOST` | API server host | 0.0.0.0 |
| `API_PORT` | API server port | 8000 |
| `DATABASE_URL` | Database connection URL | sqlite:///./devops_agent.db |
| `LOG_LEVEL` | Logging level | INFO |
| `MAX_CONCURRENT_AGENTS` | Max parallel agent execution | 3 |

See `.env.example` for full configuration options.

### MCP Server Configuration

Configure your MCP servers in the environment:

```bash
MCP_LOGS_SERVER_URL=http://your-logs-mcp-server:8081
MCP_LOGS_TOKEN=your-token

MCP_METRICS_SERVER_URL=http://your-metrics-mcp-server:8082
MCP_METRICS_TOKEN=your-token

MCP_ALERTS_SERVER_URL=http://your-alerts-mcp-server:8083
MCP_ALERTS_TOKEN=your-token
```

## 📚 API Documentation

When running in development mode, interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [LangChain](https://github.com/langchain-ai/langchain)
- MCP integration for observability tools
- OpenAI and Anthropic for LLM capabilities