# LangGraph 基于大语言模型的 DevOps 多智能体排障系统 — 迭代1设计文档

## 迭代1目标：最小可行产品(MVP)

### 核心目标
- 实现基本的多智能体排障功能
- 支持日志、指标、告警三种数据源
- 提供同步REST API接口
- 实现基本的状态管理和错误处理
- 单租户部署

## 1. 背景与目标
- **目标**: 构建一个基于 LangGraph 的多智能体 DevOps 排障系统MVP，支持从问题到根因定位与修复建议的基本自动化流程。
- **范围**: 实现核心排障功能，暂不包含高级安全、成本控制、多租户等特性。

## 2. 用户角色与用户故事
- **SRE/值班工程师**: 输入"支付延迟升高"，系统自动生成排障计划，调用日志/告警/KPI 智能体并汇总输出根因与修复建议。

## 3. 范围
### 迭代1范围：
- 计划生成、执行编排、串行排查、证据收集、根因与建议生成
- 数据源：日志、告警、指标（只读）
- 基本的错误处理和重试机制
- SQLite本地存储
- 同步REST API

### 非迭代1范围：
- 高级安全特性（认证授权、加密、审计）
- 成本控制和性能优化
- 多租户支持
- gRPC/WebSocket/CLI接口
- 自动修复执行

## 4. 简化的系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          REST API                                │
│                         (FastAPI)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    LangGraph Core                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │State Manager│  │Graph Builder │  │  Node Executor     │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                        Agent Layer                               │
│  ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────┐ ┌─────────────┐  │
│  │ Planner │ │   Log   │ │ Alarm │ │  KPI  │ │   Summary   │  │
│  │  Agent  │ │  Agent  │ │ Agent │ │ Agent │ │    Agent    │  │
│  └────┬────┘ └────┬────┘ └───┬───┘ └───┬───┘ └──────┬──────┘  │
└───────┼───────────┼──────────┼─────────┼─────────────┼──────────┘
        │           │          │         │             │
┌───────┴───────────┴──────────┴─────────┴─────────────┴──────────┐
│                       MCP Client (简化版)                         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────┴───────────────────────────────────┐
│                    External MCP Servers                           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                       │
│  │  Logs   │  │ Metrics  │  │  Alerts  │                       │
│  │ Server  │  │  Server  │  │  Server  │                       │
│  └─────────┘  └──────────┘  └──────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

## 5. 核心状态模型（简化版）

```python
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

class GraphState(TypedDict, total=False):
    """迭代1的简化状态模型"""
    # 请求信息
    request_id: str
    request: dict  # TroubleshootingRequest
    
    # 执行计划
    plan: dict  # InvestigationPlan
    tasks: List[dict]  # List[InvestigationTask]
    current_task_index: int
    
    # 收集的数据
    evidence: List[dict]  # List[Evidence]
    findings: List[dict]  # List[Finding]
    
    # 结果
    root_cause: Optional[dict]  # RootCause
    remediation: Optional[dict]  # Remediation
    
    # 控制流
    next_actions: List[str]
    errors: List[dict]  # List[AgentError]
    status: str  # "init", "planning", "investigating", "summarizing", "completed", "failed"
    done: bool
    
    # 元数据
    created_at: datetime
    updated_at: datetime
```

## 6. 数据模型（简化版）

```python
# 请求模型
@dataclass
class TroubleshootingRequest:
    id: str
    title: str
    description: Optional[str]
    service: str
    environment: str = "prod"
    severity: str = "medium"
    time_range: Dict[str, str]  # {"from": "ISO8601", "to": "ISO8601"}

# 任务模型
@dataclass
class InvestigationTask:
    task_id: str
    type: str  # "log", "alarm", "kpi"
    inputs: Dict[str, Any]
    status: str = "pending"  # "pending", "running", "completed", "failed"

# 证据模型
@dataclass
class Evidence:
    evidence_id: str
    source: str  # "log", "alarm", "kpi"
    summary: str
    raw_ref: Dict[str, Any]  # 查询参数引用
    confidence: float = 0.8

# 根因模型
@dataclass
class RootCause:
    hypothesis: str
    confidence: float
    supporting_evidence: List[str]  # evidence_ids

# 修复建议模型
@dataclass
class Remediation:
    actions: List[str]
    risk_level: str  # "low", "medium", "high"
    estimated_time: str
```

## 7. LangGraph 实现

### 7.1 图构建器

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_troubleshooting_graph():
    """构建迭代1的排障图"""
    graph = StateGraph(GraphState)
    
    # 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("log_agent", log_agent_node)
    graph.add_node("alarm_agent", alarm_agent_node)
    graph.add_node("kpi_agent", kpi_agent_node)
    graph.add_node("summarize", summarize_node)
    
    # 设置入口
    graph.set_entry_point("planner")
    
    # 添加边（简化的条件路由）
    graph.add_conditional_edges(
        "planner",
        route_next_agent,
        {
            "log_agent": "log_agent",
            "alarm_agent": "alarm_agent",
            "kpi_agent": "kpi_agent",
            "summarize": "summarize",
            "end": END
        }
    )
    
    # Agent执行后返回planner
    graph.add_edge("log_agent", "planner")
    graph.add_edge("alarm_agent", "planner")
    graph.add_edge("kpi_agent", "planner")
    
    # 总结后结束
    graph.add_edge("summarize", END)
    
    # 编译图
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
```

### 7.2 核心节点实现

```python
async def planner_node(state: GraphState) -> GraphState:
    """计划节点：生成和更新调查计划"""
    if not state.get("plan"):
        # 初始计划生成
        plan = await generate_initial_plan(state["request"])
        state["plan"] = plan
        state["tasks"] = plan["tasks"]
        state["current_task_index"] = 0
    else:
        # 根据已收集的证据更新计划
        state = await update_plan_based_on_evidence(state)
    
    # 决定下一步行动
    if state.get("done") or has_sufficient_evidence(state):
        state["next_actions"] = ["summarize"]
    else:
        next_task = get_next_task(state)
        if next_task:
            state["next_actions"] = [next_task["type"]]
        else:
            state["next_actions"] = ["summarize"]
    
    return state

async def log_agent_node(state: GraphState) -> GraphState:
    """日志分析节点"""
    task = get_current_task(state)
    
    try:
        # 调用MCP日志工具
        log_results = await query_logs(
            service=task["inputs"]["service"],
            time_range=task["inputs"]["time_range"],
            keywords=task["inputs"].get("keywords", [])
        )
        
        # 生成证据
        evidence = create_evidence_from_logs(log_results, task)
        state["evidence"].append(evidence)
        
        # 更新任务状态
        update_task_status(state, task["task_id"], "completed")
        
    except Exception as e:
        # 错误处理
        state["errors"].append({
            "agent": "log_agent",
            "error": str(e),
            "task_id": task["task_id"]
        })
        update_task_status(state, task["task_id"], "failed")
    
    return state

def route_next_agent(state: GraphState) -> str:
    """路由到下一个节点"""
    next_actions = state.get("next_actions", [])
    
    if not next_actions:
        return "end"
    
    action = next_actions[0]
    
    if action == "query_logs":
        return "log_agent"
    elif action == "query_alarms":
        return "alarm_agent"
    elif action == "query_kpis":
        return "kpi_agent"
    elif action == "summarize":
        return "summarize"
    else:
        return "end"
```

## 8. MCP 集成（简化版）

```python
class SimpleMCPClient:
    """迭代1的简化MCP客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.servers = {}
        for server_config in config["servers"]:
            self.servers[server_config["id"]] = {
                "url": server_config["url"],
                "timeout": server_config.get("timeout", 30)
            }
    
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict) -> Dict:
        """调用MCP工具"""
        server = self.servers.get(server_id)
        if not server:
            raise ValueError(f"Unknown server: {server_id}")
        
        # 简化的HTTP调用
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server['url']}/tools/{tool_name}",
                json=arguments,
                timeout=aiohttp.ClientTimeout(total=server["timeout"])
            ) as response:
                return await response.json()
```

## 9. API 接口（仅REST）

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="DevOps Agent API - Iteration 1")

class TroubleshootingRequestModel(BaseModel):
    title: str
    description: Optional[str] = None
    service: str
    environment: str = "prod"
    severity: str = "medium"
    time_range: Dict[str, str]

@app.post("/troubleshoot", response_model=Dict[str, str])
async def create_troubleshooting(request: TroubleshootingRequestModel):
    """创建排障请求（同步执行）"""
    # 创建请求ID
    request_id = generate_request_id()
    
    # 初始化状态
    initial_state = {
        "request_id": request_id,
        "request": request.dict(),
        "status": "init",
        "created_at": datetime.now()
    }
    
    # 执行排障流程
    graph = get_troubleshooting_graph()
    result = await graph.ainvoke(initial_state)
    
    # 保存结果
    save_result_to_db(request_id, result)
    
    return {
        "id": request_id,
        "status": "completed",
        "root_cause": result.get("root_cause", {}).get("hypothesis"),
        "confidence": result.get("root_cause", {}).get("confidence", 0)
    }

@app.get("/troubleshoot/{request_id}")
async def get_troubleshooting_result(request_id: str):
    """获取排障结果"""
    result = get_result_from_db(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return {
        "id": request_id,
        "status": result["status"],
        "request": result["request"],
        "evidence": result.get("evidence", []),
        "root_cause": result.get("root_cause"),
        "remediation": result.get("remediation")
    }
```

## 10. 基础错误处理

```python
class BasicErrorHandler:
    """迭代1的基础错误处理"""
    
    @staticmethod
    async def handle_agent_error(state: GraphState, error: Exception, agent_name: str):
        """处理智能体错误"""
        error_record = {
            "agent": agent_name,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        state["errors"].append(error_record)
        
        # 简单重试逻辑
        retry_count = count_retries(state, agent_name)
        if retry_count < 3 and is_retriable_error(error):
            # 标记需要重试
            state["next_actions"] = [f"retry_{agent_name}"]
        else:
            # 跳过失败的任务
            advance_to_next_task(state)
        
        return state
```

## 11. 数据存储（SQLite）

```sql
-- 简化的数据库模式
CREATE TABLE troubleshooting_requests (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    service TEXT NOT NULL,
    environment TEXT,
    severity TEXT,
    status TEXT DEFAULT 'init',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    request_json TEXT,
    result_json TEXT
);

CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    source TEXT NOT NULL,
    summary TEXT,
    raw_ref_json TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);
```

## 12. 配置文件

```yaml
# config/iteration1.yaml
app:
  name: "DevOps Agent - Iteration 1"
  version: "0.1.0"
  environment: "development"

api:
  host: "0.0.0.0"
  port: 8080
  timeout: 300  # 5分钟超时

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 2000

mcp:
  servers:
    - id: "logs"
      url: "http://localhost:8081"
      timeout: 30
    - id: "metrics"
      url: "http://localhost:8082"
      timeout: 30
    - id: "alerts"
      url: "http://localhost:8083"
      timeout: 30

database:
  path: "./data/devops_agent.db"

logging:
  level: "INFO"
  format: "json"
```

## 13. 部署（单机Docker）

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY config/ ./config/

# 创建数据目录
RUN mkdir -p /app/data

# 环境变量
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/iteration1.yaml

# 启动命令
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devops-agent:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped

  # Mock MCP服务器（用于开发测试）
  mcp-logs:
    image: mock-mcp-server:latest
    ports:
      - "8081:8080"
    environment:
      - SERVICE_TYPE=logs

  mcp-metrics:
    image: mock-mcp-server:latest
    ports:
      - "8082:8080"
    environment:
      - SERVICE_TYPE=metrics

  mcp-alerts:
    image: mock-mcp-server:latest
    ports:
      - "8083:8080"
    environment:
      - SERVICE_TYPE=alerts
```

## 14. 测试计划（迭代1）

### 14.1 单元测试
```python
# 测试计划生成
def test_planner_generates_valid_plan():
    request = create_test_request()
    plan = generate_initial_plan(request)
    assert len(plan["tasks"]) >= 3
    assert any(t["type"] == "log" for t in plan["tasks"])

# 测试证据收集
def test_evidence_creation():
    log_results = {"entries": [...]}
    evidence = create_evidence_from_logs(log_results, test_task)
    assert evidence["source"] == "log"
    assert evidence["confidence"] > 0
```

### 14.2 集成测试
```python
# 测试完整流程
async def test_end_to_end_troubleshooting():
    graph = build_troubleshooting_graph()
    state = create_test_state()
    result = await graph.ainvoke(state)
    assert result["status"] == "completed"
    assert result.get("root_cause") is not None
```

## 15. 开发计划

### 第1周：基础框架
- [ ] 搭建项目结构
- [ ] 实现GraphState和基础数据模型
- [ ] 创建LangGraph图框架
- [ ] 设置数据库模式

### 第2周：Agent实现
- [ ] 实现Planner Agent
- [ ] 实现Log Agent
- [ ] 实现Alarm Agent
- [ ] 实现KPI Agent
- [ ] 实现Summary Agent

### 第3周：集成与API
- [ ] MCP客户端集成
- [ ] REST API实现
- [ ] 错误处理机制
- [ ] 基础测试

### 第4周：测试与部署
- [ ] 集成测试
- [ ] Docker化
- [ ] 部署文档
- [ ] 用户测试

## 16. 成功标准

1. **功能完整性**
   - 能够接收排障请求并返回根因分析
   - 支持日志、指标、告警三种数据源
   - 提供结构化的修复建议

2. **基本性能**
   - 单次排障在5分钟内完成
   - 支持并发5个请求
   - 系统可用性 > 95%

3. **可用性**
   - API文档完整
   - 错误信息清晰
   - 部署过程简单

## 17. 已知限制

1. 单租户系统
2. 无认证授权机制
3. 仅支持同步API
4. 无成本控制
5. 基础错误处理
6. 无高级监控

这些限制将在迭代2中解决。