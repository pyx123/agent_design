### LangGraph 基于大语言模型的 DevOps 多智能体排障系统 — 详细设计

#### 1. 背景与目标
- **目标**: 构建一个基于 LangGraph 的多智能体 DevOps 排障系统，支持从问题到根因定位与修复建议的端到端自动化与半自动化协作。
- **愿景**: 降低 MTTR（平均修复时间）、提升排障一致性与透明度，支持企业级可观测性数据源（日志、告警、指标）与知识库联动，提供可追踪、可审计、可扩展的架构。

#### 2. 用户角色与典型用户故事
- **SRE/值班工程师**: 输入“支付延迟升高”，系统自动生成排障计划，调用日志/告警/KPI 智能体并汇总输出根因与修复建议；支持追问和增量迭代。
- **开发负责人**: 通过异步任务或 Slack/IM 触发分析，系统持续收集证据并在新线索出现时追加分析，最终产出工单描述与修复步骤草案。
- **平台工程团队**: 配置数据源（Elastic/Loki、Prometheus、Alertmanager、APM）、CMDB/服务拓扑、变更记录；配置 RBAC 与审计，将系统接入现有流水线与告警通道。

#### 3. 范围与非目标
- 范围：
  - 计划生成、执行编排、并行/串行排查、证据收集、根因与建议生成、可追踪性、最小化变更集成（只读为主，可选读写）。
  - 数据源：日志、告警、指标（必选），可扩展 APM/Tracing、变更/发布系统、CMDB、知识库。
- 非目标：
  - 自动执行破坏性操作（如直接回滚/扩容）默认禁用，仅输出建议与命令草案，人工确认后执行。

#### 4. 总体架构

##### 4.1 系统架构概览
```
┌─────────────────────────────────────────────────────────────────────┐
│                          API Gateway                                 │
│                             (REST)                                  │
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
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │ Connection   │  │   Request    │  │   Response Handler     │   │
│  │   Manager    │  │   Router     │  │   & Cache              │   │
│  └──────────────┘  └──────────────┘  └────────────────────────┘   │
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

##### 4.2 核心组件详解

- **编排核心**: 
  - LangGraph 状态机：管理排障流程的状态转换和执行逻辑
  - Flow Controller：控制智能体调用顺序、并发执行和条件分支
  - State Manager：维护GraphState的一致性和持久化

- **智能体层**:
  - Planner Agent: 
    - 职责：根据问题生成排障计划，动态路由调用其他智能体
    - 输入：用户问题描述、历史证据、执行状态
    - 输出：调查计划、下一步行动列表
  - Log Agent: 
    - 职责：面向日志平台查询、相关性分析、异常模式挖掘
    - 支持平台：Elastic/Loki等
    - 能力：全文搜索、模式匹配、时序关联
  - Alarm Agent: 
    - 职责：告警聚合、去噪、时间相关分析
    - 支持平台：Prometheus Alertmanager等
    - 能力：告警去重、根因关联、影响面分析
  - KPI Agent: 
    - 职责：指标趋势分析、异常检测、因果线索分析
    - 支持平台：Prometheus等
    - 能力：时序分析、基线对比、相关性检测
  - Summary Agent: 
    - 职责：聚合执行结果，推演根因、影响面与修复建议
    - 输出：结构化根因分析、修复方案、验证步骤

- **支撑能力**:
  - 工具层（MCP）：
    - MCP Client：统一的MCP客户端，管理与多个MCP Server的连接
    - 工具发现：动态发现和注册MCP Server提供的工具
    - 请求路由：根据工具类型路由到对应的MCP Server
    - 响应缓存：缓存高频查询结果，减少外部调用
  - 存储层：
    - SQLite：存储会话状态、执行历史、证据数据
    - 缓存：内存缓存，存储热点数据和中间结果
  - 观测层：
    - 链路追踪：完整的执行链路追踪
    - 日志聚合：结构化日志收集和分析


#### 5. LangGraph 运行时与状态模型

##### 5.1 状态管理架构
```
┌─────────────────────────────────────────────────────────────────┐
│                      GraphState Manager                          │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │State Validator│  │State Persister │  │State Transformer │  │
│  └───────────────┘  └────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

##### 5.2 GraphState 数据模型
```python
@dataclass
class GraphState:
    """单次排障会话的完整状态"""
    # 基础信息
    request: TroubleshootingRequest     # 用户请求
    session_id: str                     # 会话标识
    tenant_id: str                      # 租户标识
    
    # 执行计划
    plan: InvestigationPlan            # 当前调查计划
    tasks: List[InvestigationTask]     # 任务列表（含历史）
    current_task_id: Optional[str]     # 当前执行任务
    
    # 证据收集
    evidence: List[Evidence]           # 收集的所有证据
    findings: List[Finding]            # 阶段性发现
    evidence_graph: Dict[str, List[str]]  # 证据关联图
    
    # 执行控制
    next_actions: List[str]            # 下一步动作列表
    execution_path: List[str]          # 执行路径记录
    branch_history: List[BranchDecision]  # 分支决策历史
    
    # 结果输出
    root_cause: Optional[RootCause]    # 根因分析结果
    remediation: Optional[Remediation]  # 修复建议
    confidence_scores: Dict[str, float] # 各假设的置信度
    
    # 上下文管理
    history: List[Message]             # 对话历史
    context_window: int                # 上下文窗口大小
    memory_summary: Optional[str]      # 长对话的记忆摘要
    
    # 资源管理
    concurrency_limit: int            # 并发限制
    
    # 错误处理
    errors: List[AgentError]          # 错误记录
    retry_attempts: Dict[str, int]    # 重试次数记录
    
    # 状态标记
    status: ExecutionStatus           # 执行状态
    done: bool                        # 是否完成
    terminated_reason: Optional[str]  # 终止原因
    
    # 元数据
    created_at: datetime
    updated_at: datetime
```

##### 5.3 状态转换规则
```python
class StateTransitions:
    """状态转换定义"""
    
    VALID_TRANSITIONS = {
        ExecutionStatus.INIT: [ExecutionStatus.PLANNING],
        ExecutionStatus.PLANNING: [ExecutionStatus.INVESTIGATING, ExecutionStatus.FAILED],
        ExecutionStatus.INVESTIGATING: [
            ExecutionStatus.ANALYZING, 
            ExecutionStatus.PLANNING,  # 重新规划
            ExecutionStatus.FAILED
        ],
        ExecutionStatus.ANALYZING: [ExecutionStatus.SUMMARIZING, ExecutionStatus.INVESTIGATING],
        ExecutionStatus.SUMMARIZING: [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED],
        ExecutionStatus.FAILED: [ExecutionStatus.PLANNING],  # 允许重试
        ExecutionStatus.COMPLETED: []  # 终态
    }
    
    @staticmethod
    def is_valid_transition(from_status: ExecutionStatus, to_status: ExecutionStatus) -> bool:
        return to_status in StateTransitions.VALID_TRANSITIONS.get(from_status, [])
```

##### 5.4 执行流程设计

- **节点定义**：
  ```python
  class NodeTypes(Enum):
      PLANNER = "planner"           # 计划生成节点
      ROUTER = "router"             # 路由分发节点
      LOG_AGENT = "log_agent"       # 日志分析节点
      ALARM_AGENT = "alarm_agent"   # 告警分析节点
      KPI_AGENT = "kpi_agent"       # 指标分析节点
      SUMMARIZER = "summarizer"     # 总结分析节点
      FINALIZER = "finalizer"       # 最终处理节点
  ```

- **边定义与条件**：
  ```python
  EDGE_CONDITIONS = {
      (NodeTypes.PLANNER, NodeTypes.ROUTER): lambda state: not state.done,
      (NodeTypes.ROUTER, NodeTypes.LOG_AGENT): lambda state: "query_logs" in state.next_actions,
      (NodeTypes.ROUTER, NodeTypes.ALARM_AGENT): lambda state: "query_alarms" in state.next_actions,
      (NodeTypes.ROUTER, NodeTypes.KPI_AGENT): lambda state: "query_kpis" in state.next_actions,
      (NodeTypes.ROUTER, NodeTypes.SUMMARIZER): lambda state: "summarize" in state.next_actions,
      (NodeTypes.LOG_AGENT, NodeTypes.PLANNER): lambda state: True,
      (NodeTypes.ALARM_AGENT, NodeTypes.PLANNER): lambda state: True,
      (NodeTypes.KPI_AGENT, NodeTypes.PLANNER): lambda state: True,
      (NodeTypes.SUMMARIZER, NodeTypes.FINALIZER): lambda state: True,
  }
  ```

- **并发控制机制**：
  ```python
  class ConcurrencyController:
      def __init__(self, max_workers: int = 3, rate_limits: Dict[str, RateLimit] = None):
          self.max_workers = max_workers
          self.rate_limits = rate_limits or {}
          self.semaphore = asyncio.Semaphore(max_workers)
          self.task_queue = asyncio.Queue()
          
      async def execute_parallel(self, tasks: List[AgentTask]) -> List[TaskResult]:
          """并行执行多个智能体任务，支持限流和优先级"""
          # 按优先级排序
          sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
          
          # 创建任务组
          async with asyncio.TaskGroup() as tg:
              for task in sorted_tasks:
                  # 检查速率限制
                  if await self.check_rate_limit(task.agent_type):
                      tg.create_task(self.execute_with_limit(task))
                  else:
                      # 加入延迟队列
                      await self.task_queue.put(task)
                      
          return await self.collect_results()
  ```

#### 6. 数据契约（关键 Schema 摘要）
- TroubleshootingRequest
  - id, tenant_id, title, description, service, environment, severity, time_range, artifacts_hints
- InvestigationPlan
  - plan_id, created_by, goals[], tasks[]: InvestigationTask
- InvestigationTask
  - task_id, type in {log, alarm, kpi, knowledge, change}, inputs, hypotheses[], priority, timeout_s
- Evidence
  - evidence_id, source in {log, alarm, kpi, knowledge, change}, summary, raw_ref, time_window, quality_score
- Finding
  - finding_id, hypothesis_ref, confidence, impact_scope, supporting_evidence[]
- RootCause
  - hypothesis, confidence, affected_components[], time_correlation, change_correlation
- Remediation
  - actions[], risk, required_approvals[], validation_steps[]
- AgentError
  - agent, error_type, message, retriable, attempt
- TokenCost
  - prompt_tokens, completion_tokens, tool_calls, wall_time_ms

注：完整 JSON Schema 建议单独放入 `docs/schemas/*.json` 以便前后端/服务间对接与校验。

#### 7. 提示词与角色（Prompt）设计要点
- 通用 System 提示：
  - 目标驱动（目标/约束/输出格式），引用计划、证据、历史上下文。
  - 严格输出结构化 JSON（使用 JSON Schema 标注，模型端加“只输出 JSON”提醒）。
- Planner 提示：
  - 输入：问题描述、上下文、当前证据/发现、历史任务与结果、时间预算。
  - 输出：更新后的 InvestigationPlan 与下一步 `next_actions`。
- Log/Alarm/KPI 提示：
  - 输入：任务参数（指标名/日志索引/服务/时间窗）+ 查询建议与样例。
  - 输出：结构化证据（摘要+引用），避免长原文粘贴，提供查询复现信息。
- Summary 提示：
  - 输入：完整计划与证据集合。
  - 输出：`root_cause` + `remediation` + 验证与回滚步骤 + 工单摘要。

#### 8. 工具层（MCP）抽象
- 设计原则：由外部 MCP Server 暴露工具（tools/resources/prompts），系统通过 MCP 客户端统一发现与调用，避免为每个数据源单独写 SDK 适配层。
- 映射关系：
  - Log Agent -> MCP 工具示例：`log.search`, `log.tail`, `log.summary`
  - Alarm Agent -> MCP 工具示例：`alert.list`, `alert.get`, `alert.summary`
  - KPI Agent -> MCP 工具示例：`metric.query`, `metric.range_query`, `metric.baseline`
- LangGraph 工具桥接（示例）：
```python
from typing import Any
from langchain_core.tools import tool
from mcp_client import McpClient  # 假设的 MCP 客户端

mcp = McpClient()  # 通过配置建立到多个 MCP Server 的连接

@tool
def mcp_tool_invoker(server_id: str, tool_name: str, arguments_json: str) -> str:
    """调用外部 MCP 工具。参数为 server_id、tool_name 与 JSON 字符串化的 arguments。返回工具执行结果的 JSON 字符串。"""
    return mcp.call_tool(server_id=server_id, tool_name=tool_name, arguments_json=arguments_json)
```

- 注意事项：
  - 调用超时/重试/速率限制在本系统层实现；
  - 工具返回需附带可复现信息（查询语句、时间窗、分页游标），作为证据引用；
  - 不同租户/环境可绑定不同 MCP Server 与凭证；
  - 仅保存摘要与引用，原始数据由外部系统托管。

#### 9. LangGraph 实现蓝图（代码骨架）
```python
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class GraphState(TypedDict, total=False):
    request: dict
    plan: dict
    tasks: List[dict]
    evidence: List[dict]
    findings: List[dict]
    root_cause: Optional[dict]
    remediation: Optional[dict]
    next_actions: List[str]
    history: List[dict]
    cost_usage: dict
    errors: List[dict]
    done: bool


def planner_node(state: GraphState) -> GraphState:
    # 调用 LLM（带工具结果与历史），更新 plan/tasks/next_actions
    updated = state.copy()
    # ... 调用 LLM 并生成/更新任务 ...
    updated["next_actions"] = ["query_logs", "query_kpis"]
    return updated


def route_node(state: GraphState) -> GraphState:
    # 由 next_actions 决定后续边
    return state


def log_agent_node(state: GraphState) -> GraphState:
    # 调用日志工具，生成 evidence，可能新增 findings
    return state


def alarm_agent_node(state: GraphState) -> GraphState:
    return state


def kpi_agent_node(state: GraphState) -> GraphState:
    return state


def summarize_node(state: GraphState) -> GraphState:
    # 汇总证据 -> root_cause & remediation
    updated = state.copy()
    updated["done"] = True
    return updated


def should_route_to_logs(state: GraphState) -> bool:
    return "query_logs" in (state.get("next_actions") or [])


def should_route_to_kpis(state: GraphState) -> bool:
    return "query_kpis" in (state.get("next_actions") or [])


def should_summarize(state: GraphState) -> bool:
    return state.get("root_cause") is not None or not state.get("next_actions")


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("route", route_node)
    graph.add_node("log_agent", log_agent_node)
    graph.add_node("alarm_agent", alarm_agent_node)
    graph.add_node("kpi_agent", kpi_agent_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "route")

    # 动态边：根据 next_actions 决定流转与并行分支
    graph.add_conditional_edges(
        "route",
        lambda s: "summarize" if should_summarize(s) else "fanout",
        {
            "summarize": "summarize",
            "fanout": "planner"  # 这里可替换为并行子图控制器
        }
    )

    # 示例：直接连 summarize -> END
    graph.add_edge("summarize", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
```

说明：实际实现中，`route` 通常会构建一个“并行分发”子图（或在 Planner 内部将多个子任务分发至并发执行队列），等待所有分支完成或达到阈值后再回到 planner 迭代。

#### 10. 终止条件与迭代策略
- 终止条件：
  - 找到高置信度根因；
  - 达到时间预算；
  - 用户主动终止；
  - 计划无进一步可执行任务。
- 迭代：
  - 每轮收集新证据后，Planner 复审假设并增量调整任务集；
  - 保留历史任务以便审计与回放。

#### 11. 观测与可视化
- 运行事件总线：记录进入/退出节点、条件分支选择、错误、重试；

#### 12. 对外接口（API/CLI）

##### 12.1 RESTful API 规范

```yaml
openapi: 3.0.0
info:
  title: DevOps Agent API
  version: 1.0.0
  description: AI-powered troubleshooting system API

paths:
  /troubleshoot:
    post:
      summary: 创建排障任务
      operationId: createTroubleshootingRequest
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TroubleshootingRequest'
      responses:
        '201':
          description: 任务创建成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                    format: uuid
                  status:
                    type: string
                    enum: [queued, running]
                  mode:
                    type: string
                    enum: [sync, async]
        '400':
          $ref: '#/components/responses/BadRequest'
    
    get:
      summary: 列出排障任务
      operationId: listTroubleshootingRequests
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: size
          in: query
          schema:
            type: integer
            default: 20
        - name: status
          in: query
          schema:
            type: string
            enum: [running, completed, failed]
        - name: service
          in: query
          schema:
            type: string
      responses:
        '200':
          description: 任务列表
          content:
            application/json:
              schema:
                type: object
                properties:
                  items:
                    type: array
                    items:
                      $ref: '#/components/schemas/TroubleshootingSummary'
                  total:
                    type: integer
                  page:
                    type: integer
  
  /troubleshoot/{id}:
    get:
      summary: 获取排障详情
      operationId: getTroubleshootingRequest
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: 排障详情
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TroubleshootingResult'
        '404':
          $ref: '#/components/responses/NotFound'
    
    patch:
      summary: 更新排障任务
      operationId: updateTroubleshootingRequest
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                action:
                  type: string
                  enum: [cancel, retry, continue]
                additional_context:
                  type: string
      responses:
        '200':
          description: 更新成功
  
  /troubleshoot/{id}/evidence:
    get:
      summary: 获取收集的证据
      operationId: getEvidence
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
        - name: source
          in: query
          schema:
            type: string
            enum: [log, kpi, alarm]
      responses:
        '200':
          description: 证据列表
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Evidence'
  
  /troubleshoot/{id}/report:
    get:
      summary: 获取分析报告
      operationId: getReport
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
        - name: format
          in: query
          schema:
            type: string
            enum: [markdown, html, pdf]
            default: markdown
      responses:
        '200':
          description: 分析报告
          content:
            text/markdown:
              schema:
                type: string
            text/html:
              schema:
                type: string
            application/pdf:
              schema:
                type: string
                format: binary
  

components:
  schemas:
    TroubleshootingRequest:
      type: object
      required: [title, service]
      properties:
        title:
          type: string
          description: 问题标题
          example: "支付服务p99延迟高"
        description:
          type: string
          description: 问题详细描述
        service:
          type: string
          description: 受影响的服务
          example: "payment-service"
        environment:
          type: string
          enum: [prod, staging, dev]
          default: prod
        severity:
          type: string
          enum: [critical, high, medium, low]
          default: medium
        time_range:
          type: object
          properties:
            from:
              type: string
              format: date-time
            to:
              type: string
              format: date-time
        artifacts_hints:
          type: object
          description: 额外的上下文信息
        mode:
          type: string
          enum: [sync, async]
          default: async
    
    TroubleshootingResult:
      type: object
      properties:
        id:
          type: string
        status:
          type: string
          enum: [running, completed, failed]
        request:
          $ref: '#/components/schemas/TroubleshootingRequest'
        plan:
          $ref: '#/components/schemas/InvestigationPlan'
        evidence:
          type: array
          items:
            $ref: '#/components/schemas/Evidence'
        root_cause:
          $ref: '#/components/schemas/RootCause'
        remediation:
          $ref: '#/components/schemas/Remediation'
        created_at:
          type: string
          format: date-time
        completed_at:
          type: string
          format: date-time
```

##### 12.2 WebSocket API

```typescript
// WebSocket事件定义
interface WebSocketEvents {
  // 客户端 -> 服务器
  "subscribe": {
    request_id: string;
  };
  
  "unsubscribe": {
    request_id: string;
  };
  
  // 服务器 -> 客户端
  "status_update": {
    request_id: string;
    status: string;
    progress: number;
  };
  
  "evidence_collected": {
    request_id: string;
    evidence: Evidence;
  };
  
  "task_completed": {
    request_id: string;
    task_id: string;
    result: any;
  };
  
  "analysis_complete": {
    request_id: string;
    root_cause: RootCause;
    remediation: Remediation;
  };
}

// WebSocket连接示例
const ws = new WebSocket('wss://api.devops-agent.com/ws');

ws.on('open', () => {
  // 订阅排障任务
  ws.send(JSON.stringify({
    type: 'subscribe',
    data: { request_id: 'req_123' }
  }));
});

ws.on('message', (data) => {
  const event = JSON.parse(data);
  switch(event.type) {
    case 'status_update':
      updateUI(event.data);
      break;
    case 'evidence_collected':
      displayEvidence(event.data);
      break;
  }
});
```

#### 13. 测试策略

##### 13.1 测试架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     测试金字塔                                   │
│                                                                 │
│     ┌─────────────────────────────────────────┐               │
│     │        端到端测试 (E2E)                  │  5%          │
│     │   - 完整排障流程                         │               │
│     └───────────────┬───────────────────────────┘               │
│         ┌───────────┴───────────────────┐                     │
│         │      集成测试                  │      15%            │
│         │  - Agent间协作                 │                     │
│         │  - MCP集成                    │                     │
│         └───────────┬───────────────────┘                     │
│     ┌───────────────┴───────────────────────┐                 │
│     │          组件测试                      │    30%          │
│     │    - 单个Agent功能                     │                 │
│     │    - 状态管理                          │                 │
│     └───────────────┬───────────────────────┘                 │
│ ┌───────────────────┴───────────────────────────┐             │
│ │              单元测试                          │   50%       │
│ │  - 工具函数、解析器、验证器                    │             │
│ └───────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

##### 13.2 单元测试策略

```python
# 提示词解析测试
class TestPromptParser:
    def test_parse_planner_output_valid(self):
        """测试有效的Planner输出解析"""
        output = '{"plan": {"goals": ["定位延迟"], "tasks": []}, "next_actions": ["query_logs"]}'
        result = PromptParser.parse_planner_output(output)
        assert result["next_actions"] == ["query_logs"]
    
    def test_parse_invalid_json(self):
        """测试无效JSON处理"""
        with pytest.raises(ParseError):
            PromptParser.parse_planner_output("invalid json")

# 错误分类测试
class TestErrorClassifier:
    @pytest.mark.parametrize("error,expected", [
        (TimeoutError(), ErrorType.TRANSIENT),
        (PermissionError(), ErrorType.PERMANENT),
    ])
    def test_classify_error(self, error, expected):
        assert ErrorClassifier.classify(error) == expected
```

##### 13.3 组件测试策略

```python
# Agent组件测试
class TestPlannerAgent:
    async def test_generate_plan(self):
        """测试计划生成"""
        agent = PlannerAgent(llm=MockLLM())
        state = GraphState(request=create_test_request())
        result = await agent.generate_plan(state)
        assert len(result.plan.tasks) >= 3
        assert any(t.type == "log" for t in result.plan.tasks)

# 状态管理测试
class TestStateManager:
    async def test_state_transitions(self):
        """测试状态转换"""
        manager = StateManager()
        state = GraphState(status=ExecutionStatus.INIT)
        state = await manager.transition(state, ExecutionStatus.PLANNING)
        assert state.status == ExecutionStatus.PLANNING
```

##### 13.4 集成测试策略

```python
# 多Agent协作测试
class TestAgentIntegration:
    async def test_multi_agent_flow(self):
        """测试多智能体协作流程"""
        graph = build_test_graph()
        result = await graph.ainvoke({
            "request": create_latency_request()
        })
        assert len(result["evidence"]) >= 3
        assert result["root_cause"] is not None

# MCP集成测试
class TestMCPIntegration:
    async def test_tool_discovery(self):
        """测试工具发现"""
        client = MCPClient(test_config)
        tools = await client.discover_tools()
        assert "log.search" in tools
```

##### 13.5 端到端测试策略

```python
# 完整流程测试
class TestE2EScenarios:
    async def test_complete_troubleshooting(self):
        """测试完整排障流程"""
        # 创建请求
        resp = await client.post("/troubleshoot", json={
            "title": "高延迟问题",
            "service": "checkout"
        })
        request_id = resp.json()["id"]
        
        # 等待完成
        await wait_for_completion(client, request_id)
        
        # 验证结果
        result = await client.get(f"/troubleshoot/{request_id}")
        assert result.json()["status"] == "completed"
        assert result.json()["root_cause"] is not None
```
##### 13.6 测试数据管理

```python
class TestDataFactory:
    """测试数据工厂"""
    
    @staticmethod
    def create_log_anomaly():
        """创建日志异常数据"""
        return [
            LogEntry(level="ERROR", message="Connection timeout"),
            LogEntry(level="ERROR", message="5xx response")
        ]
    
    @staticmethod
    def create_metric_spike():
        """创建指标尖峰数据"""
        return [
            MetricPoint(value=100),  # 正常
            MetricPoint(value=500),  # 异常尖峰
        ]
```

#### 14. 参考目录结构（建议）
```
.
├─ docs/
│  ├─ design-langgraph-devops-agents.md
│  ├─ prompts/             # 各 Agent 的提示词模板（本仓已创建）
│  └─ schemas/
├─ src/
│  ├─ app.py                # API/CLI 入口
│  ├─ graph/
│  │  ├─ state.py           # GraphState 与 schema
│  │  ├─ builder.py         # LangGraph 构建
│  │  └─ nodes/             # planner/log/alarm/kpi/summary 节点实现
│  ├─ tools/                # MCP 工具桥接封装
│  ├─ services/             # 缓存、重试
└─ config/
   ├─ default.yaml
   └─ sql/
      └─ ddl.sql            # SQLite DDL（见下文）
```

#### 15. 示例请求与输出（简）
- 请求：`支付服务 p99 延迟在 10:00-10:30 升高`。
- Planner：生成查询日志（checkout pod）、指标（http_server_request_duration_seconds{service="checkout"}）与关联最近变更的任务。
- 证据：日志中 5xx 激增、指标 p99 抬升、10:05 有配置变更；
- 总结：疑似配置回滚缺失导致连接池参数过小，建议扩大连接池并回滚到稳定版本；提供验证步骤与回滚指令草案。

—— 本文档可直接指导开发落地，配合 `src/graph/builder.py` 实现与工具封装逐步交付。

#### 16. SQLite DDL（v0）
```sql
-- 基础请求与状态
CREATE TABLE IF NOT EXISTS troubleshooting_requests (
  id TEXT PRIMARY KEY,
  tenant_id TEXT,
  title TEXT,
  description TEXT,
  service TEXT,
  environment TEXT,
  severity TEXT,
  time_from TEXT,
  time_to TEXT,
  status TEXT DEFAULT 'running',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS investigation_plans (
  plan_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  created_by TEXT,
  goals_json TEXT,
  plan_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);

CREATE TABLE IF NOT EXISTS investigation_tasks (
  task_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  plan_id TEXT,
  type TEXT,
  inputs_json TEXT,
  hypotheses_json TEXT,
  priority INTEGER,
  timeout_s INTEGER,
  status TEXT DEFAULT 'pending',
  result_summary TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT,
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id),
  FOREIGN KEY(plan_id) REFERENCES investigation_plans(plan_id)
);

CREATE TABLE IF NOT EXISTS evidences (
  evidence_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  task_id TEXT,
  source TEXT,
  summary TEXT,
  raw_ref_json TEXT,
  time_from TEXT,
  time_to TEXT,
  quality_score REAL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id),
  FOREIGN KEY(task_id) REFERENCES investigation_tasks(task_id)
);

CREATE TABLE IF NOT EXISTS findings (
  finding_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  hypothesis_ref TEXT,
  confidence REAL,
  impact_scope_json TEXT,
  supporting_evidence_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);

CREATE TABLE IF NOT EXISTS root_causes (
  id TEXT PRIMARY KEY,
  request_id TEXT UNIQUE NOT NULL,
  hypothesis TEXT,
  confidence REAL,
  affected_components_json TEXT,
  time_correlation_json TEXT,
  change_correlation_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);

CREATE TABLE IF NOT EXISTS remediations (
  id TEXT PRIMARY KEY,
  request_id TEXT UNIQUE NOT NULL,
  actions_json TEXT,
  required_approvals_json TEXT,
  validation_steps_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);

CREATE TABLE IF NOT EXISTS agent_errors (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  agent TEXT,
  error_type TEXT,
  message TEXT,
  retriable INTEGER,
  attempt INTEGER,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  role TEXT,
  content TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);
```

#### 17. MCP 配置样例（v0）
```yaml
# config/default.yaml（片段）
mcp:
  servers:
    logs:
      server_id: logs
      transport: http
      endpoint: https://mcp-logs.example.com
      token_env: MCP_LOGS_TOKEN
    metrics:
      server_id: metrics
      transport: http
      endpoint: https://mcp-metrics.example.com
      token_env: MCP_METRICS_TOKEN
    alerts:
      server_id: alerts
      transport: http
      endpoint: https://mcp-alerts.example.com
      token_env: MCP_ALERTS_TOKEN
  mapping:
    log:
      server: logs
      tools: [log.search, log.summary]
    kpi:
      server: metrics
      tools: [metric.query, metric.range_query]
    alarm:
      server: alerts
      tools: [alert.list, alert.summary]
```

#### 18. API 契约（v0）
```http
POST /troubleshoot
Content-Type: application/json

{
  "title": "p99 latency high",
  "service": "checkout",
  "environment": "prod",
  "severity": "high",
  "time_range": {"from": "2025-01-01T10:00:00Z", "to": "2025-01-01T10:30:00Z"},
  "description": "p99 > 2s",
  "artifacts_hints": {"routes": ["/pay"]}
}

--> 201
{
  "id": "req_123",
  "status": "running"
}
```

```http
GET /troubleshoot/req_123

--> 200
{
  "id": "req_123",
  "status": "running|done",
  "plan": { },
  "tasks": [ ],
  "evidence": [ ],
  "root_cause": null,
  "remediation": null
}
```

#### 19. Prompt 规范（v0）
- 统一约束：所有 Agent 严格输出 JSON；失败时输出 `{ "error": {"type":..., "message":...} }`。
- Planner 输出：
```json
{
  "plan": {"goals": ["..."], "tasks": [{"task_id": "t1", "type": "log", "inputs": {}}]},
  "next_actions": ["query_logs", "query_kpis"]
}
```
- Log/Alarm/KPI 输出：
```json
{
  "evidence": [
    {"evidence_id": "e1", "source": "log", "summary": "...", "raw_ref": {"query": "...", "server": "logs"}}
  ],
  "findings": []
}
```
- Summary 输出：
```json
{
  "root_cause": {"hypothesis": "...", "confidence": 0.82},
  "remediation": {"actions": ["..."], "validation_steps": ["..."]},
  "report_md": "# Incident Summary\n..."
}
```
提示词模板位置：`docs/prompts/*.md`（已创建）。

#### 20. 最小运行示例（v0）
```python
from src.graph.builder import build_graph

if __name__ == "__main__":
    app = build_graph()
    state = {
        "request": {
            "id": "req_demo",
            "title": "p99 latency high",
            "service": "checkout",
            "environment": "prod"
        }
    }
    result = app.invoke(state)
    print(result.get("root_cause"))
```

#### 21. v0 实施清单
- 初始化 SQLite（执行本节 DDL），实现 `src/services/repo.py` 读写。
- 实现 LangGraph 节点：planner/route/log_agent/alarm_agent/kpi_agent/summarize。
- 集成 MCP 客户端与工具桥接 `src/tools/mcp_bridge.py`。
- 打通 REST API：POST/GET /troubleshoot，绑定图执行与入库。
- 填充 `docs/prompts/*.md` 的具体模板，确保输出 JSON 符合第25节。
- 增加 2-3 个合成用例数据，跑通端到端。

