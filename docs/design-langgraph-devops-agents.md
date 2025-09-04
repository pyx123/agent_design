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
│                    (REST/gRPC/WebSocket)                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                    LangGraph Orchestrator                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │State Manager│  │Flow Controller│  │ Checkpoint Manager        │ │
│  └─────────────┘  └──────────────┘  └───────────────────────────┘ │
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
  - Checkpoint Manager：支持断点续传和状态回放

- **智能体层**:
  - Planner Agent: 
    - 职责：根据问题生成排障计划，动态路由调用其他智能体
    - 输入：用户问题描述、历史证据、执行状态
    - 输出：调查计划、下一步行动列表
  - Log Agent: 
    - 职责：面向日志平台查询、相关性分析、异常模式挖掘
    - 支持平台：Elastic/Loki/Splunk/CloudWatch等
    - 能力：全文搜索、模式匹配、时序关联
  - Alarm Agent: 
    - 职责：告警聚合、去噪、时间相关分析
    - 支持平台：Prometheus Alertmanager/PagerDuty/OpsGenie等
    - 能力：告警去重、根因关联、影响面分析
  - KPI Agent: 
    - 职责：指标趋势分析、异常检测、因果线索分析
    - 支持平台：Prometheus/Thanos/InfluxDB/CloudWatch等
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
    - 缓存：Redis/内存缓存，存储热点数据和中间结果
    - 对象存储：存储大型报告和归档数据（可选）
  - 观测层：
    - OpenTelemetry：统一的可观测性框架
    - 指标收集：请求延迟、成功率、资源使用等
    - 链路追踪：完整的执行链路追踪
    - 日志聚合：结构化日志收集和分析
  - 安全与合规层：
    - 认证：API Key/JWT/OAuth2等多种认证方式
    - 授权：基于RBAC的细粒度权限控制
    - 审计：完整的操作审计日志
    - 数据安全：敏感数据加密和脱敏

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
    cost_usage: CostUsage             # 资源使用统计
    time_budget: TimeBudget           # 时间预算控制
    concurrency_limit: int            # 并发限制
    
    # 错误处理
    errors: List[AgentError]          # 错误记录
    retry_attempts: Dict[str, int]    # 重试次数记录
    circuit_breakers: Dict[str, CircuitBreakerState]  # 熔断器状态
    
    # 状态标记
    status: ExecutionStatus           # 执行状态
    done: bool                        # 是否完成
    terminated_reason: Optional[str]  # 终止原因
    
    # 元数据
    created_at: datetime
    updated_at: datetime
    version: int                      # 状态版本号
    checkpoints: List[StateCheckpoint] # 检查点记录
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
      GUARD = "guard"               # 安全检查节点
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

- **熔断器设计**：
  ```python
  class CircuitBreaker:
      def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
          self.failure_threshold = failure_threshold
          self.timeout = timeout
          self.failure_count = 0
          self.last_failure_time = None
          self.state = CircuitState.CLOSED
          
      async def call(self, func, *args, **kwargs):
          if self.state == CircuitState.OPEN:
              if self._should_attempt_reset():
                  self.state = CircuitState.HALF_OPEN
              else:
                  raise CircuitOpenError("Circuit breaker is OPEN")
                  
          try:
              result = await func(*args, **kwargs)
              self._on_success()
              return result
          except Exception as e:
              self._on_failure()
              raise
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

#### 11. 可靠性与错误处理

##### 11.1 错误分类与处理策略

```python
class ErrorType(Enum):
    """错误类型分类"""
    TRANSIENT = "transient"           # 暂时性错误（网络超时、限流等）
    PERMANENT = "permanent"           # 永久性错误（权限不足、资源不存在等）
    PARTIAL = "partial"               # 部分失败（批量操作部分成功）
    DEGRADED = "degraded"             # 降级错误（功能降级但可继续）
    CRITICAL = "critical"             # 关键错误（需要立即停止）

class ErrorHandler:
    """统一错误处理器"""
    
    ERROR_STRATEGIES = {
        ErrorType.TRANSIENT: {
            "retry": True,
            "max_attempts": 3,
            "backoff": "exponential",
            "fallback": "skip_and_continue"
        },
        ErrorType.PERMANENT: {
            "retry": False,
            "fallback": "use_alternative_source"
        },
        ErrorType.PARTIAL: {
            "retry": False,
            "fallback": "process_successful_parts"
        },
        ErrorType.DEGRADED: {
            "retry": False,
            "fallback": "continue_with_warning"
        },
        ErrorType.CRITICAL: {
            "retry": False,
            "fallback": "terminate_gracefully"
        }
    }
```

##### 11.2 重试机制设计

```python
class RetryPolicy:
    """重试策略配置"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        
    def get_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = min(
            self.initial_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay

class RetryableOperation:
    """可重试操作包装器"""
    
    async def execute_with_retry(self, 
                                operation: Callable,
                                policy: RetryPolicy,
                                error_classifier: Callable[[Exception], ErrorType]):
        attempt = 0
        last_error = None
        
        while attempt < policy.max_attempts:
            try:
                return await operation()
            except Exception as e:
                error_type = error_classifier(e)
                if error_type not in [ErrorType.TRANSIENT, ErrorType.PARTIAL]:
                    raise
                    
                last_error = e
                attempt += 1
                
                if attempt < policy.max_attempts:
                    delay = policy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    
        raise MaxRetriesExceeded(f"Operation failed after {attempt} attempts", last_error)
```

##### 11.3 降级策略

```python
class DegradationStrategy:
    """服务降级策略"""
    
    DEGRADATION_LEVELS = {
        "full_feature": {
            "data_sources": ["logs", "metrics", "alerts", "knowledge"],
            "analysis_depth": "comprehensive",
            "confidence_threshold": 0.8
        },
        "essential_only": {
            "data_sources": ["logs", "metrics"],
            "analysis_depth": "basic",
            "confidence_threshold": 0.6
        },
        "minimal": {
            "data_sources": ["logs"],
            "analysis_depth": "surface",
            "confidence_threshold": 0.4
        }
    }
    
    def __init__(self):
        self.current_level = "full_feature"
        self.degradation_reasons = []
        
    async def evaluate_and_degrade(self, 
                                  available_sources: List[str], 
                                  error_rate: float) -> str:
        """评估并决定降级级别"""
        if error_rate > 0.5 or len(available_sources) < 2:
            self.current_level = "minimal"
            self.degradation_reasons.append("High error rate or insufficient data sources")
        elif error_rate > 0.2 or len(available_sources) < 3:
            self.current_level = "essential_only"
            self.degradation_reasons.append("Moderate error rate or limited data sources")
            
        return self.current_level
```

##### 11.4 熔断器实现

```python
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 失败阈值
    success_threshold: int = 2          # 成功阈值（半开状态）
    timeout: float = 60.0              # 开启状态持续时间
    half_open_max_calls: int = 3       # 半开状态最大调用次数

class EnhancedCircuitBreaker:
    """增强型熔断器"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.metrics = CircuitBreakerMetrics()
        
    async def call(self, operation: Callable, fallback: Optional[Callable] = None):
        """通过熔断器调用操作"""
        # 检查熔断器状态
        if not self._can_proceed():
            self.metrics.record_rejection()
            if fallback:
                return await fallback()
            raise CircuitOpenError(f"Circuit breaker {self.name} is OPEN")
            
        # 执行操作
        try:
            result = await operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            if fallback and self.state == CircuitState.OPEN:
                return await fallback()
            raise
```

##### 11.5 错误聚合与分析

```python
class ErrorAggregator:
    """错误聚合分析器"""
    
    def __init__(self, window_size: int = 300):  # 5分钟窗口
        self.window_size = window_size
        self.error_buffer = deque()
        self.error_patterns = {}
        
    def record_error(self, error: AgentError):
        """记录错误"""
        self.error_buffer.append({
            "timestamp": datetime.now(),
            "error": error,
            "fingerprint": self._generate_fingerprint(error)
        })
        self._cleanup_old_errors()
        self._analyze_patterns()
        
    def _analyze_patterns(self):
        """分析错误模式"""
        # 按错误指纹分组
        fingerprint_groups = defaultdict(list)
        for item in self.error_buffer:
            fingerprint_groups[item["fingerprint"]].append(item)
            
        # 识别错误模式
        self.error_patterns = {
            "burst_errors": self._detect_burst_errors(fingerprint_groups),
            "recurring_errors": self._detect_recurring_errors(fingerprint_groups),
            "correlated_errors": self._detect_correlated_errors(fingerprint_groups)
        }
        
    def get_error_insights(self) -> Dict[str, Any]:
        """获取错误洞察"""
        return {
            "error_rate": len(self.error_buffer) / self.window_size,
            "top_errors": self._get_top_errors(),
            "patterns": self.error_patterns,
            "recommendations": self._generate_recommendations()
        }
```

##### 11.6 运行时监控与告警

```python
class RuntimeMonitor:
    """运行时监控器"""
    
    def __init__(self):
        self.metrics = {
            "agent_latency": {},
            "error_rates": {},
            "circuit_breaker_states": {},
            "resource_usage": {}
        }
        self.alert_rules = []
        self.alert_handler = AlertHandler()
        
    async def monitor_execution(self, state: GraphState):
        """监控执行状态"""
        # 收集指标
        await self._collect_metrics(state)
        
        # 检查告警规则
        alerts = self._evaluate_alert_rules()
        
        # 发送告警
        for alert in alerts:
            await self.alert_handler.send_alert(alert)
            
        # 自适应调整
        await self._adaptive_tuning(state)
        
    def _evaluate_alert_rules(self) -> List[Alert]:
        """评估告警规则"""
        alerts = []
        
        # 错误率告警
        for agent, rate in self.metrics["error_rates"].items():
            if rate > 0.3:
                alerts.append(Alert(
                    level="warning",
                    title=f"High error rate for {agent}",
                    details=f"Error rate: {rate:.2%}"
                ))
                
        # 延迟告警
        for agent, latency in self.metrics["agent_latency"].items():
            if latency > 30:  # 30秒
                alerts.append(Alert(
                    level="warning",
                    title=f"High latency for {agent}",
                    details=f"Latency: {latency:.2f}s"
                ))
                
        return alerts
```

##### 11.7 错误恢复流程

```yaml
error_recovery_flow:
  1_detect:
    - 捕获异常
    - 分类错误类型
    - 记录错误上下文
    
  2_decide:
    - 查询错误处理策略
    - 检查熔断器状态
    - 评估降级选项
    
  3_execute:
    - 执行重试（如适用）
    - 触发降级（如需要）
    - 调用备用方案
    
  4_record:
    - 更新错误统计
    - 记录恢复操作
    - 发送监控指标
    
  5_learn:
    - 分析错误模式
    - 更新处理策略
    - 优化系统配置
```

#### 12. 安全与合规

##### 12.1 安全架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Gateway                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │Authentication│  │Authorization │  │ Rate Limiting          ││
│  │  (JWT/OAuth) │  │    (RBAC)    │  │ & DDoS Protection      ││
│  └─────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Security Middleware                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │  Request    │  │   Data       │  │    Audit               ││
│  │ Validation  │  │ Sanitization │  │   Logging              ││
│  └─────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

##### 12.2 认证机制

```python
class AuthenticationManager:
    """多种认证方式支持"""
    
    SUPPORTED_METHODS = {
        "api_key": APIKeyAuthenticator,
        "jwt": JWTAuthenticator,
        "oauth2": OAuth2Authenticator,
        "mtls": MutualTLSAuthenticator
    }
    
    async def authenticate(self, request: Request) -> AuthContext:
        """统一认证入口"""
        # 识别认证方式
        auth_method = self._identify_auth_method(request)
        
        # 执行认证
        authenticator = self.SUPPORTED_METHODS[auth_method]()
        auth_context = await authenticator.authenticate(request)
        
        # 验证租户
        await self._validate_tenant(auth_context)
        
        # 刷新令牌（如需要）
        if auth_context.needs_refresh:
            auth_context = await self._refresh_token(auth_context)
            
        return auth_context

class JWTAuthenticator:
    """JWT认证实现"""
    
    def __init__(self):
        self.public_key = load_public_key()
        self.issuer = config.jwt_issuer
        self.audience = config.jwt_audience
        
    async def authenticate(self, request: Request) -> AuthContext:
        """JWT认证流程"""
        token = self._extract_token(request)
        
        try:
            # 验证签名和声明
            payload = jwt.verify(
                token,
                self.public_key,
                issuer=self.issuer,
                audience=self.audience,
                algorithms=["RS256"]
            )
            
            # 检查过期时间
            if payload["exp"] < time.time():
                raise AuthenticationError("Token expired")
                
            # 构建认证上下文
            return AuthContext(
                user_id=payload["sub"],
                tenant_id=payload.get("tenant_id"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", [])
            )
            
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
```

##### 12.3 授权与访问控制

```python
class RBACManager:
    """基于角色的访问控制"""
    
    # 预定义角色
    ROLES = {
        "admin": {
            "permissions": ["*"],  # 所有权限
            "resource_access": {"*": ["*"]}
        },
        "sre": {
            "permissions": [
                "troubleshoot.create",
                "troubleshoot.read",
                "troubleshoot.update",
                "evidence.read",
                "report.generate"
            ],
            "resource_access": {
                "logs": ["read"],
                "metrics": ["read"],
                "alerts": ["read"]
            }
        },
        "developer": {
            "permissions": [
                "troubleshoot.read",
                "evidence.read",
                "report.read"
            ],
            "resource_access": {
                "logs": ["read"],
                "metrics": ["read"]
            }
        },
        "viewer": {
            "permissions": [
                "troubleshoot.read",
                "report.read"
            ],
            "resource_access": {}
        }
    }
    
    async def authorize(self, 
                       auth_context: AuthContext, 
                       resource: str, 
                       action: str) -> bool:
        """检查授权"""
        # 检查用户权限
        if await self._check_user_permission(auth_context, resource, action):
            return True
            
        # 检查角色权限
        for role in auth_context.roles:
            if await self._check_role_permission(role, resource, action):
                return True
                
        # 检查资源级权限
        if await self._check_resource_permission(auth_context, resource, action):
            return True
            
        return False
        
    async def filter_resources(self, 
                             auth_context: AuthContext, 
                             resources: List[Resource]) -> List[Resource]:
        """根据权限过滤资源列表"""
        filtered = []
        for resource in resources:
            if await self.authorize(auth_context, resource.type, "read"):
                # 检查租户隔离
                if resource.tenant_id == auth_context.tenant_id:
                    filtered.append(resource)
                    
        return filtered
```

##### 12.4 数据安全与隐私

```python
class DataSecurityManager:
    """数据安全管理器"""
    
    def __init__(self):
        self.encryptor = AESEncryptor(key=load_encryption_key())
        self.masker = DataMasker()
        self.classifier = DataClassifier()
        
    async def process_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理敏感数据"""
        processed = {}
        
        for key, value in data.items():
            # 分类数据
            classification = self.classifier.classify(key, value)
            
            if classification == DataClass.SECRET:
                # 加密存储
                processed[key] = await self.encryptor.encrypt(value)
                processed[f"{key}_encrypted"] = True
                
            elif classification == DataClass.PII:
                # 脱敏处理
                processed[key] = self.masker.mask(value, MaskType.PARTIAL)
                
            elif classification == DataClass.INTERNAL:
                # 标记但不处理
                processed[key] = value
                processed[f"{key}_classification"] = "internal"
                
            else:
                processed[key] = value
                
        return processed

class DataMasker:
    """数据脱敏器"""
    
    MASKING_RULES = {
        "email": lambda x: x[:3] + "***@" + x.split("@")[1] if "@" in x else "***",
        "phone": lambda x: x[:3] + "****" + x[-2:] if len(x) > 5 else "***",
        "ip": lambda x: ".".join(x.split(".")[:2] + ["xxx", "xxx"]),
        "token": lambda x: x[:8] + "..." + x[-4:] if len(x) > 12 else "***",
        "password": lambda x: "*" * 8
    }
    
    def mask(self, value: str, mask_type: MaskType) -> str:
        """执行脱敏"""
        if mask_type == MaskType.FULL:
            return "***"
        elif mask_type == MaskType.PARTIAL:
            # 自动识别数据类型并应用规则
            for pattern, masker in self.MASKING_RULES.items():
                if self._matches_pattern(value, pattern):
                    return masker(value)
            # 默认部分脱敏
            return value[:3] + "***" if len(value) > 3 else "***"
        return value
```

##### 12.5 审计日志

```python
class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self):
        self.storage = AuditLogStorage()
        self.enricher = LogEnricher()
        
    async def log_event(self, event: AuditEvent):
        """记录审计事件"""
        # 丰富事件信息
        enriched_event = await self.enricher.enrich(event)
        
        # 添加安全哈希
        enriched_event.hash = self._calculate_hash(enriched_event)
        
        # 持久化存储
        await self.storage.store(enriched_event)
        
        # 实时告警（如需要）
        if enriched_event.severity >= AuditSeverity.HIGH:
            await self._send_alert(enriched_event)

@dataclass
class AuditEvent:
    """审计事件模型"""
    event_id: str
    timestamp: datetime
    event_type: str               # login, api_call, data_access, config_change
    actor: AuditActor            # 操作者信息
    resource: AuditResource      # 被访问资源
    action: str                  # create, read, update, delete
    result: str                  # success, failure, partial
    details: Dict[str, Any]      # 详细信息
    severity: AuditSeverity      # info, low, medium, high, critical
    
@dataclass
class AuditActor:
    """审计操作者"""
    user_id: str
    tenant_id: str
    ip_address: str
    user_agent: str
    auth_method: str
    roles: List[str]
```

##### 12.6 凭证管理

```python
class CredentialManager:
    """凭证安全管理"""
    
    def __init__(self):
        self.vault_client = VaultClient()
        self.rotation_scheduler = RotationScheduler()
        
    async def get_credential(self, 
                           credential_id: str, 
                           auth_context: AuthContext) -> Credential:
        """获取凭证"""
        # 检查访问权限
        if not await self._check_access(credential_id, auth_context):
            raise UnauthorizedError("Access denied to credential")
            
        # 从安全存储获取
        encrypted_cred = await self.vault_client.get(credential_id)
        
        # 解密
        credential = await self._decrypt_credential(encrypted_cred)
        
        # 记录访问
        await self._log_access(credential_id, auth_context)
        
        # 检查轮转需求
        if self._needs_rotation(credential):
            await self.rotation_scheduler.schedule(credential_id)
            
        return credential
        
    async def rotate_credential(self, credential_id: str):
        """轮转凭证"""
        # 生成新凭证
        new_credential = await self._generate_new_credential(credential_id)
        
        # 更新存储
        await self.vault_client.update(credential_id, new_credential)
        
        # 通知相关系统
        await self._notify_rotation(credential_id)
        
        # 保留旧凭证一段时间（优雅过渡）
        await self._archive_old_credential(credential_id)
```

##### 12.7 合规性控制

```python
class ComplianceManager:
    """合规性管理"""
    
    COMPLIANCE_RULES = {
        "gdpr": {
            "data_retention": 90,  # 天
            "require_consent": True,
            "right_to_deletion": True,
            "data_portability": True
        },
        "sox": {
            "audit_retention": 2555,  # 7年
            "access_review_period": 90,
            "separation_of_duties": True
        },
        "pci_dss": {
            "encryption_required": True,
            "access_logging": True,
            "vulnerability_scanning": True
        }
    }
    
    async def enforce_compliance(self, 
                               operation: str, 
                               data: Dict[str, Any], 
                               context: ComplianceContext) -> ComplianceResult:
        """执行合规性检查"""
        violations = []
        
        for framework in context.applicable_frameworks:
            rules = self.COMPLIANCE_RULES.get(framework, {})
            
            # 检查数据保留
            if "data_retention" in rules:
                if not await self._check_retention_policy(data, rules["data_retention"]):
                    violations.append(ComplianceViolation(
                        framework=framework,
                        rule="data_retention",
                        severity="high"
                    ))
                    
            # 检查加密要求
            if rules.get("encryption_required"):
                if not await self._verify_encryption(data):
                    violations.append(ComplianceViolation(
                        framework=framework,
                        rule="encryption_required",
                        severity="critical"
                    ))
                    
        return ComplianceResult(
            compliant=len(violations) == 0,
            violations=violations,
            recommendations=self._generate_recommendations(violations)
        )
```

##### 12.8 安全最佳实践

```yaml
security_best_practices:
  authentication:
    - 使用强加密算法（RSA-2048+、AES-256）
    - 实施多因素认证（MFA）
    - 定期轮转密钥和证书
    
  authorization:
    - 最小权限原则
    - 职责分离
    - 定期权限审查
    
  data_protection:
    - 传输加密（TLS 1.3+）
    - 静态加密
    - 密钥管理服务（KMS）
    
  monitoring:
    - 实时安全监控
    - 异常行为检测
    - 安全事件响应流程
    
  compliance:
    - 定期合规审计
    - 自动化合规检查
    - 合规报告生成
```

#### 13. 成本与性能

##### 13.1 成本模型

```python
@dataclass
class CostModel:
    """成本计算模型"""
    # LLM成本
    llm_costs: Dict[str, float] = field(default_factory=lambda: {
        "gpt-4": {"input": 0.03, "output": 0.06},      # per 1K tokens
        "gpt-3.5": {"input": 0.001, "output": 0.002},
        "claude-3": {"input": 0.015, "output": 0.075}
    })
    
    # 工具调用成本
    tool_costs: Dict[str, float] = field(default_factory=lambda: {
        "log.search": 0.001,      # per query
        "metric.query": 0.0005,
        "alert.list": 0.0002
    })
    
    # 存储成本
    storage_costs: Dict[str, float] = field(default_factory=lambda: {
        "sqlite": 0.0,           # 本地存储
        "s3": 0.023,            # per GB/month
        "redis": 0.016          # per GB-hour
    })

class CostCalculator:
    """成本计算器"""
    
    def calculate_request_cost(self, request: TroubleshootingRequest) -> Cost:
        """计算单次请求成本"""
        token_cost = self._calculate_token_cost(request)
        tool_cost = self._calculate_tool_cost(request)
        storage_cost = self._calculate_storage_cost(request)
        
        return Cost(
            total=token_cost + tool_cost + storage_cost,
            breakdown={
                "llm_tokens": token_cost,
                "tool_calls": tool_cost,
                "storage": storage_cost
            },
            estimated=request.status != "completed"
        )
```

##### 13.2 预算控制

```python
class BudgetController:
    """预算控制器"""
    
    def __init__(self):
        self.limits = {
            "per_request": 5.0,      # $5 per request
            "per_tenant_daily": 100.0,
            "per_tenant_monthly": 2000.0
        }
        self.usage_tracker = UsageTracker()
    
    async def check_budget(self, 
                         tenant_id: str, 
                         estimated_cost: float) -> BudgetStatus:
        """检查预算限制"""
        current_usage = await self.usage_tracker.get_usage(tenant_id)
        
        # 检查各级限制
        if estimated_cost > self.limits["per_request"]:
            return BudgetStatus(
                allowed=False,
                reason="Exceeds per-request limit",
                limit=self.limits["per_request"]
            )
        
        daily_total = current_usage.daily + estimated_cost
        if daily_total > self.limits["per_tenant_daily"]:
            return BudgetStatus(
                allowed=False,
                reason="Exceeds daily limit",
                remaining=self.limits["per_tenant_daily"] - current_usage.daily
            )
        
        return BudgetStatus(allowed=True)
    
    async def apply_cost_optimization(self, 
                                    request: TroubleshootingRequest,
                                    budget_remaining: float):
        """应用成本优化策略"""
        if budget_remaining < 1.0:
            # 低预算模式
            request.optimization_hints = {
                "use_cheaper_model": True,
                "limit_tool_calls": 10,
                "disable_parallel": True,
                "sample_data": 0.1  # 只采样10%数据
            }
        elif budget_remaining < 5.0:
            # 中等预算模式
            request.optimization_hints = {
                "prefer_cache": True,
                "limit_iterations": 3,
                "batch_size": "small"
            }
```

##### 13.3 性能优化策略

```yaml
performance_optimization:
  caching:
    strategies:
      - result_cache:
          ttl: 3600  # 1小时
          key_pattern: "{tenant}:{service}:{issue_type}:{time_window}"
          storage: redis
      
      - tool_response_cache:
          ttl: 300   # 5分钟
          key_pattern: "{tool}:{query_hash}"
          storage: memory
      
      - llm_response_cache:
          ttl: 86400  # 24小时
          key_pattern: "{model}:{prompt_hash}"
          conditions:
            - deterministic_prompts_only
            - no_realtime_data
  
  concurrency:
    agent_pool:
      min_workers: 2
      max_workers: 10
      scale_factor: 0.8  # CPU使用率触发扩容
    
    task_queue:
      type: priority_queue
      max_size: 1000
      timeout: 300
    
    rate_limiting:
      per_agent: 5    # 每个Agent并发数
      per_tenant: 20  # 每个租户并发数
  
  batching:
    log_queries:
      batch_size: 100
      wait_time: 100ms
    
    metric_queries:
      batch_size: 50
      wait_time: 50ms
```

##### 13.4 响应时间优化

```python
class PerformanceOptimizer:
    """性能优化器"""
    
    async def optimize_execution_plan(self, 
                                    plan: InvestigationPlan) -> OptimizedPlan:
        """优化执行计划"""
        # 任务依赖分析
        dependency_graph = self._build_dependency_graph(plan.tasks)
        
        # 关键路径分析
        critical_path = self._find_critical_path(dependency_graph)
        
        # 并行化优化
        parallel_groups = self._optimize_parallelization(
            tasks=plan.tasks,
            dependencies=dependency_graph,
            max_parallel=5
        )
        
        # 任务合并
        merged_tasks = self._merge_similar_tasks(plan.tasks)
        
        return OptimizedPlan(
            tasks=merged_tasks,
            execution_groups=parallel_groups,
            estimated_duration=self._estimate_duration(critical_path)
        )
    
    def _merge_similar_tasks(self, tasks: List[Task]) -> List[Task]:
        """合并相似任务减少调用次数"""
        merged = []
        
        # 按类型和目标分组
        groups = defaultdict(list)
        for task in tasks:
            key = (task.type, task.inputs.get("service"))
            groups[key].append(task)
        
        # 合并同组任务
        for (task_type, service), group_tasks in groups.items():
            if len(group_tasks) > 1 and task_type in ["log", "metric"]:
                # 合并查询条件
                merged_task = Task(
                    type=task_type,
                    inputs={
                        "service": service,
                        "queries": [t.inputs for t in group_tasks]
                    }
                )
                merged.append(merged_task)
            else:
                merged.extend(group_tasks)
        
        return merged
```

##### 13.5 资源使用监控

```python
class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": GaugeMetric("cpu_usage_percent"),
            "memory_usage": GaugeMetric("memory_usage_bytes"),
            "token_usage": CounterMetric("llm_tokens_total"),
            "api_calls": CounterMetric("api_calls_total"),
            "cache_hit_rate": GaugeMetric("cache_hit_rate")
        }
    
    async def collect_metrics(self) -> ResourceMetrics:
        """收集资源指标"""
        return ResourceMetrics(
            cpu=psutil.cpu_percent(interval=1),
            memory=psutil.virtual_memory().used,
            active_requests=await self._count_active_requests(),
            queue_depth=await self._get_queue_depth(),
            cache_stats=await self._get_cache_stats()
        )
    
    async def auto_scale_decision(self, metrics: ResourceMetrics) -> ScaleAction:
        """自动扩缩容决策"""
        if metrics.cpu > 80 or metrics.queue_depth > 100:
            return ScaleAction.SCALE_UP
        elif metrics.cpu < 20 and metrics.active_requests < 5:
            return ScaleAction.SCALE_DOWN
        return ScaleAction.NO_CHANGE
```

##### 13.6 输出优化

```python
class OutputOptimizer:
    """输出优化器"""
    
    def compress_evidence(self, evidence: List[Evidence]) -> List[Evidence]:
        """压缩证据数据"""
        compressed = []
        
        for e in evidence:
            # 移除冗余字段
            compressed_e = Evidence(
                evidence_id=e.evidence_id,
                source=e.source,
                summary=e.summary,
                confidence=e.confidence,
                # 只保留引用，不保留原始数据
                raw_ref={"url": e.raw_ref.get("url")} if e.raw_ref else None
            )
            compressed.append(compressed_e)
        
        return compressed
    
    def summarize_for_response(self, 
                             result: TroubleshootingResult,
                             detail_level: str = "summary") -> Dict:
        """根据详细程度生成响应"""
        if detail_level == "minimal":
            return {
                "status": result.status,
                "root_cause": result.root_cause.hypothesis if result.root_cause else None,
                "remediation": result.remediation.actions[0] if result.remediation else None
            }
        elif detail_level == "summary":
            return {
                "status": result.status,
                "root_cause": result.root_cause,
                "remediation": result.remediation,
                "evidence_count": len(result.evidence),
                "confidence": result.root_cause.confidence if result.root_cause else 0
            }
        else:  # full
            return result.to_dict()
```

##### 13.7 成本优化最佳实践

```yaml
cost_optimization_practices:
  llm_optimization:
    - 使用更便宜的模型进行初步分析
    - 仅在需要时升级到高级模型
    - 实施智能提示词压缩
    - 缓存常见问题的分析结果
  
  data_optimization:
    - 实施智能采样策略
    - 使用增量查询而非全量查询
    - 压缩和聚合历史数据
    - 设置合理的数据保留期限
  
  execution_optimization:
    - 优先使用缓存结果
    - 批量处理相似请求
    - 实施请求去重机制
    - 使用异步处理减少等待
  
  storage_optimization:
    - 定期清理过期数据
    - 使用分层存储策略
    - 压缩存储大型对象
    - 实施数据归档策略
```

##### 13.8 性能基准

```yaml
performance_benchmarks:
  response_time:
    p50: 5s
    p90: 15s
    p99: 30s
    max: 60s
  
  throughput:
    requests_per_second: 10
    concurrent_requests: 50
  
  resource_usage:
    cpu_per_request: 0.5 cores
    memory_per_request: 512MB
    
  cost_targets:
    average_per_request: $0.50
    p90_per_request: $2.00
    monthly_per_tenant: $500
```

#### 14. 观测与可视化
- OpenTelemetry：为节点与工具建立 span，属性含 tenant、request_id、agent、tool；
- 运行事件总线：记录进入/退出节点、条件分支选择、错误、重试；
- 可视化：LangSmith 或自建可视化，展示计划、分支与证据链路。

#### 15. 对外接口（API/CLI）

##### 15.1 RESTful API 规范

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
        '401':
          $ref: '#/components/responses/Unauthorized'
    
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
  
  /feedback:
    post:
      summary: 提交反馈
      operationId: submitFeedback
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Feedback'
      responses:
        '201':
          description: 反馈已记录

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

##### 15.2 gRPC API 定义

```protobuf
syntax = "proto3";

package devops.agent.v1;

service TroubleshootingService {
  // 创建排障请求
  rpc CreateTroubleshootingRequest(CreateRequestRequest) returns (CreateRequestResponse);
  
  // 获取排障状态（流式）
  rpc StreamTroubleshootingStatus(StreamStatusRequest) returns (stream StatusUpdate);
  
  // 获取排障结果
  rpc GetTroubleshootingResult(GetResultRequest) returns (TroubleshootingResult);
  
  // 提交反馈
  rpc SubmitFeedback(FeedbackRequest) returns (FeedbackResponse);
}

message CreateRequestRequest {
  string title = 1;
  string description = 2;
  string service = 3;
  string environment = 4;
  Severity severity = 5;
  TimeRange time_range = 6;
  map<string, string> artifacts_hints = 7;
}

message StatusUpdate {
  string request_id = 1;
  Status status = 2;
  string message = 3;
  repeated Evidence new_evidence = 4;
  Finding new_finding = 5;
  google.protobuf.Timestamp timestamp = 6;
}

enum Status {
  STATUS_UNSPECIFIED = 0;
  STATUS_QUEUED = 1;
  STATUS_RUNNING = 2;
  STATUS_COMPLETED = 3;
  STATUS_FAILED = 4;
}
```

##### 15.3 WebSocket API

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

##### 15.4 CLI 接口

```bash
# 基本用法
devops-agent troubleshoot [options]

# 选项
Options:
  -s, --service <name>        目标服务名称 (必需)
  -t, --title <title>         问题标题
  -d, --description <desc>    问题描述
  -e, --env <environment>     环境 [prod|staging|dev] (默认: prod)
  --severity <level>          严重程度 [critical|high|medium|low]
  --since <duration>          时间范围 (如: 30m, 1h, 2d)
  --from <datetime>           开始时间 (ISO 8601)
  --to <datetime>             结束时间 (ISO 8601)
  -o, --output <format>       输出格式 [json|yaml|table|markdown]
  --async                     异步执行
  --follow                    实时跟踪进度
  -v, --verbose              详细输出

# 示例
# 基本排障
devops-agent troubleshoot -s checkout -t "高延迟问题" --since 30m

# 指定时间范围
devops-agent troubleshoot -s payment \
  --from "2024-01-01T10:00:00Z" \
  --to "2024-01-01T11:00:00Z" \
  --severity high

# 异步执行并跟踪
devops-agent troubleshoot -s api-gateway \
  -t "5xx错误激增" \
  --async --follow \
  -o json

# 查看历史记录
devops-agent list --status completed --limit 10

# 获取详情
devops-agent get req_123456 --format markdown

# 导出报告
devops-agent report req_123456 --format pdf --output report.pdf
```

##### 15.5 SDK 示例

```python
# Python SDK
from devops_agent import DevOpsAgentClient

client = DevOpsAgentClient(
    api_key="your-api-key",
    base_url="https://api.devops-agent.com"
)

# 创建排障请求
response = client.troubleshoot.create(
    title="数据库连接超时",
    service="user-service",
    environment="prod",
    severity="high",
    time_range={"from": "2024-01-01T10:00:00Z", "to": "now"}
)

# 等待完成
result = client.troubleshoot.wait_for_completion(
    request_id=response.id,
    timeout=300
)

# 获取根因和建议
print(f"根因: {result.root_cause.hypothesis}")
print(f"建议: {result.remediation.actions}")
```

```javascript
// JavaScript/TypeScript SDK
import { DevOpsAgentClient } from '@devops-agent/sdk';

const client = new DevOpsAgentClient({
  apiKey: process.env.DEVOPS_AGENT_API_KEY,
  baseUrl: 'https://api.devops-agent.com'
});

// 创建排障请求
const request = await client.troubleshoot.create({
  title: '支付服务响应慢',
  service: 'payment-service',
  environment: 'prod',
  severity: 'high',
  timeRange: {
    from: new Date(Date.now() - 30 * 60 * 1000),
    to: new Date()
  }
});

// 监听实时更新
client.troubleshoot.subscribe(request.id, {
  onStatusUpdate: (status) => console.log('Status:', status),
  onEvidenceCollected: (evidence) => console.log('Evidence:', evidence),
  onComplete: (result) => {
    console.log('Root cause:', result.rootCause);
    console.log('Remediation:', result.remediation);
  }
});
```

##### 15.6 API 认证与限流

```yaml
authentication:
  methods:
    - api_key:
        header: X-API-Key
        query_param: api_key
    - bearer_token:
        header: Authorization
        format: "Bearer {token}"
    - oauth2:
        flow: authorization_code
        auth_url: https://auth.devops-agent.com/authorize
        token_url: https://auth.devops-agent.com/token

rate_limiting:
  default:
    requests_per_minute: 60
    requests_per_hour: 1000
    concurrent_requests: 10
  
  by_tier:
    free:
      requests_per_minute: 10
      requests_per_hour: 100
      concurrent_requests: 2
    pro:
      requests_per_minute: 100
      requests_per_hour: 5000
      concurrent_requests: 20
    enterprise:
      requests_per_minute: 1000
      requests_per_hour: 50000
      concurrent_requests: 100

headers:
  rate_limit:
    - X-RateLimit-Limit
    - X-RateLimit-Remaining
    - X-RateLimit-Reset
```

##### 15.7 错误响应规范

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is missing required field: service",
    "details": {
      "field": "service",
      "reason": "required"
    },
    "request_id": "req_abc123",
    "timestamp": "2024-01-01T10:00:00Z"
  }
}
```

错误代码：
- `INVALID_REQUEST`: 请求参数无效
- `UNAUTHORIZED`: 未授权
- `FORBIDDEN`: 无权限
- `NOT_FOUND`: 资源不存在
- `RATE_LIMITED`: 超过限流
- `INTERNAL_ERROR`: 内部错误
- `SERVICE_UNAVAILABLE`: 服务不可用

#### 16. 测试策略

##### 16.1 测试架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     测试金字塔                                   │
│                                                                 │
│     ┌─────────────────────────────────────────┐               │
│     │        端到端测试 (E2E)                  │  5%          │
│     │   - 完整排障流程                         │               │
│     │   - 多租户场景                           │               │
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

##### 16.2 单元测试策略

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

##### 16.3 组件测试策略

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

##### 16.4 集成测试策略

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

##### 16.5 端到端测试策略

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

##### 16.6 性能测试策略

```yaml
performance_test_scenarios:
  concurrent_requests:
    users: 50
    requests_per_user: 10
    expected_rps: 10
    expected_p99_latency: 5s
  
  sustained_load:
    duration: 300s
    target_rps: 20
    success_rate_threshold: 0.95
  
  memory_usage:
    requests: 100
    max_memory_increase: 500MB
```

##### 16.7 测试数据管理

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

##### 16.8 测试环境配置

```yaml
test_environments:
  unit:
    mock_llm: true
    mock_mcp: true
    database: ":memory:"
  
  integration:
    mock_llm: true
    mock_mcp: false
    database: "test.db"
  
  e2e:
    mock_llm: false
    api_endpoint: "http://localhost:8080"
```

##### 16.9 测试执行与报告

```bash
# 执行测试命令
make test-unit        # 单元测试
make test-integration # 集成测试
make test-e2e        # 端到端测试
make test-performance # 性能测试
make test-all        # 所有测试

# 覆盖率要求
- 单元测试覆盖率: > 90%
- 整体覆盖率: > 80%
- 关键路径覆盖率: 100%
```

##### 16.10 持续集成配置

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Unit Tests
        run: make test-unit
      - name: Run Integration Tests
        run: make test-integration
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
```

#### 17. 部署与运行

##### 17.1 部署架构

```
┌────────────────────────────────────────────────────────────────┐
│                        Load Balancer                            │
│                    (AWS ALB / Nginx / HAProxy)                  │
└───────────────┬──────────────────────┬─────────────────────────┘
                │                      │
    ┌───────────┴──────────┐  ┌───────┴───────────┐
    │   API Gateway (1)    │  │  API Gateway (2)  │  ... (N)
    │  - Rate Limiting     │  │  - Rate Limiting  │
    │  - Auth/Authz        │  │  - Auth/Authz     │
    └───────────┬──────────┘  └───────┬───────────┘
                │                      │
┌───────────────┴──────────────────────┴─────────────────────────┐
│                    Kubernetes Cluster                           │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Control Plane                         │  │
│  │  ┌─────────┐  ┌──────────┐  ┌────────────────────┐    │  │
│  │  │   HPA   │  │   VPA    │  │  Service Mesh      │    │  │
│  │  │         │  │          │  │  (Istio/Linkerd)   │    │  │
│  │  └─────────┘  └──────────┘  └────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Application Pods                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │  │
│  │  │ Orchestrator│  │Agent Workers│  │  API Service  │  │  │
│  │  │   (3 pods)  │  │  (5 pods)   │  │   (3 pods)    │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Data Layer                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │  │
│  │  │   SQLite    │  │   Redis     │  │  Object Store │  │  │
│  │  │  (PVC)      │  │  Cluster    │  │   (S3/Minio)  │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

##### 17.2 容器化设计

```dockerfile
# Dockerfile.base - 基础镜像
FROM python:3.11-slim as base

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dockerfile.app - 应用镜像
FROM base as app

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/

# 设置环境变量
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# 运行应用
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

##### 17.3 Kubernetes 资源配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: devops-agent-orchestrator
  labels:
    app: devops-agent
    component: orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: devops-agent
      component: orchestrator
  template:
    metadata:
      labels:
        app: devops-agent
        component: orchestrator
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: devops-agent
      containers:
      - name: orchestrator
        image: devops-agent:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: ENVIRONMENT
          value: production
        - name: LOG_LEVEL
          value: INFO
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: devops-agent-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: sqlite-storage
          mountPath: /data
      volumes:
      - name: config
        configMap:
          name: devops-agent-config
      - name: sqlite-storage
        persistentVolumeClaim:
          claimName: sqlite-pvc
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: component
                  operator: In
                  values:
                  - orchestrator
              topologyKey: kubernetes.io/hostname
```

##### 17.4 配置管理

```yaml
# ConfigMap - 应用配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: devops-agent-config
data:
  app.yaml: |
    server:
      host: 0.0.0.0
      port: 8080
      workers: 4
    
    langgraph:
      max_concurrent_agents: 5
      default_timeout: 300
      checkpoint_interval: 60
    
    mcp:
      servers:
        - id: logs
          endpoint: http://mcp-logs-server:8080
          timeout: 30
        - id: metrics
          endpoint: http://mcp-metrics-server:8080
          timeout: 30
        - id: alerts
          endpoint: http://mcp-alerts-server:8080
          timeout: 30
    
    observability:
      otel_endpoint: http://otel-collector:4317
      metrics_port: 9090
      trace_sample_rate: 0.1
    
    security:
      jwt_issuer: https://auth.example.com
      jwt_audience: devops-agent
      encryption_key_path: /secrets/encryption.key

# Secret - 敏感配置
apiVersion: v1
kind: Secret
metadata:
  name: devops-agent-secrets
type: Opaque
stringData:
  database-url: "sqlite:///data/devops_agent.db"
  encryption-key: "base64-encoded-key"
  mcp-tokens: |
    logs: "token-for-logs-server"
    metrics: "token-for-metrics-server"
    alerts: "token-for-alerts-server"
```

##### 17.5 自动扩缩容策略

```yaml
# HPA - 水平自动扩缩容
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: devops-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: devops-agent-orchestrator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: pending_tasks
      target:
        type: AverageValue
        averageValue: "30"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60

# VPA - 垂直自动扩缩容
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: devops-agent-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: devops-agent-orchestrator
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: orchestrator
      minAllowed:
        cpu: 250m
        memory: 256Mi
      maxAllowed:
        cpu: 4
        memory: 8Gi
```

##### 17.6 监控与告警

```yaml
# Prometheus 规则
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: devops-agent-rules
spec:
  groups:
  - name: devops-agent
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: |
        rate(devops_agent_errors_total[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} errors per second"
    
    - alert: HighLatency
      expr: |
        histogram_quantile(0.99, rate(devops_agent_request_duration_seconds_bucket[5m])) > 30
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High latency detected"
        description: "p99 latency is {{ $value }} seconds"
    
    - alert: PodMemoryUsage
      expr: |
        container_memory_usage_bytes{pod=~"devops-agent-.*"} / 
        container_spec_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage"
        description: "Pod {{ $labels.pod }} memory usage is above 90%"
```

##### 17.7 灾难恢复

```yaml
# 备份策略
backup_strategy:
  schedule:
    full_backup: "0 2 * * 0"  # 每周日凌晨2点
    incremental: "0 * * * *"   # 每小时
  
  retention:
    full_backup: 30            # 保留30天
    incremental: 168           # 保留7天
  
  targets:
    - type: sqlite
      path: /data/devops_agent.db
      destination: s3://backup-bucket/sqlite/
    
    - type: config
      path: /app/config/
      destination: s3://backup-bucket/config/
    
    - type: audit_logs
      path: /data/audit/
      destination: s3://backup-bucket/audit/

# 恢复流程
recovery_procedure:
  1_assess:
    - 评估故障范围
    - 确定恢复点目标(RPO)
    - 确定恢复时间目标(RTO)
  
  2_prepare:
    - 准备恢复环境
    - 验证备份完整性
    - 通知相关团队
  
  3_restore:
    - 恢复数据库
    - 恢复配置文件
    - 恢复审计日志
  
  4_verify:
    - 验证数据完整性
    - 运行健康检查
    - 执行功能测试
  
  5_cutover:
    - 逐步切换流量
    - 监控系统状态
    - 确认恢复成功
```

##### 17.8 运维操作手册

```yaml
operational_runbook:
  deployment:
    - 构建新镜像: make build VERSION=x.y.z
    - 推送镜像: make push VERSION=x.y.z
    - 部署到staging: kubectl apply -k overlays/staging
    - 运行冒烟测试: make test-staging
    - 部署到生产: kubectl apply -k overlays/production
    - 验证部署: kubectl rollout status deployment/devops-agent
  
  troubleshooting:
    high_memory:
      symptoms: Pod内存使用超过80%
      diagnosis:
        - kubectl top pods -l app=devops-agent
        - kubectl logs <pod> | grep "memory"
        - 检查是否有内存泄漏
      resolution:
        - 重启受影响的Pod
        - 调整内存限制
        - 优化代码减少内存使用
    
    slow_response:
      symptoms: API响应时间超过5秒
      diagnosis:
        - 检查CPU使用率
        - 分析慢查询日志
        - 检查外部依赖延迟
      resolution:
        - 扩容Pod数量
        - 优化数据库查询
        - 启用缓存
  
  maintenance:
    rolling_update:
      - 设置PodDisruptionBudget
      - 执行滚动更新
      - 监控错误率
      - 必要时回滚
    
    database_maintenance:
      - 创建维护窗口
      - 执行VACUUM操作
      - 重建索引
      - 验证性能改善
```

##### 17.9 多租户隔离

```python
class TenantIsolation:
    """租户隔离实现"""
    
    def __init__(self):
        self.namespace_template = "devops-agent-{tenant_id}"
        self.resource_quotas = {
            "small": {"cpu": "2", "memory": "4Gi", "storage": "10Gi"},
            "medium": {"cpu": "8", "memory": "16Gi", "storage": "50Gi"},
            "large": {"cpu": "32", "memory": "64Gi", "storage": "200Gi"}
        }
    
    async def provision_tenant(self, tenant: Tenant):
        """为新租户配置资源"""
        # 创建命名空间
        namespace = self.namespace_template.format(tenant_id=tenant.id)
        await self.create_namespace(namespace)
        
        # 设置资源配额
        quota = self.resource_quotas[tenant.tier]
        await self.apply_resource_quota(namespace, quota)
        
        # 配置网络策略
        await self.apply_network_policy(namespace)
        
        # 创建专用数据库
        await self.provision_database(tenant)
        
        # 配置RBAC
        await self.setup_rbac(tenant)
```

#### 18. 里程碑规划
- v0（原型）：单租户、最小工具集（Elastic/Prometheus/Alertmanager）、同步 API、基本可观测；
- v1（生产）：异步任务、并行分支、LangSmith 集成、基础缓存；
- v2（扩展）：安全与合规、成本与性能、部署与运行、风险控制、RBAC、审计、APM/Tracing、CMDB/变更、知识库、自适应提示词、自动化修复建议模板库。

#### 19. 风险与缓解

##### 19.1 技术风险

```yaml
technical_risks:
  llm_risks:
    - risk: LLM幻觉导致错误诊断
      probability: 中
      impact: 高
      mitigation:
        - 实施多重验证机制
        - 要求提供证据支撑
        - 设置置信度阈值
        - 人工审核高风险建议
    
    - risk: 模型性能退化
      probability: 低
      impact: 中
      mitigation:
        - 定期评估模型性能
        - A/B测试新模型版本
        - 保留回退选项
        - 建立性能基线
    
    - risk: API限流或服务中断
      probability: 中
      impact: 高
      mitigation:
        - 多模型供应商备份
        - 本地缓存机制
        - 优雅降级策略
        - 断路器保护
  
  data_risks:
    - risk: 敏感数据泄露
      probability: 低
      impact: 极高
      mitigation:
        - 端到端加密
        - 数据脱敏处理
        - 访问控制审计
        - 定期安全扫描
    
    - risk: 数据质量问题
      probability: 中
      impact: 中
      mitigation:
        - 数据验证规则
        - 异常检测机制
        - 数据清洗流程
        - 质量监控指标
  
  system_risks:
    - risk: 级联故障
      probability: 低
      impact: 高
      mitigation:
        - 服务隔离设计
        - 熔断器机制
        - 限流保护
        - 故障演练
    
    - risk: 资源耗尽
      probability: 中
      impact: 中
      mitigation:
        - 资源配额管理
        - 自动扩缩容
        - 预警机制
        - 容量规划
```

##### 19.2 业务风险

```yaml
business_risks:
  adoption_risks:
    - risk: 用户不信任AI建议
      probability: 高
      impact: 中
      mitigation:
        - 透明的决策过程
        - 可解释性设计
        - 逐步建立信任
        - 成功案例展示
    
    - risk: 误操作导致生产事故
      probability: 低
      impact: 极高
      mitigation:
        - 只读模式默认
        - 人工确认机制
        - 操作审计日志
        - 回滚能力
  
  cost_risks:
    - risk: 成本失控
      probability: 中
      impact: 高
      mitigation:
        - 预算控制机制
        - 成本监控告警
        - 优化策略实施
        - 分层定价模型
    
    - risk: ROI不达预期
      probability: 中
      impact: 中
      mitigation:
        - 明确成功指标
        - 持续效果评估
        - 快速迭代优化
        - 价值度量体系
```

##### 19.3 合规风险

```python
class ComplianceRiskManager:
    """合规风险管理器"""
    
    def assess_compliance_risks(self, operation: Operation) -> RiskAssessment:
        """评估合规风险"""
        risks = []
        
        # GDPR合规风险
        if operation.involves_personal_data:
            risks.append(ComplianceRisk(
                type="GDPR",
                level="high" if not operation.has_consent else "low",
                requirements=[
                    "数据最小化",
                    "用户同意",
                    "数据可删除",
                    "数据可导出"
                ]
            ))
        
        # 行业特定合规
        if operation.industry == "financial":
            risks.append(ComplianceRisk(
                type="SOX",
                level="high",
                requirements=[
                    "审计追踪",
                    "访问控制",
                    "数据完整性",
                    "职责分离"
                ]
            ))
        
        return RiskAssessment(
            risks=risks,
            overall_level=max(r.level for r in risks),
            recommendations=self._generate_recommendations(risks)
        )
```

##### 19.4 安全风险矩阵

```python
class SecurityRiskMatrix:
    """安全风险矩阵"""
    
    RISK_MATRIX = {
        "injection_attacks": {
            "prompt_injection": {
                "likelihood": 3,  # 1-5
                "impact": 4,
                "controls": [
                    "输入验证",
                    "提示词隔离",
                    "输出过滤"
                ]
            },
            "sql_injection": {
                "likelihood": 2,
                "impact": 5,
                "controls": [
                    "参数化查询",
                    "ORM使用",
                    "输入清理"
                ]
            }
        },
        "access_control": {
            "privilege_escalation": {
                "likelihood": 2,
                "impact": 5,
                "controls": [
                    "最小权限原则",
                    "定期权限审查",
                    "多因素认证"
                ]
            },
            "unauthorized_access": {
                "likelihood": 3,
                "impact": 4,
                "controls": [
                    "强认证机制",
                    "会话管理",
                    "访问日志"
                ]
            }
        }
    }
    
    def calculate_risk_score(self, threat: str) -> int:
        """计算风险分数"""
        for category, threats in self.RISK_MATRIX.items():
            if threat in threats:
                t = threats[threat]
                return t["likelihood"] * t["impact"]
        return 0
```

##### 19.5 风险监控与响应

```python
class RiskMonitor:
    """风险监控器"""
    
    def __init__(self):
        self.risk_thresholds = {
            "error_rate": 0.05,      # 5%错误率
            "latency_p99": 30,       # 30秒
            "cost_per_hour": 100,    # $100/小时
            "security_events": 10     # 10个安全事件/小时
        }
        self.alert_channels = ["email", "slack", "pagerduty"]
    
    async def monitor_risks(self):
        """持续监控风险指标"""
        while True:
            metrics = await self.collect_metrics()
            
            for metric, value in metrics.items():
                if metric in self.risk_thresholds:
                    if value > self.risk_thresholds[metric]:
                        await self.trigger_alert(
                            RiskAlert(
                                type=metric,
                                severity=self._calculate_severity(metric, value),
                                message=f"{metric} exceeded threshold: {value}",
                                recommended_actions=self._get_actions(metric)
                            )
                        )
            
            await asyncio.sleep(60)  # 每分钟检查
    
    async def incident_response(self, incident: Incident):
        """事件响应流程"""
        # 1. 评估影响
        impact = await self.assess_impact(incident)
        
        # 2. 遏制威胁
        if impact.severity >= Severity.HIGH:
            await self.contain_threat(incident)
        
        # 3. 通知相关方
        await self.notify_stakeholders(incident, impact)
        
        # 4. 启动恢复
        recovery_plan = await self.create_recovery_plan(incident)
        await self.execute_recovery(recovery_plan)
        
        # 5. 事后分析
        await self.post_incident_analysis(incident)
```

##### 19.6 风险缓解策略

```yaml
mitigation_strategies:
  preventive:
    - 安全开发生命周期(SDLC)
    - 代码审查和静态分析
    - 依赖项漏洞扫描
    - 安全配置基线
    - 定期安全培训
  
  detective:
    - 实时监控和告警
    - 异常行为检测
    - 日志分析和SIEM
    - 定期安全审计
    - 渗透测试
  
  corrective:
    - 事件响应计划
    - 自动化修复脚本
    - 回滚机制
    - 备份和恢复
    - 补丁管理
  
  adaptive:
    - 威胁情报集成
    - 机器学习异常检测
    - 自适应访问控制
    - 动态风险评分
    - 持续改进流程
```

##### 19.7 业务连续性计划

```yaml
business_continuity:
  rto_rpo:
    recovery_time_objective: 4h
    recovery_point_objective: 1h
  
  failure_scenarios:
    - scenario: 主数据中心故障
      response:
        - 切换到备用数据中心
        - DNS故障转移
        - 通知用户服务降级
      
    - scenario: LLM服务不可用
      response:
        - 切换到备用模型
        - 启用缓存模式
        - 降级到规则引擎
    
    - scenario: 大规模数据泄露
      response:
        - 立即隔离受影响系统
        - 重置所有凭证
        - 通知监管机构
        - 启动法律响应
  
  dr_testing:
    frequency: quarterly
    scenarios:
      - 数据中心切换
      - 备份恢复
      - 服务降级
    success_criteria:
      - RTO达成率 > 90%
      - 数据完整性 > 99.9%
      - 用户影响 < 10%
```

##### 19.8 风险报告模板

```python
@dataclass
class RiskReport:
    """风险报告"""
    report_id: str
    period: DateRange
    executive_summary: str
    
    identified_risks: List[Risk]
    materialized_incidents: List[Incident]
    
    risk_metrics: Dict[str, float]
    trending: Dict[str, TrendDirection]
    
    mitigation_effectiveness: Dict[str, float]
    recommendations: List[str]
    
    compliance_status: ComplianceStatus
    audit_findings: List[Finding]
    
    def generate_dashboard(self) -> Dashboard:
        """生成风险仪表板"""
        return Dashboard(
            risk_heatmap=self._create_heatmap(),
            trend_charts=self._create_trends(),
            incident_timeline=self._create_timeline(),
            action_items=self._prioritize_actions()
        )
```

#### 20. 参考目录结构（建议）
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
│  ├─ services/             # 缓存、重试/节流（v2: 安全/RBAC/审计/成本）
│  └─ observability/        # OTel 集成
└─ config/
   ├─ default.yaml
   └─ sql/
      └─ ddl.sql            # SQLite DDL（见下文）
```

#### 21. 示例请求与输出（简）
- 请求：`支付服务 p99 延迟在 10:00-10:30 升高`。
- Planner：生成查询日志（checkout pod）、指标（http_server_request_duration_seconds{service="checkout"}）与关联最近变更的任务。
- 证据：日志中 5xx 激增、指标 p99 抬升、10:05 有配置变更；
- 总结：疑似配置回滚缺失导致连接池参数过小，建议扩大连接池并回滚到稳定版本；提供验证步骤与回滚指令草案。

—— 本文档可直接指导开发落地，配合 `src/graph/builder.py` 实现与工具封装逐步交付。

#### 22. SQLite DDL（v0）
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

-- LangGraph 检查点（可选）
CREATE TABLE IF NOT EXISTS graph_checkpoints (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  node TEXT,
  state_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id)
);
```

#### 23. MCP 配置样例（v0）
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

#### 24. API 契约（v0）
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

#### 25. Prompt 规范（v0）
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

#### 26. 最小运行示例（v0）
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

#### 27. v0 实施清单
- 初始化 SQLite（执行本节 DDL），实现 `src/services/repo.py` 读写。
- 实现 LangGraph 节点：planner/route/log_agent/alarm_agent/kpi_agent/summarize。
- 集成 MCP 客户端与工具桥接 `src/tools/mcp_bridge.py`。
- 打通 REST API：POST/GET /troubleshoot，绑定图执行与入库。
- 填充 `docs/prompts/*.md` 的具体模板，确保输出 JSON 符合第25节。
- 增加 2-3 个合成用例数据，跑通端到端。

