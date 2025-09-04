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
  - 计划生成、执行编排、并行/串行排查、证据收集、根因与建议生成、可追踪性与审计、成本/配额控制、最小化变更集成（只读为主，可选读写）。
  - 数据源：日志、告警、指标（必选），可扩展 APM/Tracing、变更/发布系统、CMDB、知识库。
- 非目标：
  - 自动执行破坏性操作（如直接回滚/扩容）默认禁用，仅输出建议与命令草案，人工确认后执行。

#### 4. 总体架构
- **编排核心**: LangGraph 状态机 + 多 Agent 节点。
- **智能体**:
  - Planner Agent: 根据问题生成排障计划，动态路由调用其他智能体。
  - Log Agent: 面向日志平台（Elastic/Loki/Splunk 等）查询、相关性分析、异常模式挖掘。
  - Alarm Agent: 面向告警平台（Prometheus Alertmanager/PagerDuty 等）聚合、去噪、时间相关分析。
  - KPI Agent: 面向时序数据库（Prometheus/Thanos/Influx 等）进行指标趋势/异常检测/因果线索分析。
  - Summary Agent: 聚合执行结果，推演根因、影响面与修复建议，生成报告与工单草案。
- **支撑能力**:
  - 工具层：日志/指标/告警/变更/CMDB/知识库连接器与统一工具抽象。
  - 存储：
    - 会话/状态存储（Redis/SQLite/Postgres）
    - 证据与中间结果（Postgres/Blob）
    - 审计日志（Append-only，OLTP/OLAP 皆可）
  - 观测与审计：OpenTelemetry Trace/Log/Metrics；结构化运行事件与工具调用审计。
  - 安全：KMS/Secret 管理、RBAC、租户与数据域隔离、最小权限。

#### 5. LangGraph 运行时与状态模型
- **GraphState（单次排障的共享状态）**：
  - request: TroubleshootingRequest
  - plan: InvestigationPlan
  - tasks: InvestigationTask[]
  - evidence: Evidence[]
  - findings: Finding[]
  - root_cause: RootCause | null
  - remediation: Remediation | null
  - next_actions: string[]  // Planner 的下一步建议/分派
  - history: Message[]      // 对话/Agent 交流上下文
  - cost_usage: TokenCost   // token/时间/工具调用用量
  - errors: AgentError[]
  - done: boolean

- **节点与边**：
  - Nodes: planner, route, log_agent, alarm_agent, kpi_agent, summarize, guard, finalize
  - Edges（动态条件）:
    - planner -> route
    - route -> {log_agent | alarm_agent | kpi_agent | summarize}
    - log_agent/alarm_agent/kpi_agent -> planner（反馈证据后迭代计划）
    - planner -> summarize（当满足终止条件或足够证据时）
    - summarize -> finalize

- **并发与节流**：
  - route 节点并行触发多个 Agent（受并发上限/配额控制），聚合结果后回到 planner。
  - 对高成本工具调用设定速率限制与重试间隔。

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
  - 输入：问题描述、上下文、当前证据/发现、历史任务与结果、成本与时间预算。
  - 输出：更新后的 InvestigationPlan 与下一步 `next_actions`。
- Log/Alarm/KPI 提示：
  - 输入：任务参数（指标名/日志索引/服务/时间窗）+ 查询建议与样例。
  - 输出：结构化证据（摘要+引用），避免长原文粘贴，提供查询复现信息。
- Summary 提示：
  - 输入：完整计划与证据集合。
  - 输出：`root_cause` + `remediation` + 验证与回滚步骤 + 工单摘要。

#### 8. 工具与连接器抽象
- 接口定义（Python 伪代码）：
```python
class LogTool(Protocol):
    def search(self, query: str, time_range: dict, filters: dict) -> list[dict]:
        ...

class MetricTool(Protocol):
    def query(self, promql: str, time_range: dict, step: str | None = None) -> dict:
        ...

class AlertTool(Protocol):
    def list(self, filters: dict, time_range: dict) -> list[dict]:
        ...

class ChangeTool(Protocol):
    def recent(self, service: str, time_range: dict) -> list[dict]:
        ...
```

- 参考实现：Elastic/Loki/Prometheus/Alertmanager 的最小封装，注意：
  - 查询超时与分页；
  - 查询与原始响应入库引用（raw_ref）；
  - 租户/命名空间隔离；
  - 可观测性埋点（OTel span）。

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
  - 达到成本/时间预算；
  - 用户主动终止；
  - 计划无进一步可执行任务。
- 迭代：
  - 每轮收集新证据后，Planner 复审假设并增量调整任务集；
  - 保留历史任务以便审计与回放。

#### 11. 可靠性与错误处理
- 工具调用：超时 + 重试 + 指数退避 + 熔断；
- LLM 调用：超时 + 重试 + 限流 + 对齐（JSON 模式/约束解码）；
- 降级：某数据源不可用时靠其余证据继续推进，并在报告中标注不确定性；
- 审计：记录每次 Agent 进入、输出、工具调用与输入参数（脱敏）。

#### 12. 安全与合规
- RBAC：按租户/项目/环境限制可见服务与数据；
- 凭证管理：从环境变量/Secret 管理器注入，避免持久化；
- 数据最小化：避免长日志原文进入 LLM，上送前摘要/脱敏；
- 审计与留痕：对外输出可附带证据引用而非原始敏感内容。

#### 13. 成本与性能
- 成本模型：记录 token 与工具调用次数，按任务/租户汇总成本；
- 预算控制：Planner 依据剩余预算约束任务规模与提示词长度；
- 缓存：常见查询、指标基线、知识库段落结果缓存；
- 并发：控制分支并发度；聚合等待时间加超时阈值；
- 紧凑输出：所有 Agent 输出结构化、短摘要 + 引用。

#### 14. 观测与可视化
- OpenTelemetry：为节点与工具建立 span，属性含 tenant、request_id、agent、tool、cost；
- 运行事件总线：记录进入/退出节点、条件分支选择、错误、重试；
- 可视化：LangSmith 或自建可视化，展示计划、分支与证据链路。

#### 15. 对外接口（API/CLI）
- REST/gRPC：
  - POST /troubleshoot: 同步/异步启动一次排障（支持 `mode=async` 返回 job_id）。
  - GET /troubleshoot/{id}: 查询状态、证据、阶段性结论与最终报告。
  - POST /feedback: 用户反馈修复结果与有效性，回灌知识库。
- CLI：`devops-ai troubleshoot --service checkout --issue "p99 latency high" --since 30m`。

#### 16. 测试策略
- 单元：工具封装、提示词模板、JSON 解析；
- 合成场景：构造日志/指标/告警样例，验证从问题到根因闭环；
- 回归：固定输入下输出稳定性测试；
- 线下评测：评估指标（根因命中率、时间、成本、可读性评分）。

#### 17. 部署与运行
- 容器化：多进程/多线程工作器 + API 服务；
- 配置：`config/*.yaml` 指定数据源、限流、并发、预算、提示词插值变量；
- 租户：按租户加载不同工具配置与 RBAC 策略；
- 日志：结构化 JSON，分级输出（运行、审计、访问）。

#### 18. 里程碑规划
- v0（原型）：单租户、最小工具集（Elastic/Prometheus/Alertmanager）、同步 API、基本可观测；
- v1（生产）：异步任务、并行分支、RBAC、审计、限流配额、缓存、LangSmith 集成；
- v2（扩展）：APM/Tracing、CMDB/变更、知识库、自适应提示词、自动化修复建议模板库。

#### 19. 风险与缓解
- 数据质量与权限受限 -> 以证据引用为主，支持缺失数据下的推理并标注不确定性；
- LLM 漂移与不稳定 -> 加强 few-shot、JSON schema 校验、回归集；
- 成本失控 -> 严格预算、分支并发上限、缓存与局部重问策略。

#### 20. 参考目录结构（建议）
```
.
├─ docs/
│  ├─ design-langgraph-devops-agents.md
│  └─ schemas/
├─ src/
│  ├─ app.py                # API/CLI 入口
│  ├─ graph/
│  │  ├─ state.py           # GraphState 与 schema
│  │  ├─ builder.py         # LangGraph 构建
│  │  └─ nodes/             # planner/log/alarm/kpi/summary 节点实现
│  ├─ tools/                # 数据源工具封装
│  ├─ services/             # RBAC、审计、成本、缓存
│  └─ observability/        # OTel 集成
└─ config/
   └─ default.yaml
```

#### 21. 示例请求与输出（简）
- 请求：`支付服务 p99 延迟在 10:00-10:30 升高`。
- Planner：生成查询日志（checkout pod）、指标（http_server_request_duration_seconds{service="checkout"}）与关联最近变更的任务。
- 证据：日志中 5xx 激增、指标 p99 抬升、10:05 有配置变更；
- 总结：疑似配置回滚缺失导致连接池参数过小，建议扩大连接池并回滚到稳定版本；提供验证步骤与回滚指令草案。

—— 本文档可直接指导开发落地，配合 `src/graph/builder.py` 实现与工具封装逐步交付。

