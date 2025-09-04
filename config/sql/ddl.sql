-- DevOps Agent Database Schema
-- SQLite DDL for troubleshooting system

-- 基础请求与状态
CREATE TABLE IF NOT EXISTS troubleshooting_requests (
  id TEXT PRIMARY KEY,
  tenant_id TEXT,
  title TEXT NOT NULL,
  description TEXT,
  service TEXT NOT NULL,
  environment TEXT DEFAULT 'prod',
  severity TEXT DEFAULT 'medium',
  time_from TEXT,
  time_to TEXT,
  artifacts_hints TEXT,  -- JSON
  mode TEXT DEFAULT 'async',
  status TEXT DEFAULT 'running',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_requests_tenant ON troubleshooting_requests(tenant_id);
CREATE INDEX idx_requests_status ON troubleshooting_requests(status);
CREATE INDEX idx_requests_created ON troubleshooting_requests(created_at);

-- 调查计划
CREATE TABLE IF NOT EXISTS investigation_plans (
  plan_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  created_by TEXT DEFAULT 'planner',
  goals_json TEXT,
  tasks_json TEXT,
  version INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

CREATE INDEX idx_plans_request ON investigation_plans(request_id);

-- 调查任务
CREATE TABLE IF NOT EXISTS investigation_tasks (
  task_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  plan_id TEXT,
  type TEXT NOT NULL CHECK(type IN ('log', 'alarm', 'kpi', 'knowledge', 'change')),
  inputs_json TEXT NOT NULL,
  hypotheses_json TEXT,
  priority INTEGER DEFAULT 1,
  timeout_s INTEGER DEFAULT 30,
  status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
  result_summary TEXT,
  started_at TEXT,
  completed_at TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE,
  FOREIGN KEY(plan_id) REFERENCES investigation_plans(plan_id) ON DELETE SET NULL
);

CREATE INDEX idx_tasks_request ON investigation_tasks(request_id);
CREATE INDEX idx_tasks_status ON investigation_tasks(status);
CREATE INDEX idx_tasks_type ON investigation_tasks(type);

-- 收集的证据
CREATE TABLE IF NOT EXISTS evidences (
  evidence_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  task_id TEXT,
  source TEXT NOT NULL CHECK(source IN ('log', 'alarm', 'kpi', 'knowledge', 'change')),
  summary TEXT NOT NULL,
  raw_ref_json TEXT,  -- 包含查询语句、服务器引用等
  time_from TEXT,
  time_to TEXT,
  quality_score REAL DEFAULT 0.5,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE,
  FOREIGN KEY(task_id) REFERENCES investigation_tasks(task_id) ON DELETE SET NULL
);

CREATE INDEX idx_evidences_request ON evidences(request_id);
CREATE INDEX idx_evidences_source ON evidences(source);

-- 发现
CREATE TABLE IF NOT EXISTS findings (
  finding_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  hypothesis_ref TEXT,
  confidence REAL DEFAULT 0.5,
  impact_scope_json TEXT,
  supporting_evidence_json TEXT,  -- evidence_id数组
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

CREATE INDEX idx_findings_request ON findings(request_id);

-- 根因分析结果
CREATE TABLE IF NOT EXISTS root_causes (
  id TEXT PRIMARY KEY,
  request_id TEXT UNIQUE NOT NULL,
  hypothesis TEXT NOT NULL,
  confidence REAL DEFAULT 0.5,
  affected_components_json TEXT,
  time_correlation_json TEXT,
  change_correlation_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

-- 修复建议
CREATE TABLE IF NOT EXISTS remediations (
  id TEXT PRIMARY KEY,
  request_id TEXT UNIQUE NOT NULL,
  actions_json TEXT NOT NULL,
  risk_level TEXT DEFAULT 'medium',
  required_approvals_json TEXT,
  validation_steps_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

-- Agent错误记录
CREATE TABLE IF NOT EXISTS agent_errors (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  error_type TEXT NOT NULL,
  message TEXT,
  retriable INTEGER DEFAULT 1,
  attempt INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

CREATE INDEX idx_errors_request ON agent_errors(request_id);
CREATE INDEX idx_errors_agent ON agent_errors(agent);

-- 对话历史
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  metadata_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_request ON messages(request_id);
CREATE INDEX idx_messages_created ON messages(created_at);

-- Token使用统计
CREATE TABLE IF NOT EXISTS token_usage (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  prompt_tokens INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0,
  tool_calls INTEGER DEFAULT 0,
  wall_time_ms INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(request_id) REFERENCES troubleshooting_requests(id) ON DELETE CASCADE
);

CREATE INDEX idx_token_usage_request ON token_usage(request_id);

-- LangGraph检查点存储（用于状态持久化）
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
  thread_id TEXT NOT NULL,
  checkpoint_id TEXT NOT NULL,
  parent_id TEXT,
  checkpoint BLOB NOT NULL,
  metadata TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX idx_checkpoints_thread ON langgraph_checkpoints(thread_id);
CREATE INDEX idx_checkpoints_created ON langgraph_checkpoints(created_at);

-- 会话元数据
CREATE TABLE IF NOT EXISTS langgraph_writes (
  thread_id TEXT NOT NULL,
  checkpoint_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  channel TEXT NOT NULL,
  type TEXT,
  value BLOB,
  PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
);

CREATE INDEX idx_writes_thread ON langgraph_writes(thread_id);

-- 创建触发器自动更新updated_at
CREATE TRIGGER update_requests_timestamp 
AFTER UPDATE ON troubleshooting_requests
BEGIN
  UPDATE troubleshooting_requests SET updated_at = datetime('now')
  WHERE id = NEW.id;
END;

CREATE TRIGGER update_tasks_timestamp 
AFTER UPDATE ON investigation_tasks
BEGIN
  UPDATE investigation_tasks SET updated_at = datetime('now')
  WHERE task_id = NEW.task_id;
END;