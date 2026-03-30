# Customer Support Email Agent Demo

这个目录是一个基于 LangGraph 的教学型示例项目，用来演示如何实现一个 `customer support email agent`。

它会完成下面几类事情：

- 读取客户邮件
- 对邮件做分类
- 检索本地知识库
- 生成客服回复草稿
- 在需要时升级人工处理
- 在需要时安排后续回访

整个示例偏重“便于学习”而不是“生产级完整系统”，所以代码里加了简短注释，并且在运行过程中打印了比较详细的步骤日志。

## 目录说明

- [requirements.md](/home/xiangdejie/projects/agent/official-example/demo/requirements.md)
  需求说明
- [customer_support_email_agent.py](/home/xiangdejie/projects/agent/official-example/demo/customer_support_email_agent.py)
  核心 LangGraph agent 实现
- [support_knowledge_base.py](/home/xiangdejie/projects/agent/official-example/demo/support_knowledge_base.py)
  本地知识库示例数据
- [run_demo.py](/home/xiangdejie/projects/agent/official-example/demo/run_demo.py)
  命令行运行入口
- [requirements.txt](/home/xiangdejie/projects/agent/official-example/demo/requirements.txt)
  Python 依赖

## 运行环境

建议使用 Python 3.10+。

先进入目录并安装依赖：

```bash
cd /home/xiangdejie/projects/agent/official-example/demo
pip install -r requirements.txt
```

然后配置模型环境变量：

```bash
export DEEPSEEK_API_KEY="你的_deepseek_key"
export DEEPSEEK_MODEL="deepseek-chat"
```

`DEEPSEEK_MODEL` 是可选的，不设置时默认使用 `deepseek-chat`。

## 快速开始

运行内置的 billing 场景：

```bash
python run_demo.py --scenario billing
```

如果你想带上客户邮箱一起测试：

```bash
python run_demo.py --scenario billing --email learner@example.com
```

## 内置场景

项目内置了 5 个典型客服邮件场景：

- `password`
  示例邮件：重置密码问题
- `bug`
  示例邮件：导出 PDF 时崩溃
- `billing`
  示例邮件：订阅被重复扣费
- `feature`
  示例邮件：请求增加 dark mode
- `api`
  示例邮件：API 集成偶发 504

运行示例：

```bash
python run_demo.py --scenario password
python run_demo.py --scenario bug
python run_demo.py --scenario billing
python run_demo.py --scenario feature
python run_demo.py --scenario api
```

## 运行时你会看到什么

这个项目会按 LangGraph 节点逐步打印执行过程，便于学习。

主要会看到下面几个阶段：

- `节点 1: classify_email`
  对客户邮件做结构化分类，输出 `urgency`、`topic`、`summary` 等信息
- `路由判断: after classify_email`
  判断是否需要先查知识库
- `节点 2: search_docs`
  从本地知识库中检索相关文档
- `节点 3: handle_case_actions`
  判断是否需要升级人工、是否需要安排回访
- `节点 4: draft_response`
  生成最终发给客户的回复草稿
- `运行结束: Final State Summary`
  汇总整个 agent 的执行结果

## Python 调用示例

除了命令行方式，也可以在 Python 中直接调用这个 agent。

```python
from customer_support_email_agent import CustomerSupportEmailAgent

agent = CustomerSupportEmailAgent()

result = agent.run(
    email_text="I was charged twice for my subscription today. Please fix this urgently.",
    customer_email="learner@example.com",
)

print("Final response:")
print(result["final_response"])

print("Classification:")
print(result["classification"])

print("Retrieved docs:")
print(result["doc_results"])

print("Escalation:")
print(result["escalation_record"])

print("Follow-up:")
print(result["follow_up_record"])
```

## 代码学习建议

如果你是第一次看这个项目，建议按下面顺序阅读：

1. 先看 [requirements.md](/home/xiangdejie/projects/agent/official-example/demo/requirements.md)
   先理解这个 agent 要解决什么问题
2. 再看 [support_knowledge_base.py](/home/xiangdejie/projects/agent/official-example/demo/support_knowledge_base.py)
   了解它检索的是哪些知识
3. 再看 [customer_support_email_agent.py](/home/xiangdejie/projects/agent/official-example/demo/customer_support_email_agent.py)
   重点关注 `AgentState`、各个节点函数、图的连接方式
4. 最后跑 [run_demo.py](/home/xiangdejie/projects/agent/official-example/demo/run_demo.py)
   结合控制台输出理解执行流程

## 这个示例里演示了什么 LangGraph 思路

- 用 `StateGraph` 把整个 agent 拆成多个清晰节点
- 用共享 `state` 在节点之间传递分类结果、检索结果和动作记录
- 用条件路由控制是否先进入文档检索节点
- 用工具函数模拟客服系统里的能力
  例如：知识库检索、升级人工、安排 follow-up
- 用结构化输出让 LLM 先做稳定分类，再驱动后续流程

## 注意事项

- 这个示例依赖 DeepSeek 模型接口，所以需要有效的 `DEEPSEEK_API_KEY`
- 本项目中的知识库、升级人工、回访安排都是本地 mock 示例，目的是帮助理解 agent 工作流程
- 如果没有安装依赖，程序启动时会直接提示你安装需要的包

## 可扩展方向

你后续可以继续把这个 demo 往更真实的方向扩展，例如：

- 把本地知识库替换成向量数据库检索
- 把 `mock` 的升级工单和回访任务接入真实业务系统
- 给不同 topic 设计不同的回复模板
- 增加更多条件分支
  例如：VIP 客户、退款流程、合规审查
- 把运行日志改造成 LangSmith 或自定义 tracing

