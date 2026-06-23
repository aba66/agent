# DDSS gpt-5.5 调用失败排查报告

生成时间：2026-06-23  
项目路径：`/home/xiangdejie/projects/agent/harness/world-cup-prediction`

## 结论

`worldcup_prediction` 代码侧已能正确选择 DDSS 兼容接口参数：

- `base_url`: `https://code.ddsst.online/v1`
- `model`: `gpt-5.5`
- API key 环境变量：`DDSS_API_KEY`
- provider：`openai-compatible`

但从当前 Codex 执行环境直接访问 `https://code.ddsst.online/v1` 时，请求在网关层被拒绝，返回 Cloudflare/WAF 类错误：

```text
HTTP 403
error code: 1010
```

这不是模型返回的 JSON 错误，也不是 `model` 名称错误；请求没有进入正常 OpenAI-compatible API 响应阶段。

## 已执行的最小测试

### 1. 裸 HTTP `chat/completions` 测试

测试目标：

```text
POST https://code.ddsst.online/v1/chat/completions
model=gpt-5.5
prompt=你好，回复 ok
```

测试结果：

```text
status=http_error 403
error code: 1010
```

使用 `curl` 增加常见请求头后仍失败：

```text
HTTP/2 403
server: cloudflare

Your request was blocked.
```

### 2. `worldcup_prediction` provider 层 `responses` 测试

测试代码走项目自己的 provider：

```python
import os
from worldcup_prediction.providers import get_provider

os.environ["DDSS_API_KEY"] = "<redacted>"
provider = get_provider("auto")

print(provider.name)
print(provider.model)
print(provider.base_url)
print(provider.wire_api)

result = provider._complete_json('请只返回 JSON：{"ok": true, "text": "ok"}')
```

provider 选择结果：

```text
provider= openai-compatible
model= gpt-5.5
base_url= https://code.ddsst.online/v1
wire_api= responses
```

调用结果：

```text
ProviderError: HTTP 403: error code: 1010
```

### 3. `worldcup_prediction` provider 层强制 `chat/completions` 测试

强制设置：

```bash
export WCP_OPENAI_WIRE_API=chat_completions
```

provider 选择结果：

```text
provider= openai-compatible
model= gpt-5.5
base_url= https://code.ddsst.online/v1
wire_api= chat_completions
```

调用结果：

```text
ProviderError: HTTP 403: Forbidden
```

## 当前代码如何选择 DDSS

文件：`worldcup_prediction/providers.py`

当前逻辑：

- 若显式 `--llm-provider mock`，才使用本地 deterministic mock。
- 默认 `auto` 模式读取真实 API key。
- key 优先级包含：
  - `AUTODL_API_KEY`
  - `DDSS_API_KEY`
  - `OPENAI_API_KEY`
  - `WCP_OPENAI_API_KEY`
- 当设置 `DDSS_API_KEY` 且未覆盖 base URL 时，默认：
  - `base_url=https://code.ddsst.online/v1`
  - `model=gpt-5.5`
  - `wire_api=responses`
- 可通过 `WCP_OPENAI_WIRE_API=chat_completions` 强制走 Chat Completions。

## 本环境信息

Python：

```text
Python 3.8.10
```

OpenAI SDK：

```text
未安装 openai Python SDK
```

因此本项目当前用标准库 `urllib.request` 调用 OpenAI-compatible HTTP API，而不是官方 `openai` SDK。

测试中没有把完整 API key 写入仓库；命令里只用于临时环境变量或内存变量。

## 与“另一个成功项目”建议对照的差异

请在成功调用的项目中重点对比以下项：

1. **出口网络/IP 是否不同**
   - 当前环境返回 Cloudflare `403 error code: 1010`。
   - 如果成功项目同机但不同运行环境，可能存在代理、环境变量、容器网络、TLS 指纹或出口 IP 差异。

2. **是否通过 OpenAI Python SDK 调用**
   - 成功示例使用：
     ```python
     from openai import OpenAI
     client = OpenAI(api_key=..., base_url="https://code.ddsst.online/v1")
     ```
   - 本项目当前使用 `urllib.request`。
   - 如果 DDSS/Cloudflare 对客户端特征敏感，SDK 的请求头、HTTP 栈或 TLS 指纹可能不同。

3. **实际调用的是 `/responses` 还是 `/chat/completions`**
   - 本项目两者都试过，都被 403。
   - 成功项目可确认最终命中的路径、请求体和响应头。

4. **代理环境变量**
   - 对比：
     ```bash
     env | grep -Ei 'proxy|http_proxy|https_proxy|all_proxy|no_proxy'
     ```

5. **请求头差异**
   - 当前最小请求头：
     ```text
     Authorization: Bearer <redacted>
     Content-Type: application/json
     ```
   - `curl` 测试额外加过：
     ```text
     User-Agent: OpenAI/Python 1.0
     ```
     仍被拦截。

6. **Cloudflare 安全策略**
   - `error code: 1010` 通常意味着请求被站点安全规则拦截。
   - 需要成功项目确认是否经过了允许名单、代理、特殊 header、cookie、SDK 默认行为或不同网络出口。

## 可复现命令

在本项目目录：

```bash
cd /home/xiangdejie/projects/agent/harness/world-cup-prediction
export DDSS_API_KEY="<redacted>"
python3 -m worldcup_prediction predict --date 2026-06-23
```

预期当前环境下失败：

```text
ProviderError: HTTP 403: error code: 1010
```

强制 Chat Completions：

```bash
export DDSS_API_KEY="<redacted>"
export WCP_OPENAI_WIRE_API=chat_completions
python3 - <<'PY'
from worldcup_prediction.providers import get_provider
provider = get_provider("auto")
print(provider.name, provider.model, provider.base_url, provider.wire_api)
print(provider._complete_json('请只返回 JSON：{"ok": true, "text": "ok"}').payload)
PY
```

预期当前环境下失败：

```text
ProviderError: HTTP 403: Forbidden
```

## 当前项目测试状态

本地 deterministic mock 的单元测试通过：

```text
python3 -m unittest discover -s tests
OK
```

这说明 `worldcup_prediction` 的本地编排、数据源、报告生成、复盘账本路径可用；失败点集中在当前环境访问 DDSS 外部接口。
