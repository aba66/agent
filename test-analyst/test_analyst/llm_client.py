from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

from .models import LLMSettings
from .utils import extract_json_object


class TestAnalystLLMClient:
    """Thin wrapper over OpenAI-compatible chat completion APIs."""

    def __init__(self, settings: LLMSettings) -> None:
        normalized_base_url = self._normalize_client_base_url(settings.base_url)
        self.settings = settings.copy(update={"base_url": normalized_base_url}, deep=True)
        self._client = None
        self._chat_endpoint = self._normalize_chat_endpoint(settings.base_url)
        self.last_error: str | None = None
        if self.settings.is_configured:
            kwargs = {"api_key": self.settings.api_key}
            if self.settings.base_url:
                kwargs["base_url"] = self.settings.base_url
            self._client = OpenAI(**kwargs)

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.model)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
    ) -> dict:
        self.last_error = None
        if not self._client:
            self.last_error = (
                "当前未配置 LLM API，请设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，"
                "或兼容的 QWEN_BASE_URL / QWEN_API_KEY / QWEN_MODEL。"
            )
            raise RuntimeError(self.last_error)

        errors: list[str] = []
        for use_json_mode in (True, False):
            request_kwargs = self._build_sdk_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_json_mode=use_json_mode,
            )
            try:
                content = self._request_with_sdk(request_kwargs)
                return extract_json_object(content)
            except Exception as exc:  # pragma: no cover - provider variability
                errors.append(f"sdk(json_mode={use_json_mode}): {exc}")

            try:
                content = self._request_with_http(request_kwargs)
                return extract_json_object(content)
            except Exception as exc:  # pragma: no cover - provider variability
                errors.append(f"http(json_mode={use_json_mode}): {exc}")
                continue

        self.last_error = f"{agent_name} LLM 调用失败：{' | '.join(errors)}"
        raise RuntimeError(self.last_error)

    @staticmethod
    def _normalize_client_base_url(url: str | None) -> str | None:
        if not url:
            return None
        cleaned = url.strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned[: -len("/chat/completions")]
        return cleaned

    @staticmethod
    def _normalize_chat_endpoint(url: str | None) -> str | None:
        if not url:
            return None
        cleaned = url.strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        return f"{cleaned}/chat/completions"

    def _build_messages(self, *, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_sdk_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool,
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": self._build_messages(system_prompt=system_prompt, user_prompt=user_prompt),
        }
        if self.settings.enable_thinking is not None:
            request_kwargs["extra_body"] = {"enable_thinking": self.settings.enable_thinking}
        if use_json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}
        return request_kwargs

    def _build_http_payload(self, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": request_kwargs["model"],
            "temperature": request_kwargs["temperature"],
            "max_tokens": request_kwargs["max_tokens"],
            "messages": request_kwargs["messages"],
        }
        if request_kwargs.get("response_format"):
            payload["response_format"] = request_kwargs["response_format"]
        if request_kwargs.get("extra_body"):
            payload.update(request_kwargs["extra_body"])
        return payload

    def _request_with_sdk(self, request_kwargs: dict[str, Any]) -> str:
        if not self._client:
            raise RuntimeError("OpenAI 客户端未初始化。")
        response = self._client.chat.completions.create(**request_kwargs)
        return self._extract_content(response)

    def _request_with_http(self, request_kwargs: dict[str, Any]) -> str:
        if not self._chat_endpoint:
            raise RuntimeError("未配置可直连的 chat/completions endpoint。")

        request = urllib.request.Request(
            self._chat_endpoint,
            data=json.dumps(self._build_http_payload(request_kwargs)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._format_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"网络请求失败: {exc.reason}") from exc

        content = self._extract_content(payload)
        if not content:
            raise RuntimeError("接口返回成功，但没有拿到模型回复文本。")
        return content

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise RuntimeError("模型返回中缺少 choices。")

        message = getattr(choices[0], "message", None)
        if message is None and isinstance(choices[0], dict):
            message = choices[0].get("message", {})

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text:
                        parts.append(text)
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def _format_http_error(self, exc: urllib.error.HTTPError) -> str:
        body = exc.read().decode("utf-8", errors="ignore")
        if not body:
            return f"HTTP {exc.code}: {exc.reason}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return f"HTTP {exc.code}: {body}"

        message = (
            payload.get("error", {}).get("message")
            or payload.get("message")
            or payload.get("detail")
            or body
        )
        return f"HTTP {exc.code}: {message}"
