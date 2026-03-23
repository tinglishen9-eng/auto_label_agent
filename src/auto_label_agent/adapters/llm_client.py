import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import requests


DEFAULT_ENDPOINTS: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "kimi": "https://api.moonshot.cn/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "mproxy": "http://mproxy.search.weibo.com/llm/generate",
    "weibo_proxy": "http://mproxy.search.weibo.com/llm/generate",
}


PROVIDER_API_KEY_ENV: Dict[str, tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "kimi": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
    "moonshot": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "mproxy": (),
    "weibo_proxy": (),
}


PROVIDERS_WITHOUT_API_KEY = {"mproxy", "weibo_proxy"}


def resolve_api_key(provider: str) -> str:
    provider = provider.lower().strip()
    for env_name in PROVIDER_API_KEY_ENV.get(provider, (f"{provider.upper()}_API_KEY",)):
        value = os.getenv(env_name, "")
        if value:
            return value
    return ""


def resolve_endpoint(provider: str, endpoint: str) -> str:
    provider = provider.lower().strip()
    if endpoint:
        return endpoint
    if provider in DEFAULT_ENDPOINTS:
        return DEFAULT_ENDPOINTS[provider]
    raise ValueError(f"不支持 provider={provider}，请显式提供 endpoint。")


def provider_requires_api_key(provider: str) -> bool:
    return provider.lower().strip() not in PROVIDERS_WITHOUT_API_KEY


def extract_json_block(text: str) -> Any:
    content = text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
    if not match:
        raise ValueError(f"模型未返回合法 JSON: {text}")
    return json.loads(match.group(1))


def extract_proxy_message(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        for item in data:
            message = extract_proxy_message(item)
            if message:
                return message
        return ""
    if not isinstance(data, dict):
        return ""

    for key in ("text", "content", "response", "answer", "generated_text", "output_text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value

    message = data.get("message")
    if isinstance(message, str) and message.strip():
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_message = choice.get("message")
            if isinstance(choice_message, dict):
                content = choice_message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                return text

    for key in ("result", "data"):
        nested = data.get(key)
        if nested is not None:
            message = extract_proxy_message(nested)
            if message:
                return message

    return ""


@dataclass
class LLMClient:
    provider: str
    model: str
    endpoint: str
    api_key: str
    temperature: float = 0.1
    timeout: int = 90
    sid: str = "comprehensive_search_yybz"

    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        if self._is_proxy_provider():
            message = self._complete_proxy_text(system_prompt, user_prompt)
            return extract_json_block(message)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        response = self._post(payload)
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        return extract_json_block(message)

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        if self._is_proxy_provider():
            return self._complete_proxy_text(system_prompt, user_prompt)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        response = self._post(payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _is_proxy_provider(self) -> bool:
        return self.provider.lower().strip() in {"mproxy", "weibo_proxy"}

    def _complete_proxy_text(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        request_body = {
            "id": "123",
            "temperature": self.temperature,
            "top_p": 0.95,
            "max_tokens": 1024,
            "repetition_penalty": 1.15,
            "messages": ["user", prompt],
        }
        payload = {
            "payload": request_body,
            "sid": self.sid,
            "model": self.model,
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        response_data = response.json()
        message = extract_proxy_message(response_data)
        if message:
            return message
        return json.dumps(response_data, ensure_ascii=False)

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code < 400:
            return response

        if "response_format" not in payload:
            response.raise_for_status()
            return response

        fallback_payload = dict(payload)
        fallback_payload.pop("response_format", None)
        fallback_response = requests.post(
            self.endpoint,
            json=fallback_payload,
            headers=headers,
            timeout=self.timeout,
        )
        fallback_response.raise_for_status()
        return fallback_response
