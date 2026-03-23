import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import requests


DEFAULT_ENDPOINTS: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}


def resolve_api_key(provider: str) -> str:
    provider = provider.lower().strip()
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY", "")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    return os.getenv(f"{provider.upper()}_API_KEY", "")


def resolve_endpoint(provider: str, endpoint: str) -> str:
    provider = provider.lower().strip()
    if endpoint:
        return endpoint
    if provider in DEFAULT_ENDPOINTS:
        return DEFAULT_ENDPOINTS[provider]
    raise ValueError(f"不支持 provider={provider}，请显式提供 endpoint。")


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


@dataclass
class LLMClient:
    provider: str
    model: str
    endpoint: str
    api_key: str
    temperature: float = 0.1
    timeout: int = 90

    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
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

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
