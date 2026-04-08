import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.adapters.llm_client import (
    DEFAULT_ENDPOINTS,
    extract_proxy_message,
    provider_requires_api_key,
    resolve_api_key,
    resolve_endpoint,
    LLMClient,
)


class LLMClientTest(unittest.TestCase):
    def test_mproxy_does_not_require_api_key(self) -> None:
        self.assertFalse(provider_requires_api_key("mproxy"))
        self.assertFalse(provider_requires_api_key("weibo_proxy"))

    def test_mproxy_has_default_endpoint(self) -> None:
        self.assertEqual(
            DEFAULT_ENDPOINTS["mproxy"],
            "http://mproxy.search.weibo.com/llm/generate",
        )
        self.assertEqual(
            resolve_endpoint("mproxy", ""),
            "http://mproxy.search.weibo.com/llm/generate",
        )

    def test_resolve_mproxy_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"MPROXY_API_KEY": "test-api-key"}, clear=False):
            self.assertEqual(resolve_api_key("mproxy"), "test-api-key")

    def test_extract_proxy_message_supports_nested_choices(self) -> None:
        payload = {
            "data": {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true, "source": "mproxy"}',
                        }
                    }
                ]
            }
        }
        self.assertEqual(
            extract_proxy_message(payload),
            '{"ok": true, "source": "mproxy"}',
        )

    def test_mproxy_payload_can_include_api_key(self) -> None:
        client = LLMClient(
            provider="mproxy",
            model="qwen272b",
            endpoint="http://mproxy.search.weibo.com/llm/generate",
            api_key="demo-api-key",
        )

        captured = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"text": "ok"}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("auto_label_agent.adapters.llm_client.requests.post", new=fake_post):
            self.assertEqual(client.complete_text("system", "user"), "ok")

        self.assertEqual(captured["json"]["api_key"], "demo-api-key")
        self.assertEqual(captured["json"]["sid"], "comprehensive_search_yybz")

    def test_mproxy_retries_on_408(self) -> None:
        client = LLMClient(
            provider="mproxy",
            model="qwen272b",
            endpoint="http://mproxy.search.weibo.com/llm/generate",
            api_key="",
            proxy_max_retries=1,
        )

        calls = {"count": 0}

        class RetryResponse:
            def __init__(self, status_code: int, payload=None):
                self.status_code = status_code
                self._payload = payload or {"text": "ok"}

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    error = requests.exceptions.HTTPError("408 Client Error")
                    error.response = self
                    raise error

            def json(self):
                return self._payload

        def fake_post(url, json=None, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                return RetryResponse(408)
            return RetryResponse(200, {"text": "retry-ok"})

        with patch("auto_label_agent.adapters.llm_client.requests.post", new=fake_post):
            self.assertEqual(client.complete_text("system", "user"), "retry-ok")
        self.assertEqual(calls["count"], 2)

    def test_mproxy_raises_clear_error_after_retries(self) -> None:
        client = LLMClient(
            provider="mproxy",
            model="qwen272b",
            endpoint="http://mproxy.search.weibo.com/llm/generate",
            api_key="",
            proxy_max_retries=1,
        )

        def fake_post(url, json=None, timeout=None):
            raise requests.exceptions.Timeout("timeout")

        with patch("auto_label_agent.adapters.llm_client.requests.post", new=fake_post):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete_text("system", "user")
        self.assertIn("多次请求失败", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
