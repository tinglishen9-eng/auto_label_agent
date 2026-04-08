import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List
from urllib.parse import urlencode

import requests


logger = logging.getLogger(__name__)
ACCOUNT_SEARCH_URL = "http://profile.match.sina.com.cn/search/users.php"
SCREEN_NAME_KEYS = ("screen_name", "nick", "nickname", "name")


@dataclass
class AccountIntentSignal:
    query: str
    matched: bool = False
    intent_type: str = "none"
    exact_screen_name: str = ""
    candidate_screen_names: List[str] = field(default_factory=list)
    reason: str = ""
    request_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched": self.matched,
            "intent_type": self.intent_type,
            "exact_screen_name": self.exact_screen_name,
            "candidate_screen_names": self.candidate_screen_names,
            "reason": self.reason,
            "request_url": self.request_url,
        }


class AccountIntentDetector:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

    def detect(self, query: str) -> AccountIntentSignal:
        normalized_query = self._normalize(query)
        signal = AccountIntentSignal(query=query or "")
        if not normalized_query:
            signal.reason = "query 为空，无法判断账号意图"
            return signal

        params = {
            "os": 1,
            "page": 1,
            "count": 2,
            "xsort": 108,
            "default": 1007,
            "q": query,
            "cuid": 0,
            "homepage": 1,
            "sid": "t_wap_ios",
            "sid_group": "mobile",
            "is_teenager": 0,
        }
        signal.request_url = f"{ACCOUNT_SEARCH_URL}?{urlencode(params)}"
        try:
            data = self._fetch_json(params)
        except Exception as exc:
            signal.reason = f"账号搜索接口调用失败: {exc}"
            return signal

        candidates = self._extract_screen_names(data)
        signal.candidate_screen_names = candidates[:5]
        if not candidates:
            signal.reason = "账号搜索接口未返回可用 screen_name"
            return signal

        first_screen_name = candidates[0]
        if self._normalize(first_screen_name) != normalized_query:
            signal.reason = f"首个账号结果为 {first_screen_name}，与 query 不完全匹配"
            return signal

        signal.matched = True
        signal.exact_screen_name = first_screen_name
        signal.intent_type = "account_as_one_intent"
        signal.reason = f"query 与首个账号 {first_screen_name} 完全匹配，说明账号意图是候选需求之一；其他需求仍需结合外部知识继续理解"
        return signal

    def _fetch_json(self, params: Dict[str, Any]) -> Any:
        response = self.session.get(ACCOUNT_SEARCH_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        text = response.text.strip()
        try:
            return response.json()
        except Exception:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
            if not match:
                raise ValueError("接口返回不是合法 JSON")
            return json.loads(match.group(1))

    def _extract_screen_names(self, data: Any) -> List[str]:
        names: List[str] = []
        seen = set()

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key in SCREEN_NAME_KEYS:
                    value = node.get(key)
                    if isinstance(value, str):
                        text = value.strip()
                        if text and text not in seen:
                            seen.add(text)
                            names.append(text)
                            break
                for value in node.values():
                    visit(value)
                return
            if isinstance(node, list):
                for item in node:
                    visit(item)

        visit(data)
        return names

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip()).lower()
