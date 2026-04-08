import json
import logging
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
DEFAULT_ONLINE_KB_WORKERS = 3
DEFAULT_WEIBO_KB_WORKERS = 8
DEFAULT_WEIBO_LLM_KB_WORKERS = 3
WEIBO_AC_URL = (
    "http://i.search.weibo.com/search/libac.php"
    "?ip=10.185.70.62&port=20012&sid=weibo_search_ac&sid_group=mobile"
    "&key={query}&print_more_aclog=0&jx_cuid=3706102772&abtest=1163933937"
    "&num=15&xsort=hot&us=1&req_source_project=mi&dup=1&social_starttime=0"
    "&bubble_request=0&interaction_time=60&request_red_dot=0&post_top_limit=0"
    "&is_scroll=0&socialtime=1&isbctruncate=1&trace=2&t=0&is_media_request=0"
    "&band_rank=0&from_details_page=0&media_headlines_white=0&media_dup_hot=1"
    "&hot_review_page=0"
)
WEIBO_HBASE_URL = "http://getdata.search.weibo.com/getdata/querydata.php?condition={mid}&mode=weibo&format=json"
WEIBO_LLM_ANALYSIS_URL = "http://admin.ai.s.weibo.com/api/llm/analysis_result.json"
_thread_local = threading.local()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def overlap_score(text: str, query_terms: Iterable[str]) -> int:
    haystack = tokenize(text)
    if not haystack:
        return 0
    haystack_set = set(haystack)
    return sum(1 for term in query_terms if term in haystack_set)


def unique_keep_order(items: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


@dataclass
class KnowledgeChunk:
    source: str
    title: str
    content: str
    score: int = 0

    def to_dict(self) -> Dict[str, str]:
        return {
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "score": self.score,
        }


class LocalKnowledgeBase:
    def __init__(self, path: str):
        self.path = Path(path)
        self.records = self._load_records()

    def _load_records(self) -> List[KnowledgeChunk]:
        if not self.path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("JSON 知识库应为数组。")
            return [self._to_chunk(item, idx) for idx, item in enumerate(raw)]
        if suffix == ".jsonl":
            chunks: List[KnowledgeChunk] = []
            for idx, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                chunks.append(self._to_chunk(json.loads(line), idx))
            return chunks

        chunks = []
        for idx, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            chunks.append(KnowledgeChunk(source=str(self.path), title=f"line-{idx + 1}", content=line))
        return chunks

    def _to_chunk(self, item: object, idx: int) -> KnowledgeChunk:
        if isinstance(item, str):
            return KnowledgeChunk(source=str(self.path), title=f"item-{idx + 1}", content=item)
        if not isinstance(item, dict):
            raise ValueError("知识库条目必须为字符串或对象。")
        title = str(item.get("title") or f"item-{idx + 1}")
        content = str(item.get("content") or item.get("text") or "")
        source = str(item.get("source") or self.path)
        return KnowledgeChunk(source=source, title=title, content=content)

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        queries = unique_keep_order(queries)
        logger.info("本地知识库检索: queries=%d, top_k=%d", len(queries), top_k)
        terms: List[str] = []
        for query in queries:
            terms.extend(tokenize(query))
        if not terms:
            return []

        ranked: List[KnowledgeChunk] = []
        for chunk in self.records:
            score = overlap_score(f"{chunk.title}\n{chunk.content}", terms)
            if score <= 0:
                continue
            ranked.append(
                KnowledgeChunk(
                    source=chunk.source,
                    title=chunk.title,
                    content=chunk.content,
                    score=score,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]


class OnlineKnowledgeBase:
    def __init__(self, max_workers: int = DEFAULT_ONLINE_KB_WORKERS):
        self.max_workers = max_workers

    def _search_one(self, query: str) -> KnowledgeChunk:
        try:
            response = get_session().get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "no_redirect": "1",
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return KnowledgeChunk(
                source="duckduckgo",
                title=query,
                content=f"检索失败: {exc}",
                score=0,
            )

        parts: List[str] = []
        abstract = (data.get("Abstract") or "").strip()
        if abstract:
            parts.append(f"摘要: {abstract}")

        related = data.get("RelatedTopics") or []
        topic_texts: List[str] = []
        for item in related:
            if len(topic_texts) >= 3:
                break
            if isinstance(item, dict) and item.get("Text"):
                topic_texts.append(item["Text"].strip())
            for nested in item.get("Topics", []) if isinstance(item, dict) else []:
                if len(topic_texts) >= 3:
                    break
                if isinstance(nested, dict) and nested.get("Text"):
                    topic_texts.append(nested["Text"].strip())

        if topic_texts:
            parts.append("相关信息:")
            parts.extend(f"- {text}" for text in topic_texts[:3])

        content = "\n".join(parts) if parts else "无明显补充信息。"
        return KnowledgeChunk(
            source="duckduckgo",
            title=query,
            content=content,
            score=max(1, len(parts)),
        )

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        deduped_queries = unique_keep_order(queries)[:top_k]
        if not deduped_queries:
            return []

        logger.info("DuckDuckGo 检索: queries=%d, workers=%d", len(deduped_queries), self.max_workers)
        chunks: List[KnowledgeChunk] = [
            KnowledgeChunk(source="", title="", content="", score=0) for _ in deduped_queries
        ]
        worker_count = max(1, min(self.max_workers, len(deduped_queries)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._search_one, query): idx for idx, query in enumerate(deduped_queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunks[idx] = future.result()
        return chunks[:top_k]


class WeiboSearchKnowledgeBase:
    def __init__(self, max_workers: int = DEFAULT_WEIBO_KB_WORKERS):
        self.max_workers = max_workers

    def _extract_recall_text(self, item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        parts: List[str] = []
        for key in ("screen_name", "nick", "nickname", "name", "title", "text", "desc", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for nested_key in ("user", "card", "item"):
            nested = item.get(nested_key)
            if isinstance(nested, dict):
                for key in ("screen_name", "nick", "nickname", "name", "title", "text", "desc", "content"):
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
        return " | ".join(unique_keep_order(parts))

    def _recall_items(self, query: str, recall_size: int) -> List[Dict[str, str]]:
        count = 0
        while count < 3:
            try:
                response = get_session().get(WEIBO_AC_URL.format(query=query), timeout=10)
                response.raise_for_status()
                res = response.json()
                result = res.get("sp", {}).get("result") or []
                items: List[Dict[str, str]] = []
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    mid = str(item.get("ID") or "").strip()
                    if mid:
                        items.append(
                            {
                                "mid": mid,
                                "text": self._extract_recall_text(item),
                            }
                        )
                    if len(items) >= recall_size:
                        break
                deduped: List[Dict[str, str]] = []
                seen = set()
                for item in items:
                    mid = item["mid"]
                    if mid in seen:
                        continue
                    seen.add(mid)
                    deduped.append(item)
                logger.debug("weibo_search recall query=%s mids=%s", query, [item["mid"] for item in deduped])
                return deduped
            except Exception:
                count += 1
                traceback.print_exc()
        return []

    def _get_hbase(self, mid: str):
        count = 0
        while count < 3:
            try:
                response = get_session().get(WEIBO_HBASE_URL.format(mid=mid), timeout=5)
                response.raise_for_status()
                res = response.json()
                content = res["CONTENT"]
                islong = res.get("ISLONG")
                if islong and int(islong) == 1:
                    content = res["LONGTEXT"]

                abstract = res.get("IDX_COMPREHEND_ABSTRACT")
                keywords = res.get("IDX_COMPREHEND_KEYWORDS")
                title = res.get("IDX_COMPREHEND_TITLE")

                if abstract and abstract.strip():
                    content += "<abstract>" + abstract
                if title and title.strip():
                    content += "<title>" + title
                if keywords and keywords.strip():
                    content += "<keywords>" + keywords
                return content
            except Exception:
                count += 1
                traceback.print_exc()
        return None

    def _get_hbase_batch(self, mids: Sequence[str]):
        mids = unique_keep_order(mids)
        results = [None] * len(mids)
        if not mids:
            return results

        worker_count = max(1, min(self.max_workers, len(mids)))
        logger.info("微博搜索 HBase 拉取: mids=%d, workers=%d", len(mids), worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._get_hbase, mid): idx for idx, mid in enumerate(mids)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    traceback.print_exc()
                    results[idx] = None
        return results

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        deduped_queries = unique_keep_order(queries)
        if not deduped_queries:
            return []

        logger.info("微博搜索知识检索: queries=%d, top_k=%d", len(deduped_queries), top_k)
        collected: List[KnowledgeChunk] = []
        for query in deduped_queries:
            recall_items = self._recall_items(query, recall_size=max(top_k, 5))
            mids = [item["mid"] for item in recall_items]
            logger.debug("query=%s recalled_mids=%d", query, len(mids))
            contents = self._get_hbase_batch(mids[: max(top_k, 5)])
            for idx, ((mid, recall_item), content) in enumerate(zip(zip(mids, recall_items), contents), start=1):
                final_content = content or recall_item.get("text") or ""
                if not final_content:
                    logger.debug("query=%s mid=%s 未拿到 hbase 内容，也没有召回兜底文本", query, mid)
                    continue
                if not content:
                    logger.debug("query=%s mid=%s hbase 为空，使用召回结果文本兜底", query, mid)
                collected.append(
                    KnowledgeChunk(
                        source="weibo_search",
                        title=f"{query}-mid-{mid}",
                        content=final_content,
                        score=max(top_k - idx + 1, 1),
                    )
                )

        collected.sort(key=lambda item: item.score, reverse=True)
        return collected[:top_k]


class WeiboLLMKnowledgeBase:
    DEFAULT_CH_TYPE = "ori_data_type"
    DEFAULT_FILTER_TAG = "30"
    REQUEST_TIMEOUT = 5

    def __init__(
        self,
        max_workers: int = DEFAULT_WEIBO_LLM_KB_WORKERS,
        ch_type: str = DEFAULT_CH_TYPE,
        filter_tag: str = DEFAULT_FILTER_TAG,
        request_timeout: int = REQUEST_TIMEOUT,
    ):
        self.max_workers = max_workers
        self.ch_type = ch_type
        self.filter_tag = filter_tag
        self.request_timeout = request_timeout

    def _build_url(self, query: str) -> str:
        params = {
            "query": query,
            "ch_type": self.ch_type,
            "filter_tag": self.filter_tag,
        }
        return f"{WEIBO_LLM_ANALYSIS_URL}?{urlencode(params)}"

    def _extract_content(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        data = payload.get("data")
        if not isinstance(data, dict):
            logger.error("微博 LLM 知识接口返回结构异常: %s", type(payload))
            return ""
        for key in ("deepseek", "deepseek_stream", "preview"):
            item = data.get(key)
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    return content
        logger.error("微博 LLM 知识接口未命中预期字段，data keys=%s", list(data.keys()))
        return ""

    def _search_one(self, query: str) -> KnowledgeChunk:
        url = self._build_url(query)
        try:
            response = get_session().get(
                WEIBO_LLM_ANALYSIS_URL,
                params={
                    "query": query,
                    "ch_type": self.ch_type,
                    "filter_tag": self.filter_tag,
                },
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                logger.error("微博 LLM 知识接口状态码异常: status=%s, query=%s", response.status_code, query)
                return KnowledgeChunk(
                    source="weibo_llm",
                    title=query,
                    content=f"检索失败: HTTP {response.status_code}",
                    score=0,
                )
            payload = response.json()
            content = self._extract_content(payload)
            if not content:
                return KnowledgeChunk(
                    source="weibo_llm",
                    title=query,
                    content="无明显补充信息。",
                    score=0,
                )
            return KnowledgeChunk(
                source="weibo_llm",
                title=query,
                content=content,
                score=max(1, len(content) // 80),
            )
        except Exception as exc:
            logger.error("微博 LLM 知识接口请求失败: query=%s, error=%s, url=%s", query, exc, url)
            return KnowledgeChunk(
                source="weibo_llm",
                title=query,
                content=f"检索失败: {exc}",
                score=0,
            )

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        deduped_queries = unique_keep_order(queries)[:top_k]
        if not deduped_queries:
            return []
        logger.info("微博 LLM 知识检索: queries=%d, workers=%d", len(deduped_queries), self.max_workers)
        chunks: List[KnowledgeChunk] = [
            KnowledgeChunk(source="", title="", content="", score=0) for _ in deduped_queries
        ]
        worker_count = max(1, min(self.max_workers, len(deduped_queries)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._search_one, query): idx for idx, query in enumerate(deduped_queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunks[idx] = future.result()
        chunks.sort(key=lambda item: item.score, reverse=True)
        return chunks[:top_k]


class MultiSourceKnowledgeBase:
    def __init__(self, providers: Sequence[str]):
        normalized = []
        for provider in providers:
            name = (provider or "").strip().lower()
            if not name or name in normalized:
                continue
            normalized.append(name)
        self.providers = normalized or ["duckduck"]

    def _create_provider(self, provider: str):
        if provider == "duckduck":
            return OnlineKnowledgeBase()
        if provider == "weibo_search":
            return WeiboSearchKnowledgeBase()
        if provider == "weibo_llm":
            return WeiboLLMKnowledgeBase()
        raise ValueError(f"不支持的知识源 provider={provider}")

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        deduped_queries = unique_keep_order(queries)
        if not deduped_queries:
            return []

        logger.info("多知识源检索: providers=%s, queries=%d", ",".join(self.providers), len(deduped_queries))
        merged: List[KnowledgeChunk] = []
        for provider in self.providers:
            merged.extend(self._create_provider(provider).search(deduped_queries, top_k=top_k))
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]
