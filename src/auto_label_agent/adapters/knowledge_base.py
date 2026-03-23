import json
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import requests


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
DEFAULT_ONLINE_KB_WORKERS = 3
DEFAULT_WEIBO_KB_WORKERS = 8
WEIBO_AC_URL = (
    "http://i.search.weibo.com/search/libac.php"
    "?ip=10.54.34.27&port=20012&sid=weibo_search_ac&sid_group=mobile"
    "&key={query}&print_more_aclog=1&jx_cuid=3706102772&abtest=1163933937"
    "&num=15&xsort=hot&us=1&req_source_project=mi&dup=1&social_starttime=0"
    "&bubble_request=0&interaction_time=60&request_red_dot=0&post_top_limit=0"
    "&is_scroll=0&socialtime=1&isbctruncate=1&trace=2&t=0&is_media_request=0"
    "&band_rank=0&from_details_page=0&media_headlines_white=0&media_dup_hot=1"
    "&hot_review_page=0"
)
WEIBO_HBASE_URL = "http://getdata.search.weibo.com/getdata/querydata.php?condition={mid}&mode=weibo&format=json"
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

    def _func_bs_score(self, query: str) -> tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        keywords_list: Dict[str, List[int]] = {}
        vec_list: Dict[str, List[int]] = {}
        count = 0
        while count < 3:
            try:
                response = get_session().get(WEIBO_AC_URL.format(query=query), timeout=10)
                response.raise_for_status()
                res = response.json()
                aclog = res["sp"]["aclog"]
                lidx = aclog.find("bs_score")
                ridx = aclog.rfind("bs_score")
                if lidx < 0 or ridx < 0:
                    return keywords_list, vec_list

                for value in aclog[lidx - 1:ridx + 100].split("] ["):
                    try:
                        mid, score, bcidx, _query_idx = value.split(",")[:4]
                        mid = mid.split(":")[-1]
                        total_score = score[:4]
                        vec_score = score[-2:]
                        bcidx_value = int(bcidx.split(":")[-1])
                        target = vec_list if bcidx_value in [15, 16, 18, 20] else keywords_list
                        target[mid] = [int(total_score), int(vec_score)]
                    except Exception:
                        traceback.print_exc()
                return keywords_list, vec_list
            except Exception:
                count += 1
                traceback.print_exc()
        return keywords_list, vec_list

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

    def _recall_mids(self, query: str, recall_size: int) -> List[str]:
        keyword_scores, vec_scores = self._func_bs_score(query)
        if not keyword_scores and not vec_scores:
            return []

        v_score_recall = sorted(vec_scores.items(), key=lambda x: x[1][1], reverse=True)[:recall_size]
        t_score_recall = sorted(vec_scores.items(), key=lambda x: x[1][0], reverse=True)[:recall_size]
        w_score_recall = sorted(keyword_scores.items(), key=lambda x: x[1][0], reverse=True)[:recall_size]
        mids = [item[0] for item in v_score_recall]
        mids.extend(item[0] for item in t_score_recall)
        mids.extend(item[0] for item in w_score_recall)
        return unique_keep_order(mids)

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        deduped_queries = unique_keep_order(queries)
        if not deduped_queries:
            return []

        collected: List[KnowledgeChunk] = []
        for query in deduped_queries:
            mids = self._recall_mids(query, recall_size=max(top_k, 5))
            contents = self._get_hbase_batch(mids[: max(top_k, 5)])
            for idx, (mid, content) in enumerate(zip(mids, contents), start=1):
                if not content:
                    continue
                collected.append(
                    KnowledgeChunk(
                        source="weibo_search",
                        title=f"{query}-mid-{mid}",
                        content=content,
                        score=max(top_k - idx + 1, 1),
                    )
                )

        collected.sort(key=lambda item: item.score, reverse=True)
        return collected[:top_k]


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
        raise ValueError(f"不支持的知识源 provider={provider}")

    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        deduped_queries = unique_keep_order(queries)
        if not deduped_queries:
            return []

        merged: List[KnowledgeChunk] = []
        for provider in self.providers:
            merged.extend(self._create_provider(provider).search(deduped_queries, top_k=top_k))
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]
