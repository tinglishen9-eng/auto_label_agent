import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import requests


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def overlap_score(text: str, query_terms: Iterable[str]) -> int:
    haystack = tokenize(text)
    if not haystack:
        return 0
    haystack_set = set(haystack)
    return sum(1 for term in query_terms if term in haystack_set)


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
    def search(self, queries: List[str], top_k: int = 3) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for query in queries[:top_k]:
            try:
                response = requests.get(
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
                chunks.append(
                    KnowledgeChunk(
                        source="duckduckgo",
                        title=query,
                        content=f"检索失败: {exc}",
                        score=0,
                    )
                )
                continue

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
            chunks.append(
                KnowledgeChunk(
                    source="duckduckgo",
                    title=query,
                    content=content,
                    score=max(1, len(parts)),
                )
            )
        return chunks[:top_k]
