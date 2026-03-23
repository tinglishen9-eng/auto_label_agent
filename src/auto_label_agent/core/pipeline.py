import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from auto_label_agent.adapters.knowledge_base import (
    KnowledgeChunk,
    LocalKnowledgeBase,
    MultiSourceKnowledgeBase,
    OnlineKnowledgeBase,
    WeiboSearchKnowledgeBase,
    unique_keep_order,
)
from auto_label_agent.adapters.llm_client import LLMClient
from auto_label_agent.utils.prompt_loader import load_prompt_bundle


def pretty_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@dataclass
class AgentTrace:
    step: str
    payload: Dict[str, object]


@dataclass
class AutoLabelResult:
    label: int
    label_name: str
    reason: str
    verification_passed: bool
    final_action: str
    intent: Dict[str, object]
    score: Dict[str, object]
    verification: Dict[str, object]
    used_knowledge: List[Dict[str, object]] = field(default_factory=list)
    trace: List[Dict[str, object]] = field(default_factory=list)


class AutoLabelAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        local_kb: Optional[LocalKnowledgeBase] = None,
        online_kb_enabled: bool = False,
        online_kb_provider: str = "duckduck",
        max_rounds: int = 3,
        kb_top_k: int = 3,
        prompt_file: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.local_kb = local_kb
        self.online_kb_enabled = online_kb_enabled
        self.online_kb_provider = online_kb_provider
        self.max_rounds = max_rounds
        self.kb_top_k = kb_top_k
        self.online_kb = self._build_online_kb() if online_kb_enabled else None
        self.prompts = load_prompt_bundle(prompt_file)

    def _build_online_kb(self):
        provider = self.online_kb_provider.strip().lower()
        if provider == "duckduck":
            return OnlineKnowledgeBase()
        if provider == "weibo_search":
            return WeiboSearchKnowledgeBase()
        if provider == "both":
            return MultiSourceKnowledgeBase(["duckduck", "weibo_search"])
        raise ValueError(f"不支持的 online_kb_provider={self.online_kb_provider}")

    def run(self, query: str, doc: str) -> AutoLabelResult:
        trace: List[AgentTrace] = []
        intent_feedback = ""
        score_feedback = ""
        used_knowledge: List[KnowledgeChunk] = []
        latest_intent: Dict[str, object] = {}
        latest_score: Dict[str, object] = {}
        latest_verification: Dict[str, object] = {}

        for round_id in range(1, self.max_rounds + 1):
            latest_intent = self._understand_intent(query, doc, intent_feedback, used_knowledge)
            trace.append(AgentTrace(step=f"round_{round_id}_intent", payload=latest_intent))

            if latest_intent.get("should_use_kb"):
                used_knowledge = self._retrieve_knowledge(query, latest_intent)
                trace.append(
                    AgentTrace(
                        step=f"round_{round_id}_knowledge",
                        payload={"items": [item.to_dict() for item in used_knowledge]},
                    )
                )
                latest_intent = self._understand_intent(query, doc, intent_feedback, used_knowledge)
                trace.append(AgentTrace(step=f"round_{round_id}_intent_refined", payload=latest_intent))

            latest_score = self._score_relevance(query, doc, latest_intent, used_knowledge, score_feedback)
            trace.append(AgentTrace(step=f"round_{round_id}_score", payload=latest_score))

            latest_verification = self._verify(query, doc, latest_intent, latest_score, used_knowledge)
            trace.append(AgentTrace(step=f"round_{round_id}_verify", payload=latest_verification))

            if latest_verification.get("passed") and latest_verification.get("final_action") == "finish":
                break

            action = latest_verification.get("final_action") or "rescore"
            issues = latest_verification.get("issues") or []
            reasons = "；".join(
                issue.get("reason", "") for issue in issues if isinstance(issue, dict) and issue.get("reason")
            )
            if action == "revisit_intent":
                intent_feedback = f"质检未通过，请重点修正这些理解问题：{reasons}"
                continue
            if action == "retrieve_kb":
                intent_feedback = f"质检认为需求理解仍需补知识：{reasons}"
                latest_intent["should_use_kb"] = True
                used_knowledge = self._retrieve_knowledge(query, latest_intent, issues)
                trace.append(
                    AgentTrace(
                        step=f"round_{round_id}_knowledge_retry",
                        payload={"items": [item.to_dict() for item in used_knowledge]},
                    )
                )
                continue
            score_feedback = f"质检未通过，请重新审视标签，问题如下：{reasons}"

        return AutoLabelResult(
            label=int(latest_score.get("label", 0)),
            label_name=str(latest_score.get("label_name") or ""),
            reason=str(latest_score.get("reason") or ""),
            verification_passed=bool(latest_verification.get("passed")),
            final_action=str(latest_verification.get("final_action") or "finish"),
            intent=latest_intent,
            score=latest_score,
            verification=latest_verification,
            used_knowledge=[item.to_dict() for item in used_knowledge],
            trace=[{"step": item.step, "payload": item.payload} for item in trace],
        )

    def _understand_intent(
        self,
        query: str,
        doc: str,
        feedback: str,
        knowledge: List[KnowledgeChunk],
    ) -> Dict[str, object]:
        user_prompt = f"""请分析下面的标注样本。

query:
{query}

doc:
{doc}

已有补充知识:
{self._render_knowledge(knowledge)}

额外反馈:
{feedback or '无'}
"""
        return self.llm_client.complete_json(self.prompts["intent"], user_prompt)

    def _retrieve_knowledge(
        self,
        query: str,
        intent: Dict[str, object],
        issues: Optional[List[Dict[str, object]]] = None,
    ) -> List[KnowledgeChunk]:
        queries = [query]
        queries.extend(intent.get("kb_queries") or [])
        for issue in issues or []:
            reason = issue.get("reason")
            if reason:
                queries.append(str(reason))
        queries = unique_keep_order(queries)

        collected: List[KnowledgeChunk] = []
        if self.local_kb is not None:
            collected.extend(self.local_kb.search(queries, top_k=self.kb_top_k))
        if self.online_kb is not None:
            collected.extend(self.online_kb.search(queries, top_k=self.kb_top_k))

        collected.sort(key=lambda item: item.score, reverse=True)
        return collected[: self.kb_top_k]

    def _score_relevance(
        self,
        query: str,
        doc: str,
        intent: Dict[str, object],
        knowledge: List[KnowledgeChunk],
        feedback: str,
    ) -> Dict[str, object]:
        user_prompt = f"""请基于以下信息进行相关性标注。

query:
{query}

需求理解:
{pretty_json(intent)}

补充知识:
{self._render_knowledge(knowledge)}

doc:
{doc}

额外反馈:
{feedback or '无'}
"""
        return self.llm_client.complete_json(self.prompts["scoring"], user_prompt)

    def _verify(
        self,
        query: str,
        doc: str,
        intent: Dict[str, object],
        score: Dict[str, object],
        knowledge: List[KnowledgeChunk],
    ) -> Dict[str, object]:
        user_prompt = f"""请质检下面这次标注。

query:
{query}

需求理解:
{pretty_json(intent)}

补充知识:
{self._render_knowledge(knowledge)}

doc:
{doc}

当前标签结果:
{pretty_json(score)}
"""
        return self.llm_client.complete_json(self.prompts["verify"], user_prompt)

    def _render_knowledge(self, knowledge: List[KnowledgeChunk]) -> str:
        if not knowledge:
            return "无"
        lines: List[str] = []
        for idx, item in enumerate(knowledge, start=1):
            lines.append(
                f"[{idx}] source={item.source}; title={item.title}; score={item.score}\\n{item.content}"
            )
        return "\\n\\n".join(lines)
