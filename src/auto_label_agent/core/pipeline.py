import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from auto_label_agent.adapters.knowledge_base import (
    KnowledgeChunk,
    LocalKnowledgeBase,
    MultiSourceKnowledgeBase,
    OnlineKnowledgeBase,
    WeiboLLMKnowledgeBase,
    WeiboSearchKnowledgeBase,
    unique_keep_order,
)
from auto_label_agent.adapters.account_intent import AccountIntentDetector, AccountIntentSignal
from auto_label_agent.adapters.llm_client import LLMClient
from auto_label_agent.utils.prompt_loader import load_prompt_bundle

logger = logging.getLogger(__name__)
ACCOUNT_PATTERN = re.compile(r"@([A-Za-z0-9_\u4e00-\u9fff]+)")
CJK_TEXT_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9_]+$")


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
    prompt_version: str
    prompt_description: str
    prompt_file: str
    total_elapsed_ms: int = 0
    total_rounds_used: int = 0
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
        progress_callback: Optional[Callable[[str], None]] = None,
        knowledge_callback: Optional[Callable[[str, List[KnowledgeChunk]], None]] = None,
        account_intent_detector: Optional[AccountIntentDetector] = None,
    ):
        self.llm_client = llm_client
        self.local_kb = local_kb
        self.online_kb_enabled = online_kb_enabled
        self.online_kb_provider = online_kb_provider
        self.max_rounds = max_rounds
        self.kb_top_k = kb_top_k
        self.progress_callback = progress_callback
        self.knowledge_callback = knowledge_callback
        self.account_intent_detector = account_intent_detector or AccountIntentDetector()
        self.online_kb = self._build_online_kb() if online_kb_enabled else None
        self.prompt_bundle = load_prompt_bundle(prompt_file)
        self.prompts = self.prompt_bundle
        logger.info(
            "加载 prompt 包: version=%s, description=%s",
            self.prompt_bundle.version,
            self.prompt_bundle.description,
        )
        logger.debug("prompt_file=%s", self.prompt_bundle.prompt_file)

    def _build_online_kb(self):
        provider = self.online_kb_provider.strip().lower()
        logger.debug("初始化在线知识源: %s", provider)
        if provider == "duckduck":
            return OnlineKnowledgeBase()
        if provider == "weibo_search":
            return WeiboSearchKnowledgeBase()
        if provider == "weibo_llm":
            return WeiboLLMKnowledgeBase()
        if provider in {"both", "all"}:
            return MultiSourceKnowledgeBase(["duckduck", "weibo_search", "weibo_llm"])
        raise ValueError(f"不支持的 online_kb_provider={self.online_kb_provider}")

    def run(self, query: str, doc: str) -> AutoLabelResult:
        start_time = time.perf_counter()
        trace: List[AgentTrace] = []
        intent_feedback = ""
        score_feedback = ""
        used_knowledge: List[KnowledgeChunk] = []
        latest_intent: Dict[str, object] = {}
        latest_score: Dict[str, object] = {}
        latest_verification: Dict[str, object] = {}
        account_signal = self._detect_account_intent(query)
        trace.append(AgentTrace(step="account_intent", payload=account_signal.to_dict()))
        rounds_used = 0

        for round_id in range(1, self.max_rounds + 1):
            rounds_used = round_id
            self._update_progress("账号意图判断中", round_id=round_id, include_round=False)
            self._update_progress("需求理解中", round_id=round_id)
            logger.info("[%d/%d] 需求理解", round_id, self.max_rounds)
            latest_intent = self._understand_intent(query, intent_feedback, used_knowledge, account_signal)
            trace.append(AgentTrace(step=f"round_{round_id}_intent", payload=latest_intent))
            logger.debug("intent=%s", pretty_json(latest_intent))

            if latest_intent.get("should_use_kb"):
                self._update_progress("补充知识检索中", round_id=round_id)
                logger.info("[%d/%d] 补充知识检索", round_id, self.max_rounds)
                used_knowledge = self._retrieve_knowledge(query, latest_intent)
                self._emit_knowledge("knowledge", used_knowledge)
                logger.info("检索到 %d 条补充知识", len(used_knowledge))
                trace.append(
                    AgentTrace(
                        step=f"round_{round_id}_knowledge",
                        payload={"items": [item.to_dict() for item in used_knowledge]},
                    )
                )
                self._update_progress("需求理解中", round_id=round_id)
                logger.info("[%d/%d] 基于补充知识重新理解需求", round_id, self.max_rounds)
                latest_intent = self._understand_intent(query, intent_feedback, used_knowledge, account_signal)
                trace.append(AgentTrace(step=f"round_{round_id}_intent_refined", payload=latest_intent))
                logger.debug("intent_refined=%s", pretty_json(latest_intent))

            self._update_progress("相关性判断中", round_id=round_id)
            logger.info("[%d/%d] 相关性打分", round_id, self.max_rounds)
            latest_score = self._score_relevance(query, doc, latest_intent, used_knowledge, score_feedback)
            trace.append(AgentTrace(step=f"round_{round_id}_score", payload=latest_score))
            logger.info("当前标签: %s", latest_score.get("label"))
            logger.debug("score=%s", pretty_json(latest_score))

            self._update_progress("相关性验证中", round_id=round_id)
            logger.info("[%d/%d] 二次验证", round_id, self.max_rounds)
            latest_verification = self._verify(query, doc, latest_intent, latest_score, used_knowledge)
            trace.append(AgentTrace(step=f"round_{round_id}_verify", payload=latest_verification))
            logger.debug("verification=%s", pretty_json(latest_verification))

            if latest_verification.get("passed") and latest_verification.get("final_action") == "finish":
                self._update_progress("标注完成", round_id=round_id)
                logger.info("验证通过，流程结束")
                break

            action = latest_verification.get("final_action") or "rescore"
            issues = latest_verification.get("issues") or []
            reasons = "；".join(
                issue.get("reason", "") for issue in issues if isinstance(issue, dict) and issue.get("reason")
            )
            logger.warning("验证未通过: action=%s, reasons=%s", action, reasons or "<none>")
            if action == "revisit_intent":
                self._update_progress("重新理解需求中", round_id=round_id)
                intent_feedback = f"质检未通过，请重点修正这些理解问题：{reasons}"
                continue
            if action == "retrieve_kb":
                self._update_progress("补充知识检索中", round_id=round_id)
                intent_feedback = f"质检认为需求理解仍需补知识：{reasons}"
                latest_intent["should_use_kb"] = True
                used_knowledge = self._retrieve_knowledge(query, latest_intent, issues)
                self._emit_knowledge("knowledge_retry", used_knowledge)
                trace.append(
                    AgentTrace(
                        step=f"round_{round_id}_knowledge_retry",
                        payload={"items": [item.to_dict() for item in used_knowledge]},
                    )
                )
                logger.info("重检索后得到 %d 条补充知识", len(used_knowledge))
                continue
            self._update_progress("重新判断相关性中", round_id=round_id)
            score_feedback = f"质检未通过，请重新审视标签，问题如下：{reasons}"

        total_elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return AutoLabelResult(
            label=int(latest_score.get("label", 0)),
            label_name=str(latest_score.get("label_name") or ""),
            reason=str(latest_score.get("reason") or ""),
            verification_passed=bool(latest_verification.get("passed")),
            final_action=str(latest_verification.get("final_action") or "finish"),
            intent=latest_intent,
            score=latest_score,
            verification=latest_verification,
            prompt_version=self.prompt_bundle.version,
            prompt_description=self.prompt_bundle.description,
            prompt_file=self.prompt_bundle.prompt_file,
            total_elapsed_ms=total_elapsed_ms,
            total_rounds_used=rounds_used,
            used_knowledge=[item.to_dict() for item in used_knowledge],
            trace=[
                {
                    "step": "prompt_bundle",
                    "payload": {
                        "version": self.prompt_bundle.version,
                        "description": self.prompt_bundle.description,
                        "prompt_file": self.prompt_bundle.prompt_file,
                        "few_shot_counts": {
                            name: self.prompt_bundle.few_shot_count(name) for name in ("intent", "scoring", "verify")
                        },
                    },
                }
            ]
            + [{"step": item.step, "payload": item.payload} for item in trace],
        )

    def _update_progress(self, message: str, round_id: Optional[int] = None, include_round: bool = True) -> None:
        if self.progress_callback is not None:
            if include_round and round_id is not None:
                self.progress_callback(f"第{round_id}轮 - {message}")
                return
            self.progress_callback(message)

    def _emit_knowledge(self, stage: str, knowledge: List[KnowledgeChunk]) -> None:
        if self.knowledge_callback is not None:
            self.knowledge_callback(stage, knowledge)

    def _understand_intent(
        self,
        query: str,
        feedback: str,
        knowledge: List[KnowledgeChunk],
        account_signal: Optional[AccountIntentSignal] = None,
    ) -> Dict[str, object]:
        account_signal = account_signal or AccountIntentSignal(query=query)
        user_prompt = f"""请分析下面的 query 需求。

query:
{query}

账号意图判断:
{pretty_json(account_signal.to_dict())}

已有补充知识:
{self._render_knowledge(knowledge)}

额外反馈:
{feedback or '无'}
"""
        intent = self.llm_client.complete_json(self.prompt_bundle.render("intent"), user_prompt)
        intent = self._apply_account_signal(query, intent, account_signal)
        return self._apply_account_need_heuristic(query, intent, knowledge)

    def _detect_account_intent(self, query: str) -> AccountIntentSignal:
        self._update_progress("账号意图判断中")
        try:
            signal = self.account_intent_detector.detect(query)
            logger.info(
                "账号意图判断完成: matched=%s, intent_type=%s, exact_screen_name=%s",
                signal.matched,
                signal.intent_type,
                signal.exact_screen_name,
            )
            return signal
        except Exception as exc:
            logger.warning("账号意图判断失败: %s", exc)
            return AccountIntentSignal(query=query, reason=f"账号意图判断失败: {exc}")

    def _retrieve_knowledge(
        self,
        query: str,
        intent: Dict[str, object],
        issues: Optional[List[Dict[str, object]]] = None,
    ) -> List[KnowledgeChunk]:
        queries = self._build_kb_queries(query, intent)
        for issue in issues or []:
            reason = issue.get("reason")
            if reason:
                queries.append(str(reason))
        queries = unique_keep_order(queries)
        logger.debug("knowledge_queries=%s", queries)

        collected: List[KnowledgeChunk] = []
        if self.local_kb is not None:
            local_results = self.local_kb.search(queries, top_k=self.kb_top_k)
            collected.extend(local_results)
            logger.info("本地知识库返回 %d 条结果", len(local_results))
        if self.online_kb is not None:
            online_results = self.online_kb.search(queries, top_k=self.kb_top_k)
            collected.extend(online_results)
            logger.info("在线知识源返回 %d 条结果", len(online_results))

        collected.sort(key=lambda item: item.score, reverse=True)
        return collected[: self.kb_top_k]

    def _build_kb_queries(self, query: str, intent: Dict[str, object]) -> List[str]:
        queries = [query]
        queries.extend(intent.get("kb_queries") or [])

        query_text = (query or "").strip()
        if not self._should_expand_kb_queries(query_text):
            return unique_keep_order(queries)

        for field in ("entities", "qualifiers"):
            values = intent.get(field) or []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                text = str(value or "").strip()
                if not text:
                    continue
                if text.startswith("@") and len(text) > 1:
                    queries.append(text[1:])
                queries.append(text)

        return unique_keep_order(queries)

    def _should_expand_kb_queries(self, query: str) -> bool:
        if not query or len(query) < 4:
            return False
        if len(query) > 24:
            return False
        if re.search(r"\s", query):
            return False
        if not CJK_TEXT_PATTERN.fullmatch(query):
            return False
        # 粘连 query 往往缺少显式分隔，补充实体级查询可提升召回。
        return True

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
        return self.llm_client.complete_json(self.prompt_bundle.render("scoring"), user_prompt)

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
        return self.llm_client.complete_json(self.prompt_bundle.render("verify"), user_prompt)

    def _render_knowledge(self, knowledge: List[KnowledgeChunk]) -> str:
        if not knowledge:
            return "无"
        lines: List[str] = []
        for idx, item in enumerate(knowledge, start=1):
            lines.append(
                f"[{idx}] source={item.source}; title={item.title}; score={item.score}\\n{item.content}"
            )
        return "\\n\\n".join(lines)

    def _apply_account_need_heuristic(
        self,
        query: str,
        intent: Dict[str, object],
        knowledge: List[KnowledgeChunk],
    ) -> Dict[str, object]:
        if not isinstance(intent, dict) or not knowledge:
            return intent

        query_text = (query or "").strip().lower()
        if not query_text:
            return intent

        matched_accounts: List[str] = []
        for item in knowledge:
            for candidate in ACCOUNT_PATTERN.findall(f"{item.title}\n{item.content}"):
                account_name = candidate.strip()
                if not account_name:
                    continue
                normalized = account_name.lower()
                if normalized in query_text or query_text in normalized:
                    matched_accounts.append(account_name)

        matched_accounts = unique_keep_order(matched_accounts)
        if not matched_accounts:
            return intent

        entities = intent.get("entities") or []
        if not isinstance(entities, list):
            entities = [str(entities)]
        for account_name in matched_accounts:
            account_entity = f"@{account_name}"
            if account_entity not in entities:
                entities.append(account_entity)
        query_account_entity = f"@{query.strip()}"
        if query.strip() and query_account_entity not in entities:
            entities.append(query_account_entity)
        intent["entities"] = entities

        constraints = intent.get("constraints") or []
        if not isinstance(constraints, list):
            constraints = [str(constraints)]
        account_constraint = "query 可能是账号搜索需求，应优先按社交平台账号理解"
        if account_constraint not in constraints:
            constraints.append(account_constraint)
        intent["constraints"] = constraints

        intent["account_search_detected"] = True
        intent["matched_account_handles"] = [f"@{name}" for name in matched_accounts]

        notes = str(intent.get("notes") or "").strip()
        heuristic_note = f"补充知识命中账号 {', '.join(f'@{name}' for name in matched_accounts)}，query 可能是账号搜索需求。"
        if heuristic_note not in notes:
            intent["notes"] = f"{notes} {heuristic_note}".strip() if notes else heuristic_note

        return intent

    def _apply_account_signal(
        self,
        query: str,
        intent: Dict[str, object],
        account_signal: AccountIntentSignal,
    ) -> Dict[str, object]:
        if not isinstance(intent, dict) or not account_signal.matched:
            return intent

        intent["account_intent_detected"] = True
        intent["account_intent_type"] = account_signal.intent_type
        intent["account_exact_screen_name"] = account_signal.exact_screen_name
        intent["account_candidates"] = account_signal.candidate_screen_names

        entities = intent.get("entities") or []
        if not isinstance(entities, list):
            entities = [str(entities)]
        matched_entity = f"@{account_signal.exact_screen_name}"
        if matched_entity not in entities:
            entities.append(matched_entity)
        intent["entities"] = entities

        constraints = intent.get("constraints") or []
        if not isinstance(constraints, list):
            constraints = [str(constraints)]
        constraint = "账号搜索意图存在，但它只是候选意图之一，需要与普通语义意图并行考虑"
        if constraint not in constraints:
            constraints.append(constraint)
        intent["constraints"] = constraints

        notes = str(intent.get("notes") or "").strip()
        if account_signal.reason and account_signal.reason not in notes:
            intent["notes"] = f"{notes} {account_signal.reason}".strip() if notes else account_signal.reason

        candidate_types = intent.get("candidate_target_types") or []
        if not isinstance(candidate_types, list):
            candidate_types = [str(candidate_types)]
        if "account" not in candidate_types:
            candidate_types.append("account")
        intent["candidate_target_types"] = candidate_types

        if query.strip():
            query_handle = f"@{query.strip()}"
            if query_handle not in intent["entities"]:
                intent["entities"].append(query_handle)
        return intent
