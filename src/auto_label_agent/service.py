import json
from typing import Any, Callable, Dict, Optional

from auto_label_agent.adapters.knowledge_base import LocalKnowledgeBase
from auto_label_agent.adapters.llm_client import LLMClient
from auto_label_agent.core.pipeline import AutoLabelAgent, AutoLabelResult


def build_local_kb(kb_file: str) -> Optional[LocalKnowledgeBase]:
    if not kb_file:
        return None
    return LocalKnowledgeBase(kb_file)


def create_agent(
    provider: str,
    model: str,
    endpoint: str,
    api_key: str,
    temperature: float,
    kb_file: str,
    online_kb: bool,
    online_kb_provider: str,
    kb_top_k: int,
    max_rounds: int,
    prompt_file: Optional[str],
    progress_callback: Optional[Callable[[str], None]] = None,
    knowledge_callback: Optional[Callable[[str, list], None]] = None,
) -> AutoLabelAgent:
    client = LLMClient(
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        temperature=temperature,
    )
    return AutoLabelAgent(
        llm_client=client,
        local_kb=build_local_kb(kb_file),
        online_kb_enabled=online_kb,
        online_kb_provider=online_kb_provider,
        max_rounds=max_rounds,
        kb_top_k=kb_top_k,
        prompt_file=prompt_file or None,
        progress_callback=progress_callback,
        knowledge_callback=knowledge_callback,
    )


def build_result_payload(
    query: str,
    doc: str,
    result: AutoLabelResult,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "doc": doc,
        "label": result.label,
        "label_name": result.label_name,
        "reason": result.reason,
        "verification_passed": result.verification_passed,
        "final_action": result.final_action,
        "intent": result.intent,
        "score": result.score,
        "verification": result.verification,
        "prompt_version": result.prompt_version,
        "prompt_description": result.prompt_description,
        "prompt_file": result.prompt_file,
        "total_elapsed_ms": result.total_elapsed_ms,
        "total_rounds_used": result.total_rounds_used,
        "used_knowledge": result.used_knowledge,
        "trace": result.trace,
    }
    if extra:
        payload.update(extra)
    return payload


def dumps_json(payload: Dict[str, Any], *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False)
