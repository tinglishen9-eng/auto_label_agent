import json
import logging
import os
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from auto_label_agent.adapters.knowledge_base import LocalKnowledgeBase
from auto_label_agent.adapters.llm_client import (
    LLMClient,
    provider_requires_api_key,
    resolve_api_key,
    resolve_endpoint,
)
from auto_label_agent.core.pipeline import AutoLabelAgent

load_dotenv()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def read_text(text: str, file_path: str) -> str:
    if text:
        return text.strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return ""


def build_local_kb(kb_file: str) -> Optional[LocalKnowledgeBase]:
    if not kb_file:
        return None
    return LocalKnowledgeBase(kb_file)


@click.command(help="自动标注 Agent：先做需求理解，再打相关性标签并进行二次验证")
@click.option("--provider", default="deepseek", help="模型提供方，例如 deepseek/openai/kimi/mproxy")
@click.option("--model", default="deepseek-chat", help="模型名称，例如 deepseek-chat/gpt-4.1/moonshot-v1-8k/qwen272b")
@click.option("--endpoint", default="", help="聊天 API 地址，不填则按 provider 自动选择")
@click.option("--temperature", default=0.1, type=float, help="采样温度")
@click.option("--query", default="", help="待标注 query 文本")
@click.option("--query-file", default="", help="从文件读取 query")
@click.option("--doc", default="", help="待判断相关性的 doc 文本")
@click.option("--doc-file", default="", help="从文件读取 doc")
@click.option("--kb-file", default="", help="本地知识库文件，支持 json/jsonl/txt")
@click.option("--online-kb/--no-online-kb", default=False, help="是否允许联网补充知识")
@click.option(
    "--online-kb-provider",
    default="duckduck",
    type=click.Choice(["duckduck", "weibo_search", "both"], case_sensitive=False),
    help="在线知识源，支持 duckduck / weibo_search / both",
)
@click.option("--kb-top-k", default=3, type=int, help="最多保留多少条补充知识")
@click.option("--max-rounds", default=3, type=int, help="最多自校验迭代轮数")
@click.option("--prompt-file", default="", help="自定义单文件 prompt 配置，需包含 intent/scoring/verify 3 个字段")
@click.option("--output", default="", help="将最终结果写入 JSON 文件")
@click.option("--verbose/--no-verbose", default=False, help="是否输出更详细的调试日志")
def main(
    provider: str,
    model: str,
    endpoint: str,
    temperature: float,
    query: str,
    query_file: str,
    doc: str,
    doc_file: str,
    kb_file: str,
    online_kb: bool,
    online_kb_provider: str,
    kb_top_k: int,
    max_rounds: int,
    prompt_file: str,
    output: str,
    verbose: bool,
) -> None:
    setup_logging(verbose)
    logger.info("启动自动标注 Agent")

    query_text = read_text(query, query_file)
    doc_text = read_text(doc, doc_file)

    if not query_text:
        raise click.ClickException("query 不能为空，请使用 --query 或 --query-file。")
    if not doc_text:
        raise click.ClickException("doc 不能为空，请使用 --doc 或 --doc-file。")

    api_key = resolve_api_key(provider)
    if provider_requires_api_key(provider) and not api_key:
        raise click.ClickException(
            f"缺少 API Key，请在环境变量或 .env 中配置 {provider.upper()}_API_KEY。"
        )

    resolved_endpoint = resolve_endpoint(provider, endpoint)
    logger.info(
        "运行配置: provider=%s, model=%s, online_kb=%s, online_kb_provider=%s, local_kb=%s",
        provider,
        model,
        online_kb,
        online_kb_provider,
        bool(kb_file),
    )
    logger.debug("endpoint=%s, prompt_file=%s, output=%s", resolved_endpoint, prompt_file or "<default>", output or "<stdout>")
    logger.info("样本长度: query=%d, doc=%d", len(query_text), len(doc_text))

    client = LLMClient(
        provider=provider,
        model=model,
        endpoint=resolved_endpoint,
        api_key=api_key,
        temperature=temperature,
    )
    agent = AutoLabelAgent(
        llm_client=client,
        local_kb=build_local_kb(kb_file),
        online_kb_enabled=online_kb,
        online_kb_provider=online_kb_provider,
        max_rounds=max_rounds,
        kb_top_k=kb_top_k,
        prompt_file=prompt_file or None,
    )

    logger.info("开始执行标注流程")
    result = agent.run(query=query_text, doc=doc_text)
    logger.info(
        "标注完成: label=%s, verification_passed=%s, final_action=%s",
        result.label,
        result.verification_passed,
        result.final_action,
    )
    payload = {
        "query": query_text,
        "doc": doc_text,
        "label": result.label,
        "label_name": result.label_name,
        "reason": result.reason,
        "verification_passed": result.verification_passed,
        "final_action": result.final_action,
        "intent": result.intent,
        "score": result.score,
        "verification": result.verification,
        "used_knowledge": result.used_knowledge,
        "trace": result.trace,
    }

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    click.echo(output_text)

    if output:
        Path(output).write_text(output_text, encoding="utf-8")
        logger.info("结果已写入 %s", os.path.abspath(output))
        click.echo(f"\n结果已写入: {os.path.abspath(output)}")
