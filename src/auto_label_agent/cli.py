import json
import logging
import os
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from auto_label_agent.adapters.llm_client import (
    provider_requires_api_key,
    resolve_api_key,
    resolve_endpoint,
)
from auto_label_agent.core.pipeline import AutoLabelAgent
from auto_label_agent.service import build_result_payload, create_agent, dumps_json
from auto_label_agent.utils.input_parser import parse_input_file

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


def log_knowledge_details(stage: str, knowledge: list) -> None:
    if not knowledge:
        logger.info("外部知识明细[%s]: 无结果", stage)
        return
    logger.info("外部知识明细[%s]: %d 条", stage, len(knowledge))
    for index, item in enumerate(knowledge, start=1):
        source = getattr(item, "source", "")
        title = getattr(item, "title", "")
        score = getattr(item, "score", 0)
        content = str(getattr(item, "content", "") or "").replace("\n", " ").strip()
        if len(content) > 200:
            content = content[:200] + "..."
        logger.info(
            "  [%d] source=%s | score=%s | title=%s | content=%s",
            index,
            source,
            score,
            title,
            content,
        )


@click.command(help="自动标注 Agent：先做需求理解，再打相关性标签并进行二次验证")
@click.option("--provider", default="deepseek", help="模型提供方，例如 deepseek/openai/kimi/mproxy")
@click.option("--model", default="deepseek-chat", help="模型名称，例如 deepseek-chat/gpt-4.1/moonshot-v1-8k/qwen272b")
@click.option("--endpoint", default="", help="聊天 API 地址，不填则按 provider 自动选择")
@click.option("--temperature", default=0.1, type=float, help="采样温度")
@click.option("--input-file", default="", help="批量输入文件，按列依次为 query、doc_json、other_info_json")
@click.option("--query", default="", help="待标注 query 文本")
@click.option("--query-file", default="", help="从文件读取 query")
@click.option("--doc", default="", help="待判断相关性的 doc 文本")
@click.option("--doc-file", default="", help="从文件读取 doc")
@click.option("--kb-file", default="", help="本地知识库文件，支持 json/jsonl/txt")
@click.option("--online-kb/--no-online-kb", default=False, help="是否允许联网补充知识")
@click.option(
    "--online-kb-provider",
    default="duckduck",
    type=click.Choice(["duckduck", "weibo_search", "weibo_llm", "both", "all"], case_sensitive=False),
    help="在线知识源，支持 duckduck / weibo_search / weibo_llm / both / all",
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
    input_file: str,
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
    logger.debug(
        "endpoint=%s, prompt_file=%s, output=%s, has_api_key=%s",
        resolved_endpoint,
        prompt_file or "<default>",
        output or "<stdout>",
        bool(api_key),
    )

    agent = create_agent(
        provider=provider,
        model=model,
        endpoint=resolved_endpoint,
        api_key=api_key,
        temperature=temperature,
        kb_file=kb_file,
        online_kb=online_kb,
        online_kb_provider=online_kb_provider,
        kb_top_k=kb_top_k,
        max_rounds=max_rounds,
        prompt_file=prompt_file or None,
        knowledge_callback=log_knowledge_details,
    )

    if input_file:
        records = parse_input_file(input_file)
        logger.info("开始执行批量标注流程: records=%d", len(records))
        payloads = []
        for index, record in enumerate(records, start=1):
            logger.info("[%d/%d] 处理 row=%d, mid=%s", index, len(records), record.row_index, record.other_info.get("mid", ""))
            result = agent.run(query=record.query, doc=record.doc_text)
            payloads.append(
                build_result_payload(
                    query=record.query,
                    doc=record.doc_text,
                    result=result,
                    extra={
                        "row_index": record.row_index,
                        "doc_json": record.doc_json,
                        "other_info": record.other_info,
                        "mid": record.other_info.get("mid"),
                    },
                )
            )

        output_text = "\n".join(dumps_json(item) for item in payloads)
        click.echo(output_text)
        if output:
            Path(output).write_text(output_text + ("\n" if output_text else ""), encoding="utf-8")
            logger.info("批量结果已写入 %s", os.path.abspath(output))
            click.echo(f"\n结果已写入: {os.path.abspath(output)}")
        return

    query_text = read_text(query, query_file)
    doc_text = read_text(doc, doc_file)

    if not query_text:
        raise click.ClickException("query 不能为空，请使用 --query、--query-file 或 --input-file。")
    if not doc_text:
        raise click.ClickException("doc 不能为空，请使用 --doc 或 --doc-file。")

    logger.info("样本长度: query=%d, doc=%d", len(query_text), len(doc_text))
    logger.info("开始执行标注流程")
    result = agent.run(query=query_text, doc=doc_text)
    logger.info(
        "标注完成: label=%s, verification_passed=%s, final_action=%s",
        result.label,
        result.verification_passed,
        result.final_action,
    )
    payload = build_result_payload(query=query_text, doc=doc_text, result=result)
    output_text = dumps_json(payload, pretty=True)
    click.echo(output_text)

    if output:
        Path(output).write_text(output_text, encoding="utf-8")
        logger.info("结果已写入 %s", os.path.abspath(output))
        click.echo(f"\n结果已写入: {os.path.abspath(output)}")
