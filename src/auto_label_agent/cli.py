import json
import os
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from auto_label_agent.adapters.knowledge_base import LocalKnowledgeBase
from auto_label_agent.adapters.llm_client import LLMClient, resolve_api_key, resolve_endpoint
from auto_label_agent.core.pipeline import AutoLabelAgent

load_dotenv()


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
@click.option("--provider", default="deepseek", help="模型提供方，例如 deepseek/openai")
@click.option("--model", default="deepseek-chat", help="模型名称")
@click.option("--endpoint", default="", help="聊天 API 地址，不填则按 provider 自动选择")
@click.option("--temperature", default=0.1, type=float, help="采样温度")
@click.option("--query", default="", help="待标注 query 文本")
@click.option("--query-file", default="", help="从文件读取 query")
@click.option("--doc", default="", help="待判断相关性的 doc 文本")
@click.option("--doc-file", default="", help="从文件读取 doc")
@click.option("--kb-file", default="", help="本地知识库文件，支持 json/jsonl/txt")
@click.option("--online-kb/--no-online-kb", default=False, help="是否允许联网补充知识")
@click.option("--kb-top-k", default=3, type=int, help="最多保留多少条补充知识")
@click.option("--max-rounds", default=3, type=int, help="最多自校验迭代轮数")
@click.option("--prompt-dir", default="", help="自定义 prompt 目录，目录内需包含 3 个 system prompt 文件")
@click.option("--output", default="", help="将最终结果写入 JSON 文件")
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
    kb_top_k: int,
    max_rounds: int,
    prompt_dir: str,
    output: str,
) -> None:
    query_text = read_text(query, query_file)
    doc_text = read_text(doc, doc_file)

    if not query_text:
        raise click.ClickException("query 不能为空，请使用 --query 或 --query-file。")
    if not doc_text:
        raise click.ClickException("doc 不能为空，请使用 --doc 或 --doc-file。")

    api_key = resolve_api_key(provider)
    if not api_key:
        raise click.ClickException(
            f"缺少 API Key，请在环境变量或 .env 中配置 {provider.upper()}_API_KEY。"
        )

    client = LLMClient(
        provider=provider,
        model=model,
        endpoint=resolve_endpoint(provider, endpoint),
        api_key=api_key,
        temperature=temperature,
    )
    agent = AutoLabelAgent(
        llm_client=client,
        local_kb=build_local_kb(kb_file),
        online_kb_enabled=online_kb,
        max_rounds=max_rounds,
        kb_top_k=kb_top_k,
        prompt_dir=prompt_dir or None,
    )

    result = agent.run(query=query_text, doc=doc_text)
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
        click.echo(f"\n结果已写入: {os.path.abspath(output)}")
