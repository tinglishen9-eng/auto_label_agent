import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "system_prompts.py"
REQUIRED_PROMPT_KEYS = {"intent", "scoring", "verify"}


@dataclass(frozen=True)
class FewShotExample:
    input_text: str
    output_text: str
    title: str = ""


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system_prompt: str
    few_shot_examples: List[FewShotExample] = field(default_factory=list)

    def render(self) -> str:
        sections = [self.system_prompt.strip()]
        if self.few_shot_examples:
            example_blocks: List[str] = []
            for index, example in enumerate(self.few_shot_examples, start=1):
                header = example.title.strip() if example.title.strip() else f"示例{index}"
                example_blocks.append(
                    "\n".join(
                        [
                            f"### {header}",
                            "[输入]",
                            example.input_text.strip(),
                            "[输出]",
                            example.output_text.strip(),
                        ]
                    )
                )
            sections.append("以下是 few-shot 示例，请严格参考其输出风格和字段要求：\n\n" + "\n\n".join(example_blocks))
        return "\n\n".join(part for part in sections if part.strip())


@dataclass(frozen=True)
class PromptBundle:
    version: str
    description: str
    prompt_file: str
    templates: Dict[str, PromptTemplate]

    def __getitem__(self, key: str) -> str:
        return self.templates[key].render()

    def render(self, key: str) -> str:
        return self.templates[key].render()

    def few_shot_count(self, key: str) -> int:
        return len(self.templates[key].few_shot_examples)


def _load_module(target: Path):
    spec = importlib.util.spec_from_file_location("auto_label_agent_system_prompts", target)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载 prompt 模块: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_prompt_template(key: str, value: object) -> PromptTemplate:
    if isinstance(value, str):
        system_prompt = value.strip()
        if not system_prompt:
            raise ValueError(f"Prompt 字段 {key} 必须是非空字符串。")
        return PromptTemplate(name=key, system_prompt=system_prompt)

    if not isinstance(value, dict):
        raise ValueError(f"Prompt 字段 {key} 必须是字符串或字典。")

    system_prompt = str(value.get("system_prompt") or value.get("system") or "").strip()
    if not system_prompt:
        raise ValueError(f"Prompt 字段 {key}.system_prompt 必须是非空字符串。")

    raw_examples = value.get("few_shot_examples") or value.get("examples") or []
    if not isinstance(raw_examples, list):
        raise ValueError(f"Prompt 字段 {key}.few_shot_examples 必须是数组。")

    examples: List[FewShotExample] = []
    for index, item in enumerate(raw_examples, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt 字段 {key}.few_shot_examples[{index}] 必须是对象。")
        input_text = str(item.get("input") or item.get("input_text") or "").strip()
        output_text = str(item.get("output") or item.get("output_text") or "").strip()
        if not input_text or not output_text:
            raise ValueError(
                f"Prompt 字段 {key}.few_shot_examples[{index}] 需要包含非空的 input 和 output。"
            )
        examples.append(
            FewShotExample(
                title=str(item.get("title") or "").strip(),
                input_text=input_text,
                output_text=output_text,
            )
        )
    return PromptTemplate(name=key, system_prompt=system_prompt, few_shot_examples=examples)


def load_prompt_bundle(prompt_file: Optional[str] = None) -> PromptBundle:
    target = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    if not target.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {target}")

    module = _load_module(target)
    payload = getattr(module, "PROMPTS", None)
    if not isinstance(payload, dict):
        raise ValueError("Prompt 文件内容必须导出 PROMPTS 字典。")

    missing_keys = REQUIRED_PROMPT_KEYS - set(payload.keys())
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"Prompt 文件缺少字段: {missing_text}")

    templates: Dict[str, PromptTemplate] = {}
    for key in REQUIRED_PROMPT_KEYS:
        templates[key] = _normalize_prompt_template(key, payload.get(key))

    version = str(getattr(module, "PROMPT_VERSION", payload.get("version") or "v1")).strip()
    description = str(
        getattr(module, "PROMPT_DESCRIPTION", payload.get("description") or "default prompt bundle")
    ).strip()

    return PromptBundle(
        version=version,
        description=description,
        prompt_file=str(target.resolve()),
        templates=templates,
    )
