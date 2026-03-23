import importlib.util
from pathlib import Path
from typing import Dict, Optional


DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "system_prompts.py"
REQUIRED_PROMPT_KEYS = {"intent", "scoring", "verify"}


def load_prompt_bundle(prompt_file: Optional[str] = None) -> Dict[str, str]:
    target = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    if not target.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {target}")

    spec = importlib.util.spec_from_file_location("auto_label_agent_system_prompts", target)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载 prompt 模块: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = getattr(module, "PROMPTS", None)
    if not isinstance(payload, dict):
        raise ValueError("Prompt 文件内容必须导出 PROMPTS 字典。")

    missing_keys = REQUIRED_PROMPT_KEYS - set(payload.keys())
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"Prompt 文件缺少字段: {missing_text}")

    prompts: Dict[str, str] = {}
    for key in REQUIRED_PROMPT_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Prompt 字段 {key} 必须是非空字符串。")
        prompts[key] = value.strip()
    return prompts
