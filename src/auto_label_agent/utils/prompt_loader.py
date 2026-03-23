from pathlib import Path
from typing import Dict, Optional


DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, prompt_dir: Optional[str] = None) -> str:
    base_dir = Path(prompt_dir) if prompt_dir else DEFAULT_PROMPT_DIR
    prompt_path = base_dir / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_prompt_bundle(prompt_dir: Optional[str] = None) -> Dict[str, str]:
    return {
        "intent": load_prompt("intent_system", prompt_dir),
        "scoring": load_prompt("scoring_system", prompt_dir),
        "verify": load_prompt("verify_system", prompt_dir),
    }
