import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DOC_FIELD_ORDER = [
    "content",
    "abstract",
    "title",
    "keywords",
    "ocr_text",
    "video_voice",
    "ismodal",
]


@dataclass(frozen=True)
class InputRecord:
    row_index: int
    query: str
    doc_text: str
    doc_json: Dict[str, object]
    other_info: Dict[str, object]


def render_doc_text(doc_json: Dict[str, object]) -> str:
    lines: List[str] = []
    for key in DOC_FIELD_ORDER:
        value = doc_json.get(key)
        text = _normalize_value(value)
        if text:
            lines.append(f"{key}: {text}")

    for key, value in doc_json.items():
        if key in DOC_FIELD_ORDER:
            continue
        text = _normalize_value(value)
        if text:
            lines.append(f"{key}: {text}")

    return "\n".join(lines).strip()


def parse_input_file(path: str) -> List[InputRecord]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"输入文件不存在: {target}")

    records: List[InputRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t", 2)
            if len(parts) < 3:
                raise ValueError(f"第 {row_index} 行至少需要 3 列：query、doc_json、other_info_json")

            query = parts[0].strip()
            if not query:
                raise ValueError(f"第 {row_index} 行 query 不能为空")

            doc_json = _load_json_dict(parts[1], row_index=row_index, field_name="doc")
            other_info = _load_json_dict(parts[2], row_index=row_index, field_name="other_info")

            records.append(
                InputRecord(
                    row_index=row_index,
                    query=query,
                    doc_text=render_doc_text(doc_json),
                    doc_json=doc_json,
                    other_info=other_info,
                )
            )
    return records


def _load_json_dict(text: str, row_index: int, field_name: str) -> Dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"第 {row_index} 行 {field_name} 不是合法 JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"第 {row_index} 行 {field_name} 必须是 JSON 对象")
    return payload


def _normalize_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_normalize_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        compact = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return compact.strip()
    return str(value).strip()
