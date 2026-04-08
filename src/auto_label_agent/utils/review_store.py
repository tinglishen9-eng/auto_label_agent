import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def append_review_record(output_file: str, payload: Dict[str, Any]) -> str:
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    record = dict(payload)
    record["review_saved_at"] = datetime.now().isoformat(timespec="seconds")

    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(target.resolve())
