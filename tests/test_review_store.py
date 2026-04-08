import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.utils.review_store import append_review_record


class ReviewStoreTest(unittest.TestCase):
    def test_append_review_record_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "reviews.jsonl"
            saved_path = append_review_record(
                str(output_file),
                {
                    "source": {"row_index": 1, "mid": "123"},
                    "auto_result": {"label": 2},
                    "review": {"manual_override": True, "manual_label": "3"},
                },
            )
            self.assertEqual(Path(saved_path), output_file.resolve())
            lines = output_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["source"]["mid"], "123")
            self.assertEqual(payload["review"]["manual_label"], "3")
            self.assertIn("review_saved_at", payload)


if __name__ == "__main__":
    unittest.main()
