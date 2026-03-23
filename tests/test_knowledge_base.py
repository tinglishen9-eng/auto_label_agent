from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.adapters.knowledge_base import LocalKnowledgeBase, unique_keep_order


class KnowledgeBaseTest(unittest.TestCase):
    def test_unique_keep_order_filters_empty_and_duplicates(self) -> None:
        items = ["苹果", "", "苹果", "手机", "  手机  ", "电池"]
        self.assertEqual(unique_keep_order(items), ["苹果", "手机", "电池"])

    def test_local_search_dedup_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_path = Path(tmp_dir) / "kb.json"
            kb_path.write_text(
                '[{"title":"iPhone 13","content":"苹果 手机 电池 容量 信息"}]',
                encoding="utf-8",
            )
            kb = LocalKnowledgeBase(str(kb_path))
            results = kb.search(["苹果手机", "苹果手机", "电池容量"], top_k=3)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "iPhone 13")


if __name__ == "__main__":
    unittest.main()
