from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.adapters.knowledge_base import LocalKnowledgeBase, WeiboSearchKnowledgeBase, unique_keep_order


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

    def test_weibo_search_recall_ids_from_result_id(self) -> None:
        class StubWeiboSearchKnowledgeBase(WeiboSearchKnowledgeBase):
            def __init__(self):
                super().__init__()
                self.responses = [
                    {"sp": {"result": [{"ID": "123"}, {"ID": "456"}, {"ID": "123"}]}},
                ]

            def _recall_items(self, query: str, recall_size: int):
                result = self.responses[0]["sp"]["result"]
                items = []
                for item in result:
                    mid = str(item.get("ID") or "").strip()
                    if mid:
                        items.append({"mid": mid, "text": self._extract_recall_text(item)})
                    if len(items) >= recall_size:
                        break
                deduped = []
                seen = set()
                for item in items:
                    if item["mid"] in seen:
                        continue
                    seen.add(item["mid"])
                    deduped.append(item)
                return deduped

        kb = StubWeiboSearchKnowledgeBase()
        self.assertEqual(
            [item["mid"] for item in kb._recall_items("日沐星辰", 5)],
            ["123", "456"],
        )

    def test_weibo_search_falls_back_to_recall_text_when_hbase_empty(self) -> None:
        class StubWeiboSearchKnowledgeBase(WeiboSearchKnowledgeBase):
            def _recall_items(self, query: str, recall_size: int):
                return [{"mid": "123", "text": "日沐星辰 | 账号资料"}]

            def _get_hbase_batch(self, mids):
                return [None]

        kb = StubWeiboSearchKnowledgeBase()
        results = kb.search(["日沐星辰"], top_k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "日沐星辰 | 账号资料")


if __name__ == "__main__":
    unittest.main()
