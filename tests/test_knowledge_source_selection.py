from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.adapters.knowledge_base import MultiSourceKnowledgeBase
from auto_label_agent.core.pipeline import AutoLabelAgent


class KnowledgeSourceSelectionTest(unittest.TestCase):
    def test_multi_source_defaults_to_duckduck(self) -> None:
        kb = MultiSourceKnowledgeBase([])
        self.assertEqual(kb.providers, ["duckduck"])

    def test_multi_source_keeps_unique_order(self) -> None:
        kb = MultiSourceKnowledgeBase(["weibo_search", "duckduck", "weibo_search"])
        self.assertEqual(kb.providers, ["weibo_search", "duckduck"])

    def test_multi_source_supports_weibo_llm(self) -> None:
        kb = MultiSourceKnowledgeBase(["weibo_llm", "duckduck", "weibo_llm"])
        self.assertEqual(kb.providers, ["weibo_llm", "duckduck"])

    def test_agent_supports_weibo_llm_online_kb_provider(self) -> None:
        agent = AutoLabelAgent(
            llm_client=None,  # type: ignore[arg-type]
            online_kb_enabled=True,
            online_kb_provider="weibo_llm",
        )
        self.assertIsNotNone(agent.online_kb)

    def test_agent_supports_all_online_kb_provider(self) -> None:
        agent = AutoLabelAgent(
            llm_client=None,  # type: ignore[arg-type]
            online_kb_enabled=True,
            online_kb_provider="all",
        )
        self.assertEqual(agent.online_kb.providers, ["duckduck", "weibo_search", "weibo_llm"])  # type: ignore[union-attr]

    def test_both_now_includes_weibo_llm(self) -> None:
        agent = AutoLabelAgent(
            llm_client=None,  # type: ignore[arg-type]
            online_kb_enabled=True,
            online_kb_provider="both",
        )
        self.assertEqual(agent.online_kb.providers, ["duckduck", "weibo_search", "weibo_llm"])  # type: ignore[union-attr]

    def test_agent_rejects_unknown_online_kb_provider(self) -> None:
        with self.assertRaises(ValueError):
            AutoLabelAgent(
                llm_client=None,  # type: ignore[arg-type]
                online_kb_enabled=True,
                online_kb_provider="unknown",
            )


if __name__ == "__main__":
    unittest.main()
