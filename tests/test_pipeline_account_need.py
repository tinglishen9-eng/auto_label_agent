from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.adapters.knowledge_base import KnowledgeChunk
from auto_label_agent.core.pipeline import AutoLabelAgent


class FakeLLMClient:
    def complete_json(self, system_prompt: str, user_prompt: str):
        if "请分析下面的 query 需求" in user_prompt:
            return {
                "intent_clear": False,
                "understood_intent": "用户想搜索小红书相关内容",
                "task_objective": "理解 query 真实需求",
                "entities": ["小红书"],
                "qualifiers": [],
                "constraints": [],
                "missing_points": [],
                "should_use_kb": False,
                "kb_queries": [],
                "notes": "",
            }
        return {
            "label": 0,
            "label_name": "0-不相关",
            "reason": "",
            "matched_points": [],
            "missing_points": [],
            "evidence_spans": [],
            "confidence": "low",
        }


class PipelineAccountNeedTest(unittest.TestCase):
    def test_account_need_detected_from_knowledge_handle(self) -> None:
        agent = AutoLabelAgent(llm_client=FakeLLMClient(), prompt_file=None)
        knowledge = [
            KnowledgeChunk(
                source="test",
                title="账号资料",
                content="相关账号：@小红书官方账号，欢迎关注。",
                score=10,
            )
        ]
        intent = agent._understand_intent("小红书", feedback="", knowledge=knowledge)
        self.assertTrue(intent["account_search_detected"])
        self.assertIn("@小红书官方账号", intent["matched_account_handles"])
        self.assertIn("@小红书", intent["entities"])
        self.assertIn("账号搜索需求", intent["notes"])

    def test_build_kb_queries_expands_entities_for_concatenated_query(self) -> None:
        agent = AutoLabelAgent(llm_client=FakeLLMClient(), prompt_file=None)
        intent = {
            "entities": ["田栩", "宁梓渝", "逆爱"],
            "qualifiers": [],
            "kb_queries": ["田栩宁梓渝逆爱"],
        }

        queries = agent._build_kb_queries("田栩宁梓渝逆爱", intent)

        self.assertEqual(
            queries,
            ["田栩宁梓渝逆爱", "田栩", "宁梓渝", "逆爱"],
        )

    def test_build_kb_queries_keeps_normal_query_unchanged(self) -> None:
        agent = AutoLabelAgent(llm_client=FakeLLMClient(), prompt_file=None)
        intent = {
            "entities": ["苹果手机"],
            "qualifiers": ["电池容量"],
            "kb_queries": ["苹果手机13 电池容量"],
        }

        queries = agent._build_kb_queries("苹果手机13 电池容量", intent)

        self.assertEqual(queries, ["苹果手机13 电池容量"])


if __name__ == "__main__":
    unittest.main()
