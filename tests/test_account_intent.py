from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.adapters.account_intent import AccountIntentDetector
from auto_label_agent.core.pipeline import AutoLabelAgent


class StubDetector(AccountIntentDetector):
    def __init__(self, payload):
        super().__init__(timeout=1)
        self.payload = payload

    def _fetch_json(self, params):
        return self.payload


class FakeLLMClient:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "intent": "test",
            "entities": [],
            "constraints": [],
            "notes": "",
            "should_use_kb": False,
            "kb_queries": [],
            "label": 2,
            "label_name": "相关",
            "reason": "stub",
            "passed": True,
            "final_action": "finish",
            "issues": [],
        }


class AccountIntentDetectorTest(unittest.TestCase):
    def test_exact_match_is_one_of_account_intents(self):
        detector = StubDetector({"users": [{"screen_name": "一条闪耀的大蟒蛇"}]})
        signal = detector.detect("一条闪耀的大蟒蛇")
        self.assertTrue(signal.matched)
        self.assertEqual("account_as_one_intent", signal.intent_type)
        self.assertEqual("一条闪耀的大蟒蛇", signal.exact_screen_name)

    def test_exact_match_common_expression_is_one_of_intents(self):
        detector = StubDetector({"users": [{"screen_name": "头像"}]})
        signal = detector.detect("头像")
        self.assertTrue(signal.matched)
        self.assertEqual("account_as_one_intent", signal.intent_type)
        self.assertIn("候选需求之一", signal.reason)


class PipelineAccountIntentTest(unittest.TestCase):
    def test_pipeline_applies_account_signal(self):
        detector = StubDetector({"users": [{"screen_name": "一条闪耀的大蟒蛇"}]})
        agent = AutoLabelAgent(
            llm_client=FakeLLMClient(),
            account_intent_detector=detector,
        )
        result = agent.run(query="一条闪耀的大蟒蛇", doc="doc")
        self.assertTrue(result.intent["account_intent_detected"])
        self.assertEqual("account_as_one_intent", result.intent["account_intent_type"])
        self.assertIn("@一条闪耀的大蟒蛇", result.intent["entities"])
        self.assertIn("account", result.intent["candidate_target_types"])


if __name__ == "__main__":
    unittest.main()
