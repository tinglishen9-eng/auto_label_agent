from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.utils.prompt_loader import DEFAULT_PROMPT_DIR, load_prompt_bundle


class PromptLoaderTest(unittest.TestCase):
    def test_default_prompt_bundle_has_required_keys(self) -> None:
        bundle = load_prompt_bundle()
        self.assertEqual(set(bundle.keys()), {"intent", "scoring", "verify"})
        self.assertTrue(bundle["intent"])
        self.assertTrue(bundle["scoring"])
        self.assertTrue(bundle["verify"])

    def test_default_prompt_dir_exists(self) -> None:
        self.assertTrue(DEFAULT_PROMPT_DIR.exists())


if __name__ == "__main__":
    unittest.main()
