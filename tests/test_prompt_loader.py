from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.utils.prompt_loader import DEFAULT_PROMPT_FILE, PromptBundle, load_prompt_bundle


class PromptLoaderTest(unittest.TestCase):
    def test_default_prompt_bundle_has_required_keys(self) -> None:
        bundle = load_prompt_bundle()
        self.assertIsInstance(bundle, PromptBundle)
        self.assertEqual(set(bundle.templates.keys()), {"intent", "scoring", "verify"})
        self.assertTrue(bundle.render("intent"))
        self.assertTrue(bundle.render("scoring"))
        self.assertTrue(bundle.render("verify"))
        self.assertTrue(bundle.version)
        self.assertTrue(bundle.description)

    def test_default_prompt_bundle_has_version_and_few_shot_examples(self) -> None:
        bundle = load_prompt_bundle()
        self.assertGreaterEqual(bundle.few_shot_count("intent"), 1)
        self.assertGreaterEqual(bundle.few_shot_count("scoring"), 1)
        self.assertGreaterEqual(bundle.few_shot_count("verify"), 1)
        self.assertIn("few-shot", bundle.render("intent"))

    def test_default_prompt_file_exists(self) -> None:
        self.assertTrue(DEFAULT_PROMPT_FILE.exists())
        self.assertEqual(DEFAULT_PROMPT_FILE.name, "system_prompts.py")


if __name__ == "__main__":
    unittest.main()
