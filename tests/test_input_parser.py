from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.utils.input_parser import parse_input_file, render_doc_text


class InputParserTest(unittest.TestCase):
    def test_render_doc_text_uses_expected_fields(self) -> None:
        doc_json = {
            "content": "正文内容",
            "abstract": "摘要内容",
            "title": "标题内容",
            "keywords": "关键词1;关键词2",
            "ocr_text": "OCR 内容",
            "video_voice": "视频语音",
            "ismodal": 1,
            "extra_field": "额外信息",
        }
        text = render_doc_text(doc_json)
        self.assertIn("content: 正文内容", text)
        self.assertIn("abstract: 摘要内容", text)
        self.assertIn("title: 标题内容", text)
        self.assertIn("keywords: 关键词1;关键词2", text)
        self.assertIn("ocr_text: OCR 内容", text)
        self.assertIn("video_voice: 视频语音", text)
        self.assertIn("ismodal: 1", text)
        self.assertIn("extra_field: 额外信息", text)

    def test_parse_input_file_reads_three_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.tsv"
            path.write_text(
                '\t'.join(
                    [
                        "测试query",
                        '{"content":"正文","abstract":"摘要","title":"标题","keywords":"关键词","ocr_text":"ocr","video_voice":"voice","ismodal":0}',
                        '{"mid":"12345","source":"test"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            records = parse_input_file(str(path))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.row_index, 1)
        self.assertEqual(record.query, "测试query")
        self.assertEqual(record.other_info["mid"], "12345")
        self.assertIn("content: 正文", record.doc_text)
        self.assertIn("video_voice: voice", record.doc_text)


if __name__ == "__main__":
    unittest.main()
