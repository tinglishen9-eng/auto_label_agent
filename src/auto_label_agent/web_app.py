import json
import logging
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from auto_label_agent.adapters.llm_client import (
    provider_requires_api_key,
    resolve_api_key,
    resolve_endpoint,
)
from auto_label_agent.service import build_result_payload, create_agent
from auto_label_agent.utils.input_parser import parse_input_file
from auto_label_agent.utils.review_store import append_review_record


load_dotenv()
logger = logging.getLogger(__name__)
JOB_STORE: Dict[str, Dict[str, Any]] = {}
JOB_STORE_LOCK = threading.Lock()


class JobLogHandler(logging.Handler):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        with JOB_STORE_LOCK:
            job = JOB_STORE.get(self.job_id)
            if not job:
                return
            job.setdefault("logs", []).append(message)
            job["updated"] = True


def append_job_log(job_id: str, message: str) -> None:
    with JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if not job:
            return
        job.setdefault("logs", []).append(message)
        job["updated"] = True


def append_job_logs(job_id: str, messages: list[str]) -> None:
    for message in messages:
        append_job_log(job_id, message)


def set_job_progress(job_id: str, message: str) -> None:
    with JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if not job:
            return
        job["progress"] = message
        job["updated"] = True


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auto Label Agent Web</title>
  <style>
    :root { color-scheme: light; --bg:#f4f1ea; --panel:#fffdf7; --ink:#1d1b18; --muted:#6d665d; --accent:#0f766e; --line:#d8d0c4; --warn:#b45309; }
    body { margin:0; font-family: "Segoe UI", "PingFang SC", sans-serif; background:linear-gradient(180deg,#f3efe6,#eae4d7); color:var(--ink); }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 16px; font-size: 28px; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    .grid { display:grid; grid-template-columns: 1.1fr 0.9fr; gap:16px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }
    .row { display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:12px; }
    .row3 { display:grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:12px; }
    label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }
    input[type=text], input[type=number], textarea, select { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:12px; padding:10px 12px; background:#fff; font:inherit; }
    textarea { min-height:100px; resize:vertical; }
    .doc { min-height:180px; }
    .small { min-height:72px; }
    button { border:none; background:var(--accent); color:#fff; padding:10px 14px; border-radius:999px; cursor:pointer; font-weight:600; }
    button.secondary { background:#5b6b7a; }
    button.warn { background:var(--warn); }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f7f4ee; border-radius:12px; padding:12px; border:1px solid var(--line); max-height:420px; overflow:auto; }
    .meta { color:var(--muted); font-size:13px; margin-top:8px; }
    .span2 { grid-column: 1 / -1; }
    .checkbox { display:flex; gap:8px; align-items:center; margin-top:8px; }
    .status { margin-top:10px; font-size:13px; color:var(--muted); }
    @media (max-width: 980px) { .grid, .row, .row3 { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Auto Label Agent Web</h1>
  <div class="grid">
    <section class="panel">
      <details>
        <summary style="cursor:pointer;font-weight:700;margin-bottom:12px;">运行配置</summary>
      <div class="row3">
        <div><label>Provider</label><input id="provider" type="text" value="mproxy"></div>
        <div><label>Model</label><input id="model" type="text" value="qwen272b"></div>
        <div><label>Endpoint</label><input id="endpoint" type="text" value=""></div>
      </div>
      <div class="row3">
        <div><label>Temperature</label><input id="temperature" type="number" step="0.1" value="0.1"></div>
        <div><label>KB Top K</label><input id="kb_top_k" type="number" value="3"></div>
        <div><label>Max Rounds</label><input id="max_rounds" type="number" value="3"></div>
      </div>
      <div class="row">
        <div><label>Prompt File</label><input id="prompt_file" type="text" value=""></div>
        <div><label>KB File</label><input id="kb_file" type="text" value=""></div>
      </div>
      <div class="row3">
        <div><label>Online KB</label><select id="online_kb"><option value="false">false</option><option value="true">true</option></select></div>
        <div><label>Online KB Provider</label><select id="online_kb_provider"><option value="duckduck">duckduck</option><option value="weibo_search">weibo_search</option><option value="weibo_llm">weibo_llm</option><option value="both">both</option><option value="all">all</option></select></div>
        <div><label>Review Output File</label><input id="review_output_file" type="text" value="data/review_output.jsonl"></div>
      </div>
      </details>
      <h2>样本输入</h2>
      <div class="row">
        <div class="span2"><label>Input File</label><input id="input_file" type="text" placeholder="批量文件路径，可选"></div>
      </div>
      <div class="actions">
        <button type="button" class="secondary" id="load_button">加载当前记录</button>
        <button type="button" class="secondary" id="prev_button">上一条</button>
        <button type="button" class="secondary" id="next_button">下一条</button>
      </div>
      <div class="meta" id="record_meta">尚未加载批量记录</div>
      <div class="row3">
        <div><label>Current Index</label><input id="record_index" type="number" value="0"></div>
        <div><label>Row Index</label><input id="row_index" type="text" value="" readonly></div>
        <div><label>Mid</label><input id="mid" type="text" value="" readonly></div>
      </div>
      <div class="row">
        <div class="span2"><label>Query</label><textarea id="query" class="small"></textarea></div>
      </div>
      <div class="row">
        <div class="span2"><label>Doc Text</label><textarea id="doc" class="doc"></textarea></div>
      </div>
      <div class="row">
        <div><label>Doc JSON</label><textarea id="doc_json"></textarea></div>
        <div><label>Other Info JSON</label><textarea id="other_info"></textarea></div>
      </div>
      <div class="actions">
        <button type="button" id="run_button">自动标注</button>
      </div>
      <div class="status" id="run_status"></div>
    </section>
    <section class="panel">
      <h2>自动结果</h2>
      <pre id="auto_result">{}</pre>
      <h2>人工复核</h2>
      <div class="checkbox"><input id="review_passed" type="checkbox"><label for="review_passed">人工复核通过</label></div>
      <div class="checkbox"><input id="manual_override" type="checkbox"><label for="manual_override">人工覆盖自动标签</label></div>
      <div class="row3">
        <div><label>人工标签</label><select id="manual_label"><option value="">不填写</option><option value="0">0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></div>
        <div><label>错误类型</label><input id="error_types" type="text" placeholder="如 missing_entity, topic_drift"></div>
        <div><label>Reviewer</label><input id="reviewer" type="text" placeholder="可选"></div>
      </div>
      <div class="row">
        <div><label>人工原因</label><textarea id="manual_reason"></textarea></div>
        <div><label>错误原因/分析</label><textarea id="error_reason"></textarea></div>
      </div>
      <div class="row">
        <div class="span2"><label>复核备注</label><textarea id="review_notes"></textarea></div>
      </div>
      <div class="actions">
        <button type="button" class="warn" id="save_button">保存复核结果</button>
      </div>
      <div class="status" id="review_status"></div>
    </section>
  </div>
</div>
<script>
let lastAutoResult = null;
let totalRecords = 0;
let currentJobId = null;
let pollTimer = null;

function cfg() {
  return {
    provider: document.getElementById("provider").value,
    model: document.getElementById("model").value,
    endpoint: document.getElementById("endpoint").value,
    temperature: parseFloat(document.getElementById("temperature").value || "0.1"),
    kb_file: document.getElementById("kb_file").value,
    online_kb: document.getElementById("online_kb").value === "true",
    online_kb_provider: document.getElementById("online_kb_provider").value,
    kb_top_k: parseInt(document.getElementById("kb_top_k").value || "3", 10),
    max_rounds: parseInt(document.getElementById("max_rounds").value || "3", 10),
    prompt_file: document.getElementById("prompt_file").value
  };
}

function fillRecord(data) {
  document.getElementById("query").value = data.query || "";
  document.getElementById("doc").value = data.doc || "";
  document.getElementById("doc_json").value = JSON.stringify(data.doc_json || {}, null, 2);
  document.getElementById("other_info").value = JSON.stringify(data.other_info || {}, null, 2);
  document.getElementById("row_index").value = data.row_index || "";
  document.getElementById("mid").value = data.mid || "";
  totalRecords = data.total_records || 0;
  document.getElementById("record_meta").innerText = `当前记录 ${data.index + 1}/${data.total_records}，row=${data.row_index}，mid=${data.mid || ""}`;
}

async function loadRecord() {
  try {
    const inputFile = document.getElementById("input_file").value.trim();
    if (!inputFile) { document.getElementById("record_meta").innerText = "请先填写 Input File"; return; }
    const index = document.getElementById("record_index").value || "0";
    const res = await fetch(`/api/record?input_file=${encodeURIComponent(inputFile)}&index=${encodeURIComponent(index)}`);
    const data = await res.json();
    if (!res.ok) { document.getElementById("record_meta").innerText = data.error || "加载失败"; return; }
    fillRecord(data);
  } catch (err) {
    document.getElementById("record_meta").innerText = `加载失败: ${err}`;
  }
}

function prevRecord() {
  const indexNode = document.getElementById("record_index");
  indexNode.value = Math.max(0, parseInt(indexNode.value || "0", 10) - 1);
  loadRecord();
}

function nextRecord() {
  const indexNode = document.getElementById("record_index");
  const next = parseInt(indexNode.value || "0", 10) + 1;
  indexNode.value = totalRecords > 0 ? Math.min(totalRecords - 1, next) : next;
  loadRecord();
}

async function runAutoLabel() {
  const button = document.getElementById("run_button");
  button.disabled = true;
  try {
    document.getElementById("run_status").innerText = "任务创建中...";
    const body = {
      ...cfg(),
      query: document.getElementById("query").value,
      doc: document.getElementById("doc").value
    };
    const res = await fetch("/api/auto_label_async", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) {
      document.getElementById("run_status").innerText = data.error || "自动标注失败";
      return;
    }
    currentJobId = data.job_id;
    document.getElementById("run_status").innerText = "任务已启动";
    if (pollTimer) { clearInterval(pollTimer); }
    pollTimer = setInterval(pollJobStatus, 1200);
    await pollJobStatus();
  } catch (err) {
    document.getElementById("run_status").innerText = `自动标注失败: ${err}`;
  } finally {
    button.disabled = false;
  }
}

async function pollJobStatus() {
  try {
    if (!currentJobId) { return; }
    const res = await fetch(`/api/job_status?job_id=${encodeURIComponent(currentJobId)}`);
    const data = await res.json();
    if (!res.ok) {
      document.getElementById("run_status").innerText = data.error || "任务状态查询失败";
      return;
    }
    document.getElementById("run_status").innerText = data.progress || `状态: ${data.status}`;
    if (data.status === "completed") {
      lastAutoResult = data.result;
      document.getElementById("auto_result").innerText = JSON.stringify(data.result, null, 2);
      const elapsed = data.result && typeof data.result.total_elapsed_ms === "number"
        ? `，耗时 ${(data.result.total_elapsed_ms / 1000).toFixed(2)}s`
        : "";
      document.getElementById("run_status").innerText = `自动标注完成${elapsed}`;
      clearInterval(pollTimer);
      pollTimer = null;
    } else if (data.status === "failed") {
      document.getElementById("run_status").innerText = `自动标注失败: ${data.error || ""}`;
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (err) {
    document.getElementById("run_status").innerText = `任务状态查询失败: ${err}`;
  }
}

async function saveReview() {
  try {
    const outputFile = document.getElementById("review_output_file").value.trim();
    if (!outputFile) { document.getElementById("review_status").innerText = "请填写 Review Output File"; return; }
    const body = {
      output_file: outputFile,
      auto_result: lastAutoResult,
      source: {
        input_file: document.getElementById("input_file").value,
        record_index: parseInt(document.getElementById("record_index").value || "0", 10),
        row_index: document.getElementById("row_index").value,
        mid: document.getElementById("mid").value,
        query: document.getElementById("query").value,
        doc: document.getElementById("doc").value,
        doc_json: JSON.parse(document.getElementById("doc_json").value || "{}"),
        other_info: JSON.parse(document.getElementById("other_info").value || "{}")
      },
      review: {
        review_passed: document.getElementById("review_passed").checked,
        manual_override: document.getElementById("manual_override").checked,
        manual_label: document.getElementById("manual_label").value,
        manual_reason: document.getElementById("manual_reason").value,
        error_types: document.getElementById("error_types").value,
        error_reason: document.getElementById("error_reason").value,
        review_notes: document.getElementById("review_notes").value,
        reviewer: document.getElementById("reviewer").value
      }
    };
    const res = await fetch("/api/save_review", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    document.getElementById("review_status").innerText = res.ok ? `已保存到 ${data.output_file}` : (data.error || "保存失败");
  } catch (err) {
    document.getElementById("review_status").innerText = `保存失败: ${err}`;
  }
}

function bootPage() {
  const loadButton = document.getElementById("load_button");
  const prevButton = document.getElementById("prev_button");
  const nextButton = document.getElementById("next_button");
  const runButton = document.getElementById("run_button");
  const saveButton = document.getElementById("save_button");
  if (loadButton) { loadButton.addEventListener("click", loadRecord); }
  if (prevButton) { prevButton.addEventListener("click", prevRecord); }
  if (nextButton) { nextButton.addEventListener("click", nextRecord); }
  if (runButton) { runButton.addEventListener("click", runAutoLabel); }
  if (saveButton) { saveButton.addEventListener("click", saveReview); }
  document.getElementById("run_status").innerText = "页面初始化完成";
}

document.addEventListener("DOMContentLoaded", bootPage);
</script>
</body>
</html>
"""


def start_web_app(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _build_handler())
    logger.info("Web UI 已启动: http://%s:%s", host, port)
    server.serve_forever()


def _build_handler():
    class AutoLabelWebHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(HTML_PAGE)
                return
            if parsed.path == "/api/record":
                self._handle_record(parsed.query)
                return
            if parsed.path == "/api/job_status":
                self._handle_job_status(parsed.query)
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/auto_label":
                self._handle_auto_label()
                return
            if parsed.path == "/api/auto_label_async":
                self._handle_auto_label_async()
                return
            if parsed.path == "/api/save_review":
                self._handle_save_review()
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("web " + format, *args)

        def _handle_record(self, query_string: str) -> None:
            try:
                params = parse_qs(query_string)
                input_file = params.get("input_file", [""])[0]
                index = int(params.get("index", ["0"])[0])
                records = parse_input_file(input_file)
                if not records:
                    raise ValueError("输入文件中没有可用记录")
                if index < 0 or index >= len(records):
                    raise ValueError(f"记录索引越界: {index}")
                record = records[index]
                self._send_json(
                    {
                        "index": index,
                        "total_records": len(records),
                        "row_index": record.row_index,
                        "query": record.query,
                        "doc": record.doc_text,
                        "doc_json": record.doc_json,
                        "other_info": record.other_info,
                        "mid": record.other_info.get("mid"),
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _handle_auto_label(self) -> None:
            try:
                body = self._read_json_body()
                self._send_json(self._run_auto_label(body))
            except Exception as exc:
                logger.exception("web auto label failed")
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _handle_auto_label_async(self) -> None:
            try:
                body = self._read_json_body()
                job_id = uuid.uuid4().hex
                with JOB_STORE_LOCK:
                    JOB_STORE[job_id] = {
                        "status": "queued",
                        "progress": "排队中",
                        "logs": ["任务已创建"],
                        "result": None,
                        "error": "",
                    }
                worker = threading.Thread(target=self._run_job, args=(job_id, body), daemon=True)
                worker.start()
                self._send_json({"ok": True, "job_id": job_id})
            except Exception as exc:
                logger.exception("web auto label async failed")
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _handle_job_status(self, query_string: str) -> None:
            params = parse_qs(query_string)
            job_id = params.get("job_id", [""])[0]
            with JOB_STORE_LOCK:
                job = JOB_STORE.get(job_id)
            if not job:
                self._send_json({"error": f"job 不存在: {job_id}"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)

        def _handle_save_review(self) -> None:
            try:
                body = self._read_json_body()
                output_file = str(body.get("output_file") or "").strip()
                if not output_file:
                    raise ValueError("output_file 不能为空")

                auto_result = body.get("auto_result") or {}
                review = body.get("review") or {}
                source = body.get("source") or {}
                final_label = review.get("manual_label") if review.get("manual_override") and review.get("manual_label") != "" else auto_result.get("label")
                payload = {
                    "source": source,
                    "auto_result": auto_result,
                    "review": review,
                    "final_label": final_label,
                }
                saved_path = append_review_record(output_file, payload)
                self._send_json({"ok": True, "output_file": saved_path})
            except Exception as exc:
                logger.exception("web save review failed")
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _run_auto_label(self, body: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
            query = str(body.get("query") or "").strip()
            doc = str(body.get("doc") or "").strip()
            if not query or not doc:
                raise ValueError("query 和 doc 不能为空")
            if job_id:
                append_job_logs(
                    job_id,
                    [
                        f"收到输入: query_len={len(query)}, doc_len={len(doc)}",
                        "开始检查运行配置",
                    ],
                )

            provider = str(body.get("provider") or "deepseek")
            api_key = resolve_api_key(provider)
            if provider_requires_api_key(provider) and not api_key:
                raise ValueError(f"缺少 API Key，请在环境变量或 .env 中配置 {provider.upper()}_API_KEY。")
            if job_id:
                set_job_progress(job_id, "初始化中")

            agent = create_agent(
                provider=provider,
                model=str(body.get("model") or "deepseek-chat"),
                endpoint=resolve_endpoint(provider, str(body.get("endpoint") or "")),
                api_key=api_key,
                temperature=float(body.get("temperature") or 0.1),
                kb_file=str(body.get("kb_file") or ""),
                online_kb=bool(body.get("online_kb")),
                online_kb_provider=str(body.get("online_kb_provider") or "duckduck"),
                kb_top_k=int(body.get("kb_top_k") or 3),
                max_rounds=int(body.get("max_rounds") or 3),
                prompt_file=str(body.get("prompt_file") or "") or None,
                progress_callback=(lambda message: set_job_progress(job_id, message)) if job_id else None,
            )
            if job_id:
                set_job_progress(job_id, "第1轮 - 准备开始")
            result = agent.run(query=query, doc=doc)
            if job_id:
                set_job_progress(job_id, "整理结果中")
            payload = build_result_payload(query=query, doc=doc, result=result)
            if job_id:
                set_job_progress(job_id, "标注完成")
            return payload

        def _run_job(self, job_id: str, body: Dict[str, Any]) -> None:
            handler = JobLogHandler(job_id)
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            with JOB_STORE_LOCK:
                JOB_STORE[job_id]["status"] = "running"
                JOB_STORE[job_id]["progress"] = "启动中"
            try:
                result = self._run_auto_label(body, job_id=job_id)
                with JOB_STORE_LOCK:
                    JOB_STORE[job_id]["status"] = "completed"
                    JOB_STORE[job_id]["progress"] = "标注完成"
                    JOB_STORE[job_id]["result"] = result
            except Exception as exc:
                logger.exception("background auto label failed")
                with JOB_STORE_LOCK:
                    JOB_STORE[job_id]["status"] = "failed"
                    JOB_STORE[job_id]["progress"] = "标注失败"
                    JOB_STORE[job_id]["error"] = str(exc)
            finally:
                root_logger.removeHandler(handler)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return payload

        def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return AutoLabelWebHandler
