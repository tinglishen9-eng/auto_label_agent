# Auto Label Agent

一个独立的自动标注 agent 项目，输入 `query` 和 `doc`，先判断需求理解是否充分，再决定是否调用额外知识库辅助理解，随后输出相关性标签 `0-3`，并在标注后做二次验证；如果验证不通过，会根据原因自动回退到需求理解、补知识或重打分步骤。

## 目录结构

```text
auto_label_agent/
  data/                      # 样本、知识库、输出结果等数据文件
  src/
    auto_label_agent/
      adapters/             # 大模型、搜索、数据库、MCP 等外部依赖适配层
      core/                 # 核心业务流程与状态机
      prompts/              # 单文件 prompt 配置
      utils/                # 纯工具函数
      cli.py                # CLI 入口
  tests/                    # 单元测试
  run.py                    # 启动脚本
  README.md
```

## 结构建议

- 大模型调用和外部知识调用，不建议放进 `utils`。
- 更推荐放在 `adapters` 或 `clients`。因为这类代码本质上是在适配外部系统，不是通用工具。
- `utils` 更适合放纯函数，例如 prompt 加载、文本处理、JSON 清洗、路径拼装。
- `core` 里只放业务流程，比如需求理解、打分、验证、回退重跑。

当前项目的核心位置：

- `src/auto_label_agent/adapters/llm_client.py`：大模型调用，已支持 `deepseek/openai/kimi/mproxy`
- `src/auto_label_agent/adapters/knowledge_base.py`：本地知识库、DuckDuckGo 和微博搜索在线知识调用
- `src/auto_label_agent/core/pipeline.py`：自动标注主流程
- `src/auto_label_agent/utils/prompt_loader.py`：prompt 加载工具
- `src/auto_label_agent/prompts/system_prompts.py`：单文件 prompt 配置
- `src/auto_label_agent/prompts/few_shot_examples.py`：few-shot 示例配置
- `data/`：预留给样本、知识库和输出结果
- `tests/`：测试目录
- `run.py`：推荐启动脚本

## 在线知识源

支持：

- `--online-kb-provider duckduck`
- `--online-kb-provider weibo_search`
- `--online-kb-provider both`

示例：

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python run.py ^
  --provider mproxy ^
  --model qwen272b ^
  --online-kb ^
  --online-kb-provider both ^
  --query "胖梦聚餐合影" ^
  --doc "..."
```

## 自定义 AI 调用

支持：

- `--provider mproxy`
- `--provider weibo_proxy`

默认接口：

```text
http://mproxy.search.weibo.com/llm/generate
```

这个 provider 默认不要求 API Key，但如果你的代理环境需要，也支持通过环境变量或 `.env` 配置 `api_key`，程序会自动读取并透传。

支持的 `api_key` 环境变量：

```env
MPROXY_API_KEY=your_api_key
WEIBO_PROXY_API_KEY=your_api_key
API_KEY=your_api_key
```

示例：

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python run.py ^
  --provider mproxy ^
  --model qwen272b ^
  --query "苹果手机13的电池容量" ^
  --doc "iPhone 13 配备 3240mAh 电池，支持视频播放..."
```

## Kimi 配置

支持：

- `--provider kimi`
- `--provider moonshot`

默认接口：

```text
https://api.moonshot.cn/v1/chat/completions
```

环境变量支持：

```env
KIMI_API_KEY=your_kimi_key
MOONSHOT_API_KEY=your_moonshot_key
```

## 流程

1. 需求理解：抽取意图、实体、限定词、隐含约束，判断是否需要额外知识。
2. 补知识：仅在理解不足时调用本地知识库或在线知识，帮助澄清 query。
3. 相关性打分：结合 `query + 需求理解 + doc` 给出 `0-3` 标签。
4. 二次验证：检查是否存在实体/限定词缺失、主题偏移、不满足需求、语义不一致等问题。
5. 自动回退：验证不通过时，根据 `final_action` 重新走 `revisit_intent`、`retrieve_kb` 或 `rescore`。

## 标签定义

- `0`：不相关
- `1`：弱相关
- `2`：较相关
- `3`：完全相关

## 启动

推荐方式：

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python run.py --help
```

命令行入口会继续保留，不影响原有单条或批量标注方式。

如果想看到更完整的日志和进度提示，可以加上：

```bash
conda run -n base python run.py --verbose ...
```

开启后会输出：

- 当前运行配置
- `需求理解 / 补充知识检索 / 相关性打分 / 二次验证` 的阶段进度
- 本地知识库和在线知识源返回数量
- 最终标签、验证结果和输出文件位置

## Web 界面

项目同时提供一个轻量 Web 界面，便于人工复核。

启动方式：

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python run_web.py --host 127.0.0.1 --port 18081
```

打开：

```text
http://127.0.0.1:18081
```

Web 界面支持：

- 保留命令行之外的可视化操作
- 运行配置区域可收起
- 加载 `--input-file` 同格式的三列批量文件
- 逐条查看 `query / doc / doc_json / other_info / mid`
- 直接触发自动标注
- 自动标注过程日志和阶段进度实时更新到页面
- 人工复核通过/不通过
- 人工覆盖标签
- 手工填写错误类型、错误原因、备注、reviewer
- 将人工复核结果追加写入 `jsonl` 文件

## 批量输入文件

支持通过 `--input-file` 直接读取批量样本文件。

输入文件默认按制表符分成 3 列：

1. 第 1 列：`query`
2. 第 2 列：`doc`
   这是一个 JSON 对象，常见字段包括：
   `content`、`abstract`、`title`、`keywords`、`ocr_text`、`video_voice`、`ismodal`
3. 第 3 列：`other_info`
   这是一个 JSON 对象，当前会保留并回写，常见字段包括 `mid`

程序会自动把第 2 列 `doc_json` 展开成模型可读文本，再送入标注流程。

批量输出建议写成 `jsonl`：

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python run.py ^
  --provider mproxy ^
  --model qwen272b ^
  --input-file .\data\sample.tsv ^
  --output .\data\result.jsonl ^
  --verbose
```

每条输出会保留：

- `query`
- `doc`
- `doc_json`
- `other_info`
- `mid`
- 标注结果和 trace

## Prompt 配置

默认会从 `src/auto_label_agent/prompts/system_prompts.py` 读取一整套“prompt bundle”。现在这套配置已经支持：

- `PROMPT_VERSION`：prompt 版本号
- `PROMPT_DESCRIPTION`：这版 prompt 的说明
- `PROMPTS`：三个阶段的 prompt 配置
- `few_shot_examples`：每个阶段自己的 few-shot 示例

这样后面调 prompt 时，可以明确记录“用了哪一版、带了哪些示例”，不会出现改来改去但忘了效果对应哪版的问题。

为了便于手动维护，当前默认 prompt 已整理成两部分：

- `src/auto_label_agent/prompts/system_prompts.py`
  - 只保留版本、说明、三段 system prompt 和最终组装逻辑
- `src/auto_label_agent/prompts/few_shot_examples.py`
  - 单独维护 `intent / scoring / verify` 的 few-shot 示例

如果你主要是在调示例，优先改 `few_shot_examples.py` 会更方便。

如果你想切换成另一套 prompt，可以传：

```bash
--prompt-file path\to\system_prompts.py
```

项目里也保留了第一版 prompt：

- `src/auto_label_agent/prompts/system_prompts_v1.py`

如果你想切回第一版，可以这样跑：

```bash
--prompt-file .\src\auto_label_agent\prompts\system_prompts_v1.py
```

Python 文件建议导出：

- `PROMPT_VERSION`
- `PROMPT_DESCRIPTION`
- `PROMPTS`

`PROMPTS` 的结构示例：

```python
PROMPT_VERSION = "v1.2.0"
PROMPT_DESCRIPTION = "实验版 prompt，强化时间限定词和实体对齐"

PROMPTS = {
    "intent": {
        "system_prompt": "...",
        "few_shot_examples": [
            {
                "title": "示例标题",
                "input": "query: ...",
                "output": "{\"intent_clear\": true}"
            }
        ],
    },
    "scoring": {
        "system_prompt": "...",
        "few_shot_examples": [],
    },
    "verify": {
        "system_prompt": "...",
        "few_shot_examples": [],
    },
}
```

运行结果里也会带上：

- `prompt_version`
- `prompt_description`
- `prompt_file`

便于做回归对比和效果归档。

## 测试

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python -m unittest discover -s tests
```

