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

这个 provider 默认不要求 API Key。

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

如果想看到更完整的日志和进度提示，可以加上：

```bash
conda run -n base python run.py --verbose ...
```

开启后会输出：

- 当前运行配置
- `需求理解 / 补充知识检索 / 相关性打分 / 二次验证` 的阶段进度
- 本地知识库和在线知识源返回数量
- 最终标签、验证结果和输出文件位置

## Prompt 配置

默认会从 `src/auto_label_agent/prompts/system_prompts.py` 读取三段系统 prompt。你可以只改这一个文件，而不需要改 Python 逻辑代码。

如果你想切换成另一套 prompt，可以传：

```bash
--prompt-file path\to\system_prompts.py
```

Python 文件内需要导出：

- `PROMPTS`
- `intent`
- `scoring`
- `verify`

## 测试

```bash
cd d:\微博\trae_test\auto_label_agent
conda run -n base python -m unittest discover -s tests
```

