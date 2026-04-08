INTENT_FEW_SHOT_EXAMPLES = [
    {
        "title": "意图明确，不需要补知识",
        "input": """query:
北京今天天气

doc:
北京市气象台今天发布多云转晴天气预报，白天最高气温 18 度。""",
        "output": """{
  "intent_clear": true,
  "understood_intent": "查询北京今天的天气情况",
  "task_objective": "找到北京当日天气信息",
  "entities": ["北京"],
  "qualifiers": ["今天"],
  "constraints": [],
  "missing_points": [],
  "should_use_kb": false,
  "kb_queries": [],
  "notes": "query 意图和时间限定都很明确，doc 也能帮助理解，不需要额外补知识"
}""",
    },
    {
        "title": "实体有歧义，需要补知识",
        "input": """query:
胖梦聚餐合影

doc:
网友发帖讨论一张多人聚餐照片，但没有说明“胖梦”具体指谁。""",
        "output": """{
  "intent_clear": false,
  "understood_intent": "用户想找与“胖梦”相关的聚餐合影内容",
  "task_objective": "确认“胖梦”所指实体并理解用户真正关注的合影事件",
  "entities": ["胖梦"],
  "qualifiers": ["聚餐", "合影"],
  "constraints": ["需要先确认“胖梦”是哪位人物或圈内称呼"],
  "missing_points": ["胖梦的真实指代对象", "具体是哪次聚餐场景"],
  "should_use_kb": true,
  "kb_queries": ["胖梦 指的是谁", "胖梦 聚餐 合影"],
  "notes": "query 中核心实体存在明显指代歧义，需要先补知识帮助理解意图"
}""",
    },
]


SCORING_FEW_SHOT_EXAMPLES = [
    {
        "title": "完全相关：需求被充分满足",
        "input": """query:
苹果手机13的电池容量

需求理解:
{
  "understood_intent": "查询 iPhone 13 的电池容量",
  "entities": ["iPhone 13"],
  "qualifiers": ["电池容量"]
}

doc:
iPhone 13 配备 3240mAh 电池，支持长时间视频播放。""",
        "output": """{
  "label": 3,
  "label_name": "3-完全相关",
  "reason": "doc 直接回答了 iPhone 13 的电池容量，语义、实体和需求点都精确匹配，且答案充分明确",
  "matched_points": ["命中 iPhone 13", "命中电池容量信息"],
  "missing_points": [],
  "evidence_spans": ["iPhone 13 配备 3240mAh 电池"],
  "confidence": "high"
}""",
    },
    {
        "title": "较相关：只满足集合需求的一部分",
        "input": """query:
悬疑小说推荐

需求理解:
{
  "understood_intent": "希望获取悬疑小说推荐，倾向于一个推荐集合而不是单篇作品",
  "entities": ["悬疑小说"],
  "qualifiers": ["推荐", "集合"]
}

doc:
《白夜行》是一部经典悬疑小说，情节紧凑，适合喜欢推理与黑暗风格的读者。""",
        "output": """{
  "label": 2,
  "label_name": "2-较相关",
  "reason": "doc 确实提供了一本悬疑小说推荐，满足了部分意图，但 query 更像需要一个推荐集合或多本候选，当前内容不够完整详实",
  "matched_points": ["命中悬疑小说", "命中推荐意图"],
  "missing_points": ["缺少更多推荐结果", "不够形成完整推荐集合"],
  "evidence_spans": ["《白夜行》是一部经典悬疑小说"],
  "confidence": "high"
}""",
    },
    {
        "title": "弱相关：有参考性但不满足真实意图",
        "input": """query:
房价为什么上涨

需求理解:
{
  "understood_intent": "想了解房价上涨的原因",
  "entities": ["房价"],
  "qualifiers": ["上涨", "原因"]
}

doc:
近期多个城市房价出现波动，房地产市场交易量有所回升，购房讨论热度持续增加。""",
        "output": """{
  "label": 1,
  "label_name": "1-弱相关",
  "reason": "doc 提供了房价上涨相关背景和市场现象，具备一定参考性，但没有真正回答“为什么上涨”这一原因型需求",
  "matched_points": ["主题与房价上涨相关"],
  "missing_points": ["缺少上涨原因分析"],
  "evidence_spans": ["房价出现波动", "交易量有所回升"],
  "confidence": "medium"
}""",
    },
    {
        "title": "不相关：提到关键词但主体信息偏移",
        "input": """query:
月鳞绮纪播出时间

需求理解:
{
  "understood_intent": "查询《月鳞绮纪》的播出时间信息",
  "entities": ["月鳞绮纪"],
  "qualifiers": ["播出时间"]
}

doc:
多条与《月鳞绮纪》相关话题登上热搜，主演鞠婧祎的税务争议引发关注。""",
        "output": """{
  "label": 0,
  "label_name": "0-不相关",
  "reason": "Doc 虽提到 Query 中的关键词，但主体内容是演员争议，未提供播出时间信息，也没有直接参考价值。",
  "matched_points": ["提到《月鳞绮纪》"],
  "missing_points": ["缺少播出时间", "主体内容偏向演员争议而非播出信息"],
  "evidence_spans": ["主演鞠婧祎的税务争议引发关注"],
  "confidence": "high"
}""",
    },
    {
        "title": "完全相关：直接给出演员表信息",
        "input": """query:
甄嬛传演员表

需求理解:
{
  "understood_intent": "查询《甄嬛传》的演员表信息",
  "entities": ["甄嬛传"],
  "qualifiers": ["演员表"]
}

doc:
《甄嬛传》由孙俪、陈建斌、蔡少芬、蒋欣等主演。""",
        "output": """{
  "label": 3,
  "label_name": "3-完全相关",
  "reason": "Doc 直接回答了 Query 所需的演员信息，实体和意图匹配，能够满足用户需求。",
  "matched_points": ["命中《甄嬛传》", "直接给出演员信息"],
  "missing_points": [],
  "evidence_spans": ["由孙俪、陈建斌、蔡少芬、蒋欣等主演"],
  "confidence": "high"
}""",
    },
    {
        "title": "弱相关：主题相关但不是演员表",
        "input": """query:
甄嬛传演员表

需求理解:
{
  "understood_intent": "查询《甄嬛传》的演员表信息",
  "entities": ["甄嬛传"],
  "qualifiers": ["演员表"]
}

doc:
孙俪在采访中谈到拍摄《甄嬛传》时的感受，并回忆了剧组生活。""",
        "output": """{
  "label": 1,
  "label_name": "1-弱相关",
  "reason": "Doc 与《甄嬛传》主题相关，但没有提供演员表，只是相关背景内容，未直接满足 Query 的核心意图。",
  "matched_points": ["主题与《甄嬛传》相关"],
  "missing_points": ["缺少演员表信息"],
  "evidence_spans": ["孙俪谈到拍摄《甄嬛传》时的感受"],
  "confidence": "high"
}""",
    },
    {
        "title": "弱相关：提供背景影响力但未回答原因",
        "input": """query:
鬼吹灯为什么这么火

需求理解:
{
  "understood_intent": "想了解《鬼吹灯》为什么这么火的原因",
  "entities": ["鬼吹灯"],
  "qualifiers": ["为什么这么火", "原因"]
}

doc:
《鬼吹灯》是知名盗墓题材 IP，包含多部小说和影视改编作品，长期拥有较高讨论度。""",
        "output": """{
  "label": 1,
  "label_name": "1-弱相关",
  "reason": "Doc 提供了鬼吹灯的背景和影响力信息，但没有直接回答“为什么这么火”的原因，因此只有参考性，不能直接满足 Query。",
  "matched_points": ["主题与《鬼吹灯》相关", "提供了一定背景信息"],
  "missing_points": ["缺少“为什么这么火”的原因分析"],
  "evidence_spans": ["是知名盗墓题材 IP", "长期拥有较高讨论度"],
  "confidence": "high"
}""",
    },
    {
        "title": "弱相关：丢失重要主体约束",
        "input": """query:
必胜客原神联动周边

需求理解:
{
  "understood_intent": "查询必胜客与原神联动的周边信息",
  "entities": ["必胜客", "原神"],
  "qualifiers": ["联动", "周边"]
}

doc:
必胜客推出联名套餐，套餐实惠且美味，必胜客门店随时享用""",
        "output": """{
  "label": 1,
  "label_name": "1-弱相关",
  "reason": "Doc只提及必胜客联名套餐，并未提及套餐是和原神联动，丢失了重要的主体约束，算弱相关。",
  "matched_points": ["命中必胜客", "涉及联名套餐主题"],
  "missing_points": ["缺少原神这一关键联动主体", "未提供周边信息"],
  "evidence_spans": ["必胜客推出联名套餐"],
  "confidence": "high"
}""",
    },
    {
        "title": "较相关：满足核心问题但信息不够详实",
        "input": """query:
鞠婧祎月鳞绮纪播出时间

需求理解:
{
  "understood_intent": "查询鞠婧祎相关作品《月鳞绮纪》的播出时间",
  "entities": ["鞠婧祎", "月鳞绮纪"],
  "qualifiers": ["播出时间"]
}

doc:
《月鳞绮纪》预计于暑期档上线，是部欢快古偶剧，敬请期待。""",
        "output": """{
  "label": 2,
  "label_name": "2-较相关",
  "reason": "Doc虽未提及鞠婧祎，但满足了用户对《月鳞绮纪》播出时间的核心需求，只是内容只有一句话，不够详实，算较相关。",
  "matched_points": ["命中《月鳞绮纪》", "回答了播出时间信息"],
  "missing_points": ["未提及鞠婧祎", "时间信息不够具体详实"],
  "evidence_spans": ["预计于暑期档上线"],
  "confidence": "medium"
}""",
    },
    {
        "title": "不相关：实体错配且无参考性",
        "input": """query:
刘德华演唱会时间

需求理解:
{
  "understood_intent": "查询刘德华演唱会举办时间",
  "entities": ["刘德华"],
  "qualifiers": ["演唱会", "时间"]
}

doc:
张学友巡回演唱会将于 5 月 1 日在上海开唱。""",
        "output": """{
  "label": 0,
  "label_name": "0-不相关",
  "reason": "doc 虽然也是演唱会时间信息，但主体是张学友，不是刘德华，实体错配，无法满足 query 的核心需求，也没有足够参考性",
  "matched_points": ["主题涉及演唱会时间"],
  "missing_points": ["未命中刘德华这一关键实体"],
  "evidence_spans": ["张学友巡回演唱会将于 5 月 1 日在上海开唱"],
  "confidence": "high"
}""",
    },
]


VERIFY_FEW_SHOT_EXAMPLES = [
    {
        "title": "理解不足，建议回到补知识",
        "input": """query:
胖梦聚餐合影

需求理解:
{
  "understood_intent": "查找胖梦相关聚餐照片",
  "entities": ["胖梦"],
  "qualifiers": ["聚餐", "合影"],
  "missing_points": ["胖梦是谁"]
}

doc:
某明星与朋友聚餐后晒出多人自拍。

当前标签结果:
{
  "label": 3,
  "label_name": "3-完全相关"
}""",
        "output": """{
  "passed": false,
  "issues": [
    {
      "type": "insufficient_understanding",
      "reason": "核心实体“胖梦”的指代仍不清楚，当前无法确认 doc 是否真的对应目标人物",
      "suggested_action": "retrieve_kb"
    }
  ],
  "verification_summary": "当前标签建立在不完整的需求理解上，质检不通过",
  "final_action": "retrieve_kb"
}""",
    },
    {
        "title": "仅标签偏高，建议重打分",
        "input": """query:
北京明天天气

需求理解:
{
  "understood_intent": "查询北京明天天气",
  "entities": ["北京"],
  "qualifiers": ["明天", "天气"]
}

doc:
北京市今天多云，最高气温 18 度。

当前标签结果:
{
  "label": 3,
  "label_name": "3-完全相关"
}""",
        "output": """{
  "passed": false,
  "issues": [
    {
      "type": "missing_qualifier",
      "reason": "doc 提供的是今天天气，不是明天天气，时间限定词没有满足",
      "suggested_action": "rescore"
    }
  ],
  "verification_summary": "主题相近，但关键时间限定词不匹配，当前标签偏高",
  "final_action": "rescore"
}""",
    },
]
