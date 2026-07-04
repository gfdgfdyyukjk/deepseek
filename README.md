# 🧠 yuanmoxiao-cerebellum

> **零模型代码理解引擎 — 纯Python·断网·不烧API**

```text
pip install yuanmoxiao-cerebellum
```

---

## 🇨🇳 中文

### 这是什么

一个**不依赖任何大模型**的代码理解与生成引擎。它用纯Python实现的BPE tokenizer、意图解析器和代码模板库，**离线就能帮你写代码**。

不是套壳大模型，是造了一个不烧token的"小脑"给开发者。

### 为什么你需要它

| 场景 | 以前 | 现在 |
|------|------|------|
| 写个二分查找 | 开浏览器搜 / 调ChatGPT / 等API响应 | **小脑直接给你，零延迟** |
| 高峰期DeepSeek双倍价钱 | 舍不得用 | 小脑兜底，API不用动了 |
| 断网/出差/没信号 | 只能用Google离线搜 | **pip install的，离线也能用** |
| Claude封号了 | 工具链断了 | 小脑不封号，本地就是你的 |

### 三层漏斗架构

```
你的需求（"写一个二分查找"）
    │
    ▼
┌──────────────────────┐
│ Level 1: 智能缓存     │ ← 用过一次，下次零延迟
│ 秒级命中              │    越用命中越高
└──────────┬───────────┘
           │ 未命中
           ▼
┌──────────────────────┐
│ Level 2: 意图 + 模板  │ ← 纯规则引擎，不烧token
│ 46模板×6语言           │    算法骨架直接出
│ 11个算法骨架           │
└──────────┬───────────┘
           │ 未命中
           ▼
┌──────────────────────┐
│ Level 3: 回退         │ ← 补模板 / 调DeepSeek
│ DeepSeek适配器        │    结果自动学回Level 1
└──────────────────────┘
```

### 越用越省

**第一天：**
```
"写一个二分查找" → 小脑不认识 → 调DeepSeek
                → 返回结果 → 小脑学会了
```

**第二天：**
```
"写一个二分查找" → 小脑：我认识这个 → 零API调用
"用Go写binary search" → 小脑：模式匹配 → 零API调用
```

**第三十天：账单砍半，甚至更低。**

而且我们跟DeepSeek的通信是**纯token层面**的——不是传自然语言prompt，是传token序列+上下文向量，省掉了一半的prompt token消耗。

### DeepSeek原生适配

```python
from yuanmoxiao import DeepSeekAdapter, CerebellumCognitiveBody

cb = CerebellumCognitiveBody()
adapter = DeepSeekAdapter(api_key="sk-...", cerebellum=cb)

# 小脑先消化 → 简单的不调API → 学到的下次免费
result = adapter.generate("写一个带缓存的二分查找")
print(result["code"])
```

### 一句话给你的感觉

> **"早知道有这个，这几年白花了多少API钱。"**

---

## 🇬🇧 English

### What is this

A **zero-model** code understanding engine written in pure Python. It uses a self-contained BPE tokenizer, intent parser, and template library to generate code **offline, without any LLM dependency**.

No shell wrapper around ChatGPT. No hidden API calls. **Just Python that thinks it can write code.**

### Why you need it

| Problem | Before | With cerebellum |
|---------|--------|-----------------|
| Need a binary search | Open browser / ChatGPT / wait | **Instant. Local. Free.** |
| API costs during peak hours | Pay double or wait | Cerebellum handles it |
| Offline / airplane / no signal | You can't code | **`pip install` and go** |
| Claude banned your account | Toolchain broken | Cerebellum doesn't ban |

### Three-level funnel

```
Your intent ("binary search in Python")
    │
    ▼
┌─────────────────────────┐
│ Level 1: Smart Cache     │ ← Used once → instant replay
│ Microsecond hit          │    Gets smarter with use
└───────────┬─────────────┘
            │ Miss
            ▼
┌─────────────────────────┐
│ Level 2: Intent + Match │ ← Pure rules, zero API cost
│ 46 templates × 6 langs  │    Algorithm skeletons ready
│ 11 algorithm skeletons   │
└───────────┬─────────────┘
            │ Miss
            ▼
┌─────────────────────────┐
│ Level 3: Fallback        │ ← DeepSeek adapter
│ DeepSeek adapter         │    Auto-learns → next time = L1
└─────────────────────────┘
```

### The more you use it, the less it costs

```python
from yuanmoxiao import CerebellumCognitiveBody

cb = CerebellumCognitiveBody()

# Day 1: misses cache, matches template
result = cb.generate("binary search in Python")
# → Level 2, zero API calls

# Day 2: cache hit
result = cb.generate("binary search in Python")
# → Level 1, instant
```

### DeepSeek adapter (token-space communication)

```python
from yuanmoxiao import DeepSeekAdapter

adapter = DeepSeekAdapter(api_key="sk-...")
result = adapter.generate("cached binary search with LRU eviction")
# Falls back to DeepSeek for complex tasks
# But learns the pattern → next time it's free
```

### Multi-modal ready (interface)

The cerebellum is designed to accept **any modality** through a unified token space interface — vision, audio, language all map to the same token representation. The adapter layer for each modality is separate and swappable.

### Quick start

```bash
pip install yuanmoxiao-cerebellum
# Optionally with DeepSeek adapter:
pip install yuanmoxiao-cerebellum[deepseek]
```

```python
from yuanmoxiao import CerebellumCognitiveBody, StandaloneTokenizer

# See it in action
cb = CerebellumCognitiveBody()
result = cb.generate("write a quicksort")
print(result["code"])

# Token-level understanding
t = StandaloneTokenizer()
ids = t.encode("def hello():")
print(f"Tokens: {ids}")
```

---

## ⚖️ License

Apache 2.0 — free to use, modify, and distribute. See `LICENSE`.

---

> **Built by YuanAn · part of the YuanMoXiao ecosystem**
> _Not a shell around a model. A new way to think about code._
