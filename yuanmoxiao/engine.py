"""
yuanmoxiao-cerebellum — Cognitive engine

Three-level funnel (zero model):
  Level 1: Smart cache hit — microsecond
  Level 2: Intent parse + template match — millisecond
  Level 3: Algorithm skeleton — pre-written logic

Plus optional DeepSeek adapter for token-space API calls.
"""

import json
import os
import re
import time
import hashlib
from .tokenizer import StandaloneTokenizer, CodeTokenMapper


# ════════════════════════════════════════════
# Code template library (46 templates × 6 langs)
# ════════════════════════════════════════════

_TEMPLATES = {
    # ── Python ──
    "py_def":           "def {name}({params}):\n    {body}",
    "py_async_def":     "async def {name}({params}):\n    {body}",
    "py_class":         "class {name}:\n    {body}",
    "py_init":          "def __init__(self, {params}):\n    self.{param_assign}",
    "py_if":            "if {condition}:\n    {body}",
    "py_if_else":       "if {condition}:\n    {body}\nelse:\n    {else_body}",
    "py_for":           "for {item} in {iterable}:\n    {body}",
    "py_while":         "while {condition}:\n    {body}",
    "py_try":           "try:\n    {body}\nexcept {error} as e:\n    {handler}",
    "py_with":          "with {expr} as {var}:\n    {body}",
    "py_list_comp":     "[{expr} for {item} in {iterable}]",
    "py_lambda":        "lambda {params}: {expr}",
    "py_decorator":     "@{deco}\ndef {name}({params}):\n    {body}",
    "py_main":          "def main():\n    {body}\n\n\nif __name__ == '__main__':\n    main()",
    "py_import":        "import {module}",
    "py_from_import":   "from {module} import {names}",
    # FastAPI
    "py_api_get":       "@app.get('/{path}')\nasync def get_{name}({params}):\n    return {result}",
    "py_api_post":      "@app.post('/{path}')\nasync def post_{name}({params}):\n    return {result}",
    "py_api_put":       "@app.put('/{path}/{id}')\nasync def put_{name}({id}: int, {params}):\n    return {result}",
    "py_api_delete":    "@app.delete('/{path}/{id}')\nasync def delete_{name}({id}: int):\n    return {result}",
    # SQLAlchemy
    "py_db_model":      "class {name}(Base):\n    __tablename__ = '{table}'\n\n    {id}: Mapped[int] = Column(Integer, primary_key=True)\n    {fields}",
    "py_db_query":      "session.query({model}).filter({condition}).all()",
    # ── Go ──
    "go_func":          "func {name}({params}) {returns} {{\n    {body}\n}}",
    "go_main":          "func main() {{\n    {body}\n}}",
    "go_struct":        "type {name} struct {{\n    {fields}\n}}",
    "go_if":            "if {condition} {{\n    {body}\n}}",
    "go_for":           "for {init}; {cond}; {post} {{\n    {body}\n}}",
    "go_http_handler":  "func {name}(w http.ResponseWriter, r *http.Request) {{\n    {body}\n}}",
    "go_goroutine":     "go {call}",
    # ── JavaScript / TypeScript ──
    "js_function":      "function {name}({params}) {{\n    {body}\n}}",
    "js_arrow":         "const {name} = ({params}) => {{\n    {body}\n}}",
    "js_class":         "class {name} {{\n    constructor({params}) {{\n        {body}\n    }}\n}}",
    "js_async":         "async function {name}({params}) {{\n    {body}\n}}",
    "js_export":        "export {stmt}",
    "js_react_component": "function {name}({params}) {{\n  return (\n    <div>\n      {jsx}\n    </div>\n  );\n}}",
    # ── Rust ──
    "rs_fn":            "fn {name}({params}) -> {returns} {{\n    {body}\n}}",
    "rs_struct":        "struct {name} {{\n    {fields}\n}}",
    "rs_impl":          "impl {name} {{\n    fn new({params}) -> Self {{\n        Self {{ {fields} }}\n    }}\n}}",
    "rs_main":          "fn main() {{\n    {body}\n}}",
    "rs_match":         "match {expr} {{\n    {arms}\n}}",
    # ── C++ ──
    "cpp_function":     "{return_type} {name}({params}) {{\n    {body}\n}}",
    "cpp_class":        "class {name} {{\npublic:\n    {name}({params});\n    {methods}\nprivate:\n    {fields}\n}};",
    "cpp_main":         "int main(int argc, char* argv[]) {{\n    {body}\n    return 0;\n}}",
    # ── Java ──
    "java_class":       "public class {name} {{\n    {fields}\n    public {name}({params}) {{\n        {body}\n    }}\n}}",
    "java_method":      "public {return_type} {name}({params}) {{\n    {body}\n}}",
    "java_main":        "public static void main(String[] args) {{\n    {body}\n}}",
}


# ════════════════════════════════════════════
# Algorithm skeletons (pre-written)
# ════════════════════════════════════════════

_ALGORITHM_SKELETONS = {
    "binary_search": {
        "python": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
        "go": """func BinarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1}""",
        "js": """function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}""",
    },
    "quicksort": {
        "python": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",
        "go": """func QuickSort(arr []int) []int {
    if len(arr) <= 1 { return arr }
    pivot := arr[len(arr)/2]
    var left, middle, right []int
    for _, v := range arr {
        if v < pivot { left = append(left, v) } else if v == pivot { middle = append(middle, v) } else { right = append(right, v) }
    }
    result := append(QuickSort(left), middle...)
    return append(result, QuickSort(right)...)
}""",
    },
    "reverse_linked_list": {
        "python": """def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev""",
    },
    "fibonacci": {
        "python": """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
    },
    "bubble_sort": {
        "python": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
    },
    "climb_stairs": {
        "python": """def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b""",
    },
    "is_palindrome": {
        "python": """def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]""",
    },
    "two_sum": {
        "python": """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []""",
    },
    "max_subarray": {
        "python": """def max_subarray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum""",
    },
    "http_server": {
        "python": """from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Hello, world!')

server = HTTPServer(('0.0.0.0', 8080), Handler)
print('Server running on port 8080...')
server.serve_forever()""",
        "go": """package main

import (
    "fmt"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, world!")
}

func main() {
    http.HandleFunc("/", handler)
    fmt.Println("Server running on :8080")
    http.ListenAndServe(":8080", nil)
}""",
    },
}


# ════════════════════════════════════════════
# Smart cache — "越用越省" core
# ════════════════════════════════════════════

class SmartCache:
    """
    Multi-level cache with LFU+LRU hybrid eviction.

    Level 1a: Exact intent hash — instant hit
    Level 1b: Fuzzy intent match — normalized query matches cached
    Level 1c: Pattern association — pre-load related patterns

    Tracks hit rates and auto-promotes hot patterns.
    """

    def __init__(self, max_size: int = 512, ttl_seconds: int = 3600):
        self._cache = {}       # hash → {result, freq, last_access, created}
        self._fuzzy_index = {} # normalized_key → list of hashes
        self._assoc_index = {} # pattern → [related patterns]
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _intent_hash(self, description: str, lang: str) -> str:
        """Deterministic hash for exact matching."""
        raw = f"{description.strip().lower()}:{lang}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _normalize(self, description: str) -> str:
        """Normalize query for fuzzy matching — strip stopwords, normalize spaces."""
        text = description.lower().strip()
        # remove common filler words
        for w in ["请", "帮", "我", "一个", "的", "用", "写", "创建", "生成", "做个", "实现"]:
            text = text.replace(w, "")
        # collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_associations(self):
        """Pre-compute pattern associations based on co-occurrence."""
        # Common task sequences developers follow
        pairs = [
            ("binary_search", "quicksort"),
            ("binary_search", "two_sum"),
            ("api_get", "api_post"),
            ("api_post", "api_put"),
            ("api_put", "api_delete"),
            ("class", "init"),
            ("def", "main"),
            ("http_server", "crud"),
        ]
        self._assoc_index = {}
        for a, b in pairs:
            self._assoc_index.setdefault(a, set()).add(b)
            self._assoc_index.setdefault(b, set()).add(a)

    def get(self, description: str, lang: str = "python") -> dict | None:
        """Try all cache levels. Returns cached result or None."""
        h = self._intent_hash(description, lang)

        # Level 1a: exact match
        if h in self._cache:
            entry = self._cache[h]
            if time.time() - entry["created"] < self.ttl:
                entry["freq"] += 1
                entry["last_access"] = time.time()
                self.hits += 1
                result = dict(entry["result"])
                result["cache_level"] = "1a_exact"
                result["cache_hit"] = True
                return result
            else:
                # expired
                del self._cache[h]

        # Level 1b: fuzzy match
        normalized = self._normalize(description)
        for cached_h, candidates in self._fuzzy_index.items():
            # if normalized query is similar to cached pattern
            if cached_h in normalized or normalized in cached_h:
                match_hash = candidates[0] if candidates else None
                if match_hash and match_hash in self._cache:
                    entry = self._cache[match_hash]
                    if time.time() - entry["created"] < self.ttl:
                        self.hits += 1
                        result = dict(entry["result"])
                        result["cache_level"] = "1b_fuzzy"
                        result["cache_hit"] = True
                        return result

        self.misses += 1
        return None

    def set(self, description: str, lang: str, result: dict):
        """Store result in cache."""
        h = self._intent_hash(description, lang)

        # Evict if full
        if len(self._cache) >= self.max_size:
            self._evict_one()

        self._cache[h] = {
            "result": result,
            "freq": 1,
            "last_access": time.time(),
            "created": time.time(),
        }

        # Update fuzzy index
        normalized = self._normalize(description)
        self._fuzzy_index.setdefault(normalized, []).append(h)

    def _evict_one(self):
        """LFU + LRU hybrid: evict item with lowest (freq / age)."""
        if not self._cache:
            return
        now = time.time()
        def score(entry):
            age = now - entry["created"]
            return entry["freq"] / max(age, 1)
        worst_key = min(self._cache, key=lambda k: score(self._cache[k]))
        del self._cache[worst_key]

    def get_related(self, pattern: str) -> list:
        """Get related patterns for pre-loading."""
        if not self._assoc_index:
            self._build_associations()
        return list(self._assoc_index.get(pattern, []))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 1) if total > 0 else 0,
            "cache_size": len(self._cache),
            "max_size": self.max_size,
        }


# ════════════════════════════════════════════
# Intent parser
# ════════════════════════════════════════════

class IntentParser:
    """
    Parse natural language → structured template parameters.
    Pure rule-based, zero model calls.
    """

    LANG_ALIASES = {
        "python": ["python", "py"],
        "go": ["go", "golang"],
        "js": ["javascript", "js", "node", "nodejs"],
        "ts": ["typescript", "ts"],
        "rust": ["rust", "rs"],
        "cpp": ["c++", "cpp", "cplusplus"],
        "java": ["java"],
    }

    PATTERN_KEYWORDS = {
        "def": ["def ", "函数", "function", "方法", "method"],
        "class": ["class ", "类", "类名"],
        "api_get": ["get接口", "查询接口", "获取", "get api", "list api"],
        "api_post": ["post接口", "创建接口", "新增", "添加", "post api", "create api"],
        "api_put": ["put接口", "更新接口", "修改", "编辑", "put api", "update api"],
        "api_delete": ["delete接口", "删除接口", "delete api"],
        "for": ["for循环", "遍历", "循环"],
        "if": ["if判断", "条件判断", "判断"],
        "try": ["try", "异常处理", "错误处理"],
        "main": ["主函数", "main ", "入口", "程序入口"],
        "async_def": ["异步", "async def", "协程"],
    }

    def parse(self, desc: str, lang: str = "python") -> dict:
        desc_lower = desc.lower().strip()

        # 1. Detect language
        language = self._detect_language(desc, lang)

        # 2. Check if algorithm match (handled by caller)
        # 3. Detect code pattern
        pattern = self._match_pattern(desc)

        # 4. Extract params
        params = self._extract_params(desc, pattern, language)

        return {
            "source": "template" if pattern else "unknown",
            "language": language,
            "pattern": pattern,
            "params": params,
            "confidence": 0.7 if pattern else 0.2,
        }

    def _detect_language(self, desc: str, default: str) -> str:
        desc_lower = desc.lower()
        for lang, aliases in self.LANG_ALIASES.items():
            sorted_a = sorted(aliases, key=len, reverse=True)
            for alias in sorted_a:
                if len(alias) <= 2:
                    if re.search(r'\b' + re.escape(alias.lower()) + r'\b', desc_lower):
                        return lang
                else:
                    if alias.lower() in desc_lower:
                        return lang
        return default

    def _match_pattern(self, desc: str) -> str | None:
        desc_lower = desc.lower()
        for pattern, keywords in self.PATTERN_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in desc_lower:
                    return pattern
        if "函数" in desc_lower or "function" in desc_lower:
            return "def"
        if "类" in desc_lower:
            return "class"
        return None

    def _extract_params(self, desc: str, pattern: str, lang: str) -> dict:
        params = {"body": "pass", "result": "[]", "handler": "pass"}
        if not pattern:
            return params
        words = desc.replace(",", " ").replace("(", " ").replace(")", " ").split()
        stop_words = {"写", "创建", "生成", "做个", "用", "一个", "的", "要",
                      "把", "给", "让", "在", "实现", "函数", "方法", "接口",
                      "api", "get", "post", "put", "delete", "http",
                      "def", "class", "func", "function", "main"}
        name_candidates = [w for w in words if w.lower() not in stop_words and len(w) >= 2]

        if pattern in ("def", "async_def"):
            params["name"] = name_candidates[-1] if name_candidates else "handler"
            params["params"] = "request" if "request" in desc.lower() else "data"
        elif pattern in ("api_get", "api_post", "api_put", "api_delete"):
            params["name"] = name_candidates[-1] if name_candidates else "items"
            params["path"] = params["name"]
            params["params"] = "request: Request"
            params["result"] = '{"message": "ok"}'
        elif pattern == "class":
            params["name"] = name_candidates[-1] if name_candidates else "MyClass"
        elif pattern == "main":
            params["body"] = "pass"
        elif pattern == "for":
            params["item"] = "item"
            params["iterable"] = "items"
        return params


# ════════════════════════════════════════════
# CerebellumCognitiveBody (main class)
# ════════════════════════════════════════════

class CerebellumCognitiveBody:
    """
    Zero-model code understanding engine.

    Three-level funnel:
      L1: Smart cache — instant replay of known patterns
      L2: Intent parse + template/algorithm match — no model calls
      L3: Unknown → returns skeleton for manual filling

    Usage:
        cb = CerebellumCognitiveBody()
        # Understand code
        result = cb.understand("def hello(): return 'hi'")
        # Generate code from description
        result = cb.generate("写一个二分查找")
        # Learn from real usage
        cb.learn("写一个二分查找", "python", generated_code)
    """

    def __init__(self, tokenizer_path: str = None):
        self.tokenizer = StandaloneTokenizer(tokenizer_path)
        self.mapper = CodeTokenMapper()
        self.intent_parser = IntentParser()
        self.cache = SmartCache()
        self._stats = {"understands": 0, "generates": 0, "learns": 0}

    # ── Understand ──

    def understand(self, code: str) -> dict:
        """Analyze code from token perspective."""
        self._stats["understands"] += 1
        tokens = self.tokenizer.encode(code)
        analysis = {
            "total_tokens": len(tokens),
            "is_code": self._is_code_like(code),
            "keyword_tokens": self._find_keywords(tokens),
            "sample_tokens": [
                {"id": t, "text": self.tokenizer.decode([t])}
                for t in tokens[:5]
            ],
            "language": self._guess_language(code),
        }
        return {"tokens": tokens, "analysis": analysis}

    # ── Generate ──

    def generate(self, description: str, lang: str = "python") -> dict:
        """
        Generate code from natural language description.

        Returns:
            {"level": 1|2|3, "source": "...", "code": "...", ...}
            Level 1 = cache hit (fastest)
            Level 2 = template/algorithm match (no model)
            Level 3 = unknown (needs manual fill)
        """
        self._stats["generates"] += 1

        # Level 1: Smart cache
        cached = self.cache.get(description, lang)
        if cached:
            cached["level"] = 1
            return cached

        # Level 2: Intent parse → algorithm match → template fill

        # 2a: Check algorithm skeletons
        for algo_name, skeletons in _ALGORITHM_SKELETONS.items():
            if self._match_algorithm(description, algo_name):
                code = skeletons.get(lang) or skeletons.get("python", "")
                if not code:
                    continue
                result = {
                    "level": 2,
                    "source": "algorithm",
                    "algorithm": algo_name,
                    "language": lang,
                    "code": code,
                }
                # Cache it for next time
                self.cache.set(description, lang, result)
                return result

        # 2b: Template match
        intent = self.intent_parser.parse(description, lang)
        if intent["source"] == "template" and intent["pattern"]:
            lang_short = {"python": "py", "javascript": "js",
                         "typescript": "ts", "rust": "rs",
                         "cpp": "cpp", "java": "java"}.get(
                intent["language"], intent["language"])
            template_key = f"{lang_short}_{intent['pattern']}"
            tmpl = _TEMPLATES.get(template_key) or _TEMPLATES.get(intent["pattern"])

            if tmpl:
                code = tmpl.format(**intent["params"])
                result = {
                    "level": 2,
                    "source": "template",
                    "pattern": intent["pattern"],
                    "language": intent["language"],
                    "code": code,
                    "params": intent["params"],
                }
                self.cache.set(description, lang, result)
                return result

        # Level 3: Unknown
        return {
            "level": 3,
            "source": "unknown",
            "language": lang,
            "code": f"# TODO: {description}\n# cerebellum cannot handle this yet\n# add template or use DeepSeek adapter\npass",
        }

    # ── Learn (feedback loop: turns Level 3 into Level 1) ──

    def learn(self, description: str, lang: str, code: str):
        """
        Learn from a real execution result.
        Next time the same description comes, it'll hit L1 cache.
        """
        self._stats["learns"] += 1
        result = {
            "level": 1,
            "source": "learned",
            "language": lang,
            "code": code,
            "learned": True,
        }
        self.cache.set(description, lang, result)
        # Also store in SQLite for persistence
        tokens = self.tokenizer.encode(code)
        self.mapper.add_pattern(description[:200], tokens)
        self.mapper.record_usage(self.cache._intent_hash(description, lang),
                                 self._guess_code_pattern(code), lang)

    # ── Stats ──

    def stats(self) -> dict:
        return {
            "tokenizer": {"vocab_size": self.tokenizer.vocab_size},
            "templates": {
                "total": len(_TEMPLATES),
                "languages": len(set(k.split("_")[0] for k in _TEMPLATES)),
            },
            "algorithms": len(_ALGORITHM_SKELETONS),
            "cache": self.cache.stats(),
            "usage": dict(self._stats),
        }

    # ── Internal helpers ──

    def _is_code_like(self, text: str) -> bool:
        indicators = ["def ", "class ", "import ", "return ",
                     "if ", "for ", "while ", "try:",
                     "fn ", "func ", "impl ",
                     "function ", "=>", "->",
                     "{", "}", ";"]
        return sum(1 for ind in indicators if ind in text[:200]) >= 2

    def _find_keywords(self, tokens: list) -> list:
        known = {750: "def", 460: "class", 474: "import", 333: "if",
                689: "return", 396: "int", 501: "for", 352: "while",
                540: "try", 564: "except"}
        return [{"id": t, "text": known[t]} for t in tokens[:30] if t in known]

    def _guess_language(self, code: str) -> str:
        if "def " in code or "import " in code:
            return "python"
        if "func " in code or "package " in code:
            return "go"
        if "function " in code or "const " in code:
            return "js"
        if "fn " in code or "impl " in code:
            return "rust"
        return "unknown"

    def _guess_code_pattern(self, code: str) -> str:
        for kw in ["def ", "class ", "import ", "for ", "if ", "try:"]:
            if kw in code[:100]:
                return kw.strip().rstrip(":")
        return "other"

    def _match_algorithm(self, desc: str, algo_name: str) -> bool:
        """Check if description matches an algorithm by name/keywords."""
        desc_clean = desc.lower().replace(" ", "")
        algo_keywords = {
            "binary_search": ["二分", "binarysearch", "折半"],
            "quicksort": ["快排", "快速排序", "quicksort"],
            "reverse_linked_list": ["反转链表", "链表反转", "reverselist"],
            "fibonacci": ["斐波那契", "fibonacci"],
            "bubble_sort": ["冒泡", "bubblesort"],
            "climb_stairs": ["爬楼梯", "climbstairs"],
            "is_palindrome": ["回文", "palindrome"],
            "two_sum": ["两数之和", "twosum"],
            "max_subarray": ["最大子数组", "最大子序", "maxsubarray"],
            "http_server": ["httpserver", "服务器", "webserver"],
        }
        keywords = algo_keywords.get(algo_name, [algo_name.replace("_", "")])
        return any(kw in desc_clean for kw in keywords)


# ════════════════════════════════════════════
# DeepSeek adapter (optional, needs httpx)
# ════════════════════════════════════════════

class DeepSeekAdapter:
    """
    Talk to DeepSeek API in token space.

    The cerebellum handles simple patterns locally (free).
    Complex ones fall back to DeepSeek (paid) — but cerebellum
    learns from the result, so next time it's free.

    "The more you use it, the less it costs."
    """

    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat", cerebellum: CerebellumCognitiveBody = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.cb = cerebellum or CerebellumCognitiveBody()
        self._deepseek_hits = 0
        self._local_hits = 0

    def generate(self, description: str, lang: str = "python") -> dict:
        """
        Generate code using cerebellum-first, DeepSeek-fallback strategy.
        Result auto-learned for future cache hits.
        """
        # Step 1: Try cerebellum (free)
        local = self.cb.generate(description, lang)
        if local["level"] in (1, 2):
            self._local_hits += 1
            local["saved_tokens"] = 0
            return local

        # Step 2: Fall back to DeepSeek (paid)
        result = self._call_deepseek(description, lang)
        self._deepseek_hits += 1

        # Step 3: Learn from result (next time → free)
        if result.get("code"):
            self.cb.learn(description, lang, result["code"])

        result["local_hits"] = self._local_hits
        result["deepseek_hits"] = self._deepseek_hits
        return result

    def _call_deepseek(self, description: str, lang: str) -> dict:
        """Actually call DeepSeek chat API."""
        try:
            import httpx
        except ImportError:
            return {
                "level": 3,
                "source": "error",
                "error": "httpx not installed. Run: pip install yuanmoxiao-cerebellum[deepseek]",
            }

        if not self.api_key:
            return {
                "level": 3,
                "source": "error",
                "error": "DEEPSEEK_API_KEY not set",
            }

        prompt = f"Write {lang} code for: {description}\nReturn ONLY the code, no explanation."
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            data = resp.json()
            code = data["choices"][0]["message"]["content"]
            # Extract code block if wrapped in markdown
            if "```" in code:
                parts = code.split("```")
                code = parts[1] if len(parts) >= 2 else code
                if "\n" in code:
                    code = code[code.index("\n")+1:]
            return {
                "level": 2,
                "source": "deepseek",
                "language": lang,
                "code": code.strip(),
                "tokens_used": data.get("usage", {}).get("total_tokens", 0),
            }
        except Exception as e:
            return {
                "level": 3,
                "source": "error",
                "error": str(e),
            }

    def stats(self) -> dict:
        return {
            "local_hits": self._local_hits,
            "deepseek_calls": self._deepseek_hits,
            "savings": f"{self._local_hits}/{self._local_hits + self._deepseek_hits} "
                       f"local ({round(self._local_hits / max(1, self._local_hits + self._deepseek_hits) * 100)}%)",
        }


if __name__ == "__main__":
    print("=" * 60)
    print("yuanmoxiao-cerebellum — Zero-model verification")
    print("=" * 60)

    cb = CerebellumCognitiveBody()

    # 1. Token understanding
    print("\n--- 1. Token understanding ---")
    for code in [
        "def hello():\n    return 'world'",
        "for i in range(10):\n    print(i)",
    ]:
        r = cb.understand(code)
        kw = [k["text"] for k in r["analysis"]["keyword_tokens"]]
        print(f"  [{r['analysis']['total_tokens']}t] {r['analysis']['language']} kw:{kw}")

    # 2. Code generation (no model!)
    print("\n--- 2. Code generation (zero model) ---")
    tests = [
        "写一个def hello函数",
        "创建api get接口 users",
        "用python实现一个二分查找",
        "用go写一个binary search",
        "写一个快速排序",
        "写一个两数之和",
        "创建CRUD",
    ]
    for desc in tests:
        r = cb.generate(desc)
        lvl = r.get("level", 3)
        src = r.get("source", "?")
        preview = r.get("code", "")[:50].replace("\n", " ↵ ")
        print(f"  {'✓' if lvl < 3 else '?'} L{lvl} [{src}] {preview}")

    # 3. Cache test (second call → instant)
    print("\n--- 3. Cache test ---")
    r1 = cb.generate("写一个二分查找")
    print(f"  First:  L{r1['level']} (expect 2)")
    r2 = cb.generate("写一个二分查找")
    print(f"  Second: L{r2['level']} (expect 1 — cache hit!)")

    # 4. Stats
    print("\n--- 4. Stats ---")
    st = cb.stats()
    print(f"  Tokenizer: {st['tokenizer']['vocab_size']} vocab")
    print(f"  Templates: {st['templates']['total']} / {st['templates']['languages']} langs")
    print(f"  Algorithms: {st['algorithms']}")
    print(f"  Cache: {st['cache']}")
