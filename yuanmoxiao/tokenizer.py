"""
yuanmoxiao-cerebellum — Tokenizer module

Pure Python BPE tokenizer (GPT-2 ByteLevel style).
Zero external dependencies — no tokenizers lib, no numpy, no torch.

Usage:
    from yuanmoxiao import StandaloneTokenizer
    t = StandaloneTokenizer()
    ids = t.encode("def hello():")
    text = t.decode(ids)
"""

import json
import os
import sqlite3
import urllib.request
import shutil
from pathlib import Path


# ── Byte-level encoder (GPT-2 style) ──

def _gpt2_byte_encoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _gpt2_byte_encoder()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}

# Default tokenizer: Qwen2.5 (BPE family, compatible with most code models)
_DEFAULT_TOKENIZER_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/tokenizer.json"
)
_DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".cache", "yuanmoxiao")


def _ensure_tokenizer(path: str | None) -> str:
    """Auto-download tokenizer.json if not present."""
    if path and os.path.exists(path):
        return path
    if path:
        return path  # caller specified a path that doesn't exist; let error surface

    cache_dir = _DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, "tokenizer.json")
    if os.path.exists(cached):
        return cached

    print(f"[yuanmoxiao] Downloading tokenizer from {_DEFAULT_TOKENIZER_URL} ...")
    try:
        urllib.request.urlretrieve(_DEFAULT_TOKENIZER_URL, cached)
        print(f"[yuanmoxiao] Tokenizer saved to {cached}")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download tokenizer.json.\n"
            f"URL: {_DEFAULT_TOKENIZER_URL}\n"
            f"Error: {e}\n\n"
            f"Alternatively, place your own tokenizer.json at:\n"
            f"  {cached}\n"
            f"or pass the path to StandaloneTokenizer(path='...')."
        )
    return cached


# ── Standalone BPE Tokenizer ──

class StandaloneTokenizer:
    """
    Pure Python BPE tokenizer — zero external dependencies.

    Auto-downloads Qwen2.5 tokenizer.json on first use,
    or accepts a custom path.

    Args:
        path: Path to tokenizer.json. If None, auto-downloads to ~/.cache/yuanmoxiao/
    """

    def __init__(self, path: str = None):
        self.path = _ensure_tokenizer(path)
        self.vocab = {}
        self.id_to_token = {}
        self.merges = {}
        self.special_tokens = {}
        self.byte_encoder = _BYTE_ENCODER
        self.byte_decoder = _BYTE_DECODER
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model = data.get("model", {})
        self.vocab = model.get("vocab", {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        merge_list = model.get("merges", [])
        for i, m in enumerate(merge_list):
            self.merges[tuple(m.split())] = i
        for at in data.get("added_tokens", []):
            self.special_tokens[at["content"]] = at["id"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list:
        """Text → list of token IDs."""
        for st, sid in sorted(self.special_tokens.items(), key=lambda x: -len(x[0])):
            if st in text:
                parts = text.split(st)
                result = []
                for i, part in enumerate(parts):
                    if part:
                        result.extend(self._encode_piece(part))
                    if i < len(parts) - 1:
                        result.append(sid)
                return result
        return self._encode_piece(text)

    def encode_batch(self, texts: list) -> list:
        return [self.encode(t) for t in texts]

    def decode(self, tokens: list) -> str:
        """Token ID list → text."""
        bytes_list = []
        for tid in tokens:
            token_str = self.id_to_token.get(tid, '')
            for ch in token_str:
                b = self.byte_decoder.get(ch, ord(ch))
                bytes_list.append(b)
        try:
            return bytes(bytes_list).decode('utf-8', errors='replace')
        except Exception:
            return ''.join(self.id_to_token.get(t, '') for t in tokens)

    def decode_batch(self, batch: list) -> list:
        return [self.decode(b) for b in batch]

    def _encode_piece(self, text: str) -> list:
        words = self._pre_tokenize(text)
        result = []
        for word in words:
            byte_chars = [self.byte_encoder[b] for b in word.encode('utf-8')]
            tokens = self._bpe_merge(''.join(byte_chars))
            result.extend(tokens)
        return result

    def _pre_tokenize(self, text: str) -> list:
        """GPT-2 style ByteLevel pre-tokenizer."""
        tokens = []
        i = 0
        while i < len(text):
            if text[i] in (' ', '\t', '\n', '\r'):
                j = i
                while j < len(text) and text[j] in (' ', '\t', '\n', '\r'):
                    j += 1
                tokens.append(('space', text[i:j]))
                i = j
            elif text[i].isalnum() or text[i] == '_':
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                tokens.append(('word', text[i:j]))
                i = j
            else:
                tokens.append(('punct', text[i]))
                i += 1
        result = []
        for idx, (typ, tok) in enumerate(tokens):
            if typ == 'space':
                continue
            elif idx == 0:
                result.append(tok)
            elif idx > 0 and tokens[idx-1][0] == 'space':
                result.append(' ' + tok)
            else:
                result.append(tok)
        return result

    def _bpe_merge(self, token_str: str) -> list:
        if token_str in self.vocab:
            return [self.vocab[token_str]]
        chars = list(token_str)
        if len(chars) == 1:
            return [self.vocab.get(chars[0], 0)]
        pairs = {}
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i+1])
            pairs[pair] = self.merges.get(pair, 99999999)
        while pairs:
            best_pair = min(pairs, key=lambda p: pairs[p])
            if pairs[best_pair] == 99999999:
                break
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                    new_chars.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            pairs = {}
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pairs[pair] = self.merges.get(pair, 99999999)
        return [self.vocab.get(c, 0) for c in chars]


# ── CodeTokenMapper (SQLite-backed pattern learning) ──

class CodeTokenMapper:
    """
    SQLite-backed token-pattern mapper.
    Stores token_id→category mappings and learned code patterns.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            _DEFAULT_CACHE_DIR, "token_map.db"
        )
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""CREATE TABLE IF NOT EXISTS token_map (
                token_id INTEGER PRIMARY KEY, text TEXT NOT NULL,
                category TEXT DEFAULT 'unknown', freq INTEGER DEFAULT 0,
                first_seen REAL DEFAULT (strftime('%s','now')))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS token_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT, pattern TEXT NOT NULL,
                token_ids TEXT NOT NULL, category TEXT DEFAULT 'code',
                freq INTEGER DEFAULT 1, UNIQUE(pattern))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_hash TEXT NOT NULL,
                pattern TEXT,
                language TEXT,
                freq INTEGER DEFAULT 1,
                last_seen REAL DEFAULT (strftime('%s','now')),
                UNIQUE(intent_hash))""")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[token_map] DB init: {e}")

    def add_pattern(self, pattern: str, token_ids: list):
        import json
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR IGNORE INTO token_patterns "
                "(pattern, token_ids, category) VALUES (?, ?, 'code')",
                (pattern[:200], json.dumps(token_ids))
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def record_usage(self, intent_hash: str, pattern: str = None, lang: str = None):
        """Track how often a pattern is used (for smart cache eviction)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO usage_stats (intent_hash, pattern, language, freq, last_seen) "
                "VALUES (?, ?, ?, 1, strftime('%s','now')) "
                "ON CONFLICT(intent_hash) DO UPDATE SET "
                "freq = freq + 1, last_seen = strftime('%s','now')",
                (intent_hash, pattern or '', lang or '')
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_hot_patterns(self, limit: int = 10) -> list:
        """Most frequently used patterns — for cache pre-warming."""
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT pattern, language, freq FROM usage_stats "
                "WHERE pattern != '' ORDER BY freq DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            return [{"pattern": r[0], "language": r[1], "freq": r[2]} for r in rows]
        except Exception:
            return []

    def stats(self) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            token_cnt = conn.execute("SELECT COUNT(*) FROM token_map").fetchone()[0]
            pat_cnt = conn.execute("SELECT COUNT(*) FROM token_patterns").fetchone()[0]
            usage_cnt = conn.execute("SELECT COUNT(*) FROM usage_stats").fetchone()[0]
            conn.close()
            return {
                "token_maps": token_cnt,
                "token_patterns": pat_cnt,
                "usage_tracked": usage_cnt,
            }
        except Exception:
            return {"error": "db unavailable"}


if __name__ == "__main__":
    t = StandaloneTokenizer()
    print(f"✅ StandaloneTokenizer: vocab={t.vocab_size}")
    for s in ["def hello():", "print(42)", "import os"]:
        ids = t.encode(s)
        back = t.decode(ids)
        ok = "✅" if back == s else "⚠️"
        print(f"  {ok} {s:20s} → {len(ids):2d}t → {ids[:6]}...")
