"""
yuanmoxiao-cerebellum — Zero-model code understanding engine

Core exports:
  CerebellumCognitiveBody  — Full pipeline (understand + generate + learn)
  StandaloneTokenizer      — Pure Python BPE tokenizer (no external deps)
  DeepSeekAdapter          — Optional: talk to DeepSeek in token space
"""

from .tokenizer import StandaloneTokenizer, CodeTokenMapper

try:
    from .engine import (
        CerebellumCognitiveBody,
        IntentParser,
        SmartCache,
        DeepSeekAdapter,
    )
except ImportError:
    # engine has optional deps; expose what's available
    CerebellumCognitiveBody = None
    IntentParser = None
    DeepSeekAdapter = None

__all__ = [
    "CerebellumCognitiveBody",
    "StandaloneTokenizer",
    "CodeTokenMapper",
    "IntentParser",
    "SmartCache",
    "DeepSeekAdapter",
]

__version__ = "0.1.0"
