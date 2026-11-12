"""Causal language model architecture used by the supervised QA pipeline."""

from .model_gen import CausalLanguageModel as QALanguageModel

__all__ = ["QALanguageModel"]
