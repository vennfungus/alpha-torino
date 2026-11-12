"""Prompt and sequence formatting for byte-level question answering."""

from collections.abc import Mapping

from .tokenizer import encode_text

ANSWER_TERMINATOR = b"\n\n"
_CONTEXT_PREFIX = b"Context: "
_QUESTION_PREFIX = b"\nQuestion: "
_ANSWER_PREFIX = b"\nAnswer: "


def _safe_utf8_prefix(value: bytes, max_bytes: int) -> bytes:
    return value[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")


def _fit_context(context: str, max_bytes: int, answer: bytes = b"") -> bytes:
    encoded = context.encode("utf-8")
    if len(encoded) <= max_bytes:
        return encoded
    if max_bytes <= 0:
        return b""

    answer_start = encoded.find(answer) if answer else -1
    if answer_start < 0:
        start = 0
    else:
        spare = max(0, max_bytes - len(answer))
        start = max(0, answer_start - spare // 2)
        start = min(start, max(0, len(encoded) - max_bytes))
    return _safe_utf8_prefix(encoded[start : start + max_bytes], max_bytes)


def build_prompt(
    context: str,
    question: str,
    max_prompt_bytes: int,
    preserve_answer: bytes = b"",
) -> str:
    """Format a prompt, trimming context to fit while keeping the question intact."""
    if max_prompt_bytes < 1:
        raise ValueError("max_prompt_bytes must be positive")
    question_bytes = question.encode("utf-8")
    fixed_prompt = _CONTEXT_PREFIX + _QUESTION_PREFIX + question_bytes + _ANSWER_PREFIX
    context_budget = max_prompt_bytes - len(fixed_prompt)
    if context_budget < 0:
        raise ValueError("Question is too long to fit within the model block_size")

    context_bytes = _fit_context(context, context_budget, preserve_answer)
    prompt = _CONTEXT_PREFIX + context_bytes + _QUESTION_PREFIX + question_bytes + _ANSWER_PREFIX
    return prompt.decode("utf-8")


def build_training_sequence(
    record: Mapping[str, str],
    block_size: int,
    max_answer_bytes: int,
) -> tuple[list[int], int]:
    """Return UTF-8 byte IDs for prompt + answer + terminator and prompt length."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if max_answer_bytes < 1:
        raise ValueError("max_answer_bytes must be positive")

    context = record["context"]
    question = record["question"]
    answer = record["answer"]
    if not all(isinstance(value, str) and value.strip() for value in (context, question, answer)):
        raise ValueError("QA examples must have non-empty context, question, and answer strings")

    answer_bytes = answer.encode("utf-8")
    fixed_prompt = _CONTEXT_PREFIX + _QUESTION_PREFIX + question.encode("utf-8") + _ANSWER_PREFIX
    answer_budget = block_size + 1 - len(fixed_prompt) - len(ANSWER_TERMINATOR)
    if answer_budget < 1:
        raise ValueError("Question is too long to fit within block_size")
    answer_bytes = _safe_utf8_prefix(answer_bytes, min(max_answer_bytes, answer_budget))
    if not answer_bytes:
        raise ValueError("Question and answer do not fit within block_size")

    prompt_budget = block_size + 1 - len(answer_bytes) - len(ANSWER_TERMINATOR)
    prompt = build_prompt(context, question, prompt_budget, answer_bytes)
    prompt_bytes = prompt.encode("utf-8")
    sequence = prompt_bytes + answer_bytes + ANSWER_TERMINATOR
    if len(sequence) > block_size + 1:
        raise RuntimeError("Formatted QA sequence exceeds block_size")
    return encode_text(sequence.decode("utf-8")), len(prompt_bytes)
