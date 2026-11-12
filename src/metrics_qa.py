"""SQuAD-style normalisation and exact-match/F1 metrics."""

import re
import string


def normalize_answer(answer: str) -> str:
    """Apply the standard SQuAD English answer normalisation."""
    answer = answer.lower()
    answer = "".join(character for character in answer if character not in string.punctuation)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    common = set(predicted_tokens) & set(reference_tokens)
    common_count = sum(
        min(predicted_tokens.count(token), reference_tokens.count(token))
        for token in common
    )
    if common_count == 0:
        return 0.0
    precision = common_count / len(predicted_tokens)
    recall = common_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
