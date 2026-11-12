import pytest

from src.inference_qa import generate_answer
from src.model_qa import QALanguageModel


def test_generate_answer_supports_empty_generation() -> None:
    model = QALanguageModel(256, 96, d_model=8, n_heads=2, n_layers=1)

    assert generate_answer(model, "A context.", "A question?", max_new_tokens=0) == ""


def test_generate_answer_rejects_invalid_generation_settings() -> None:
    model = QALanguageModel(256, 96, d_model=8, n_heads=2, n_layers=1)

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_answer(model, "A context.", "A question?", max_new_tokens=-1)
