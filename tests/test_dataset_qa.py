import json

import pytest

from src.dataset_gen import encode_text
from src.dataset_qa import QADataset, collate_qa_batch, load_qa_jsonl
from src.format_qa import ANSWER_TERMINATOR, build_prompt


def record(
    context: str = "Brazil is in South America.",
    question: str = "Where is Brazil?",
    answer: str = "South America",
) -> dict[str, str]:
    return {"context": context, "question": question, "answer": answer}


def test_qa_dataset_masks_prompt_and_trains_answer_and_terminator() -> None:
    sample = record()
    dataset = QADataset([sample], block_size=128, max_answer_bytes=32)

    input_ids, targets = dataset[0]
    prompt = build_prompt(sample["context"], sample["question"], 128 - len("South America\n\n"))
    prompt_length = len(encode_text(prompt))
    first_answer_target = prompt_length - 1

    assert len(input_ids) <= 128
    assert targets[:first_answer_target].tolist() == [-100] * first_answer_target
    assert targets[first_answer_target].item() == ord("S")
    assert targets[-2:].tolist() == list(ANSWER_TERMINATOR)


def test_qa_dataset_preserves_answer_when_context_must_be_trimmed() -> None:
    sample = record(context=("irrelevant " * 80) + "Target answer here" + (" tail" * 80),
                    question="What text is targeted?", answer="Target answer")
    dataset = QADataset([sample], block_size=90, max_answer_bytes=24)

    input_ids, _ = dataset[0]
    prompt_text = bytes(input_ids.tolist()).decode("utf-8", errors="ignore")

    assert len(input_ids) <= 90
    assert "Target answer" in prompt_text
    assert "Question: What text is targeted?" in prompt_text


def test_collate_qa_batch_pads_inputs_and_masks_padding_targets() -> None:
    dataset = QADataset([record(), record(context="Short.", question="Where?", answer="Here.")],
                        block_size=128, max_answer_bytes=32)
    batch = collate_qa_batch([dataset[0], dataset[1]])

    assert batch[0].ndim == 2
    assert batch[0].shape == batch[1].shape
    for row, (_, targets) in enumerate([dataset[0], dataset[1]]):
        assert batch[1][row, len(targets) :].tolist() == [-100] * (
            batch[1].shape[1] - len(targets)
        )


def test_load_qa_jsonl_validates_records_and_utf8(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    expected = record(answer="Đà Nẵng")
    path.write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")

    assert load_qa_jsonl(path) == [expected]

    path.write_text('{"context": "c", "question": "q", "answer": ""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string"):
        load_qa_jsonl(path)


def test_load_qa_jsonl_rejects_malformed_json_with_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n{bad}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_qa_jsonl(path)


def test_qa_dataset_rejects_question_that_cannot_fit() -> None:
    dataset = QADataset([record(question="A very long question?")], block_size=16)

    with pytest.raises(ValueError, match="too long"):
        dataset[0]
