import pytest
import torch

from src.dataset import (
    BufferedShuffleSampler,
    CausalLMDataset,
    MappedByteCorpus,
    MappedCausalLMDataset,
    decode_tokens,
    encode_text,
    load_raw_corpus,
)
from src.inference import generate
from src.model import CausalLanguageModel


def test_model_returns_logits_and_loss() -> None:
    model = CausalLanguageModel(vocab_size=32, block_size=8, d_model=16, n_heads=4, n_layers=1)
    input_ids = torch.randint(0, 32, (2, 8))

    logits, loss = model(input_ids, input_ids)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert torch.isfinite(loss)


def test_model_rejects_context_longer_than_block_size() -> None:
    model = CausalLanguageModel(vocab_size=32, block_size=4, d_model=16, n_heads=4, n_layers=1)

    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.ones((1, 5), dtype=torch.long))


def test_dataset_pairs_each_token_with_its_next_token() -> None:
    dataset = CausalLMDataset([1, 2, 3, 4], block_size=2)

    input_ids, targets = dataset[0]

    assert input_ids.tolist() == [1, 2]
    assert targets.tolist() == [2, 3]


def test_mapped_corpus_preserves_trimmed_joined_text_and_samples(tmp_path) -> None:
    (tmp_path / "first.txt").write_text("\u2003Xin chào\u2003", encoding="utf-8")
    (tmp_path / "second.txt").write_text("  Thế giới  ", encoding="utf-8")
    expected = "Xin chào\n\nThế giới".encode("utf-8")

    with MappedByteCorpus(tmp_path) as corpus:
        assert corpus.byte_length == len(expected)
        assert corpus.read(0, corpus.byte_length) == expected

        dataset = MappedCausalLMDataset(
            corpus,
            start=0,
            end=corpus.byte_length,
            block_size=5,
        )
        input_ids, target_ids = dataset[0]

    assert bytes(input_ids.tolist()) == expected[:5]
    assert bytes(target_ids.tolist()) == expected[1:6]


def test_buffered_shuffle_sampler_visits_every_sample_once() -> None:
    dataset = CausalLMDataset(torch.arange(20), block_size=2)
    sample_indices = list(BufferedShuffleSampler(dataset, buffer_size=4))

    assert len(sample_indices) == len(dataset)
    assert sorted(sample_indices) == list(range(len(dataset)))


def test_utf8_byte_tokenizer_round_trips_vietnamese_text() -> None:
    text = "Xin chào thế giới"

    assert decode_tokens(encode_text(text)) == text


def test_raw_corpus_loads_text_files_and_joins_documents(tmp_path) -> None:
    (tmp_path / "first.txt").write_text("Xin chào", encoding="utf-8")
    (tmp_path / "second.txt").write_text("Thế giới", encoding="utf-8")

    corpus, files = load_raw_corpus(tmp_path)

    assert corpus == "Xin chào\n\nThế giới"
    assert len(files) == 2


def test_generate_appends_requested_number_of_tokens() -> None:
    model = CausalLanguageModel(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1)

    output = generate(model, torch.tensor([[1, 2]]), max_new_tokens=3)

    assert output.shape == (1, 5)
