import pytest

from scripts.create_tinystories_subset import create_subset


def test_create_subset_preserves_stories_and_source(tmp_path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "processed" / "subset.txt"
    source_text = (
        "Story one, first paragraph.\n"
        "Story one, second paragraph.\n"
        "<|endoftext|>\n"
        "Story two, complete text.\n"
        "<|endoftext|>\n"
    )
    source.write_text(source_text, encoding="utf-8")

    story_count, selected_count = create_subset(source, output, 1, seed=42)

    assert story_count == 2
    assert selected_count == 2
    assert source.read_text(encoding="utf-8") == source_text
    assert output.read_text(encoding="utf-8") == source_text


def test_create_subset_rejects_same_source_and_output(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Story.\n<|endoftext|>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        create_subset(source, source, 1, seed=42)
