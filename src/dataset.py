"""Dataset utilities for next-token language modelling."""

import codecs
import mmap
import random
from bisect import bisect_right
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import torch
from torch.utils.data import Dataset, Sampler


class MappedByteCorpus:
    """Expose sorted text files as one memory-mapped UTF-8 byte stream."""

    def __init__(self, raw_dir: str | Path) -> None:
        raw_path = Path(raw_dir)
        files = sorted(raw_path.rglob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {raw_path}")

        self.files = files
        self._handles = []
        self._maps: list[mmap.mmap] = []
        self._segments: list[tuple[int, int, mmap.mmap | bytes, int]] = []
        self._segment_starts: list[int] = []
        self.byte_length = 0

        try:
            for file in files:
                handle = file.open("rb")
                size = file.stat().st_size
                if size == 0:
                    handle.close()
                    continue

                mapped_file = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                self._handles.append(handle)
                self._maps.append(mapped_file)
                self._validate_utf8(mapped_file)
                start, end = self._trim_bounds(mapped_file)
                if start == end:
                    continue

                if self.byte_length:
                    self._add_segment(b"\n\n", 0, 2)
                self._add_segment(mapped_file, start, end - start)
        except Exception:
            self.close()
            raise

        if self.byte_length == 0:
            self.close()
            raise ValueError(f"All .txt files in {raw_path} are empty")

    @staticmethod
    def _validate_utf8(mapped_file: mmap.mmap) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")()
        chunk_size = 1024 * 1024
        for offset in range(0, len(mapped_file), chunk_size):
            decoder.decode(mapped_file[offset : offset + chunk_size])
        decoder.decode(b"", final=True)

    @staticmethod
    def _trim_bounds(mapped_file: mmap.mmap) -> tuple[int, int]:
        start = 0
        end = len(mapped_file)
        while start < end:
            first_byte = mapped_file[start]
            if first_byte < 0x80:
                char_length = 1
            elif 0xC2 <= first_byte <= 0xDF:
                char_length = 2
            elif 0xE0 <= first_byte <= 0xEF:
                char_length = 3
            elif 0xF0 <= first_byte <= 0xF4:
                char_length = 4
            else:
                char_length = 1
            character = bytes(mapped_file[start : start + char_length]).decode("utf-8")
            if not character.isspace():
                break
            start += char_length

        while start < end:
            char_start = end - 1
            while char_start > start and mapped_file[char_start] & 0xC0 == 0x80:
                char_start -= 1
            character = bytes(mapped_file[char_start:end]).decode("utf-8")
            if not character.isspace():
                break
            end = char_start
        return start, end

    def _add_segment(self, source: mmap.mmap | bytes, offset: int, length: int) -> None:
        start = self.byte_length
        self._segment_starts.append(start)
        self._segments.append((start, start + length, source, offset))
        self.byte_length += length

    def read(self, start: int, end: int) -> bytes:
        """Read a byte range from the virtual corpus without loading it all."""
        if start < 0 or end < start or end > self.byte_length:
            raise IndexError("byte range is outside the corpus")

        result = bytearray()
        segment_index = max(0, bisect_right(self._segment_starts, start) - 1)
        position = start
        while position < end:
            segment_start, segment_end, source, source_offset = self._segments[segment_index]
            chunk_end = min(end, segment_end)
            source_start = source_offset + position - segment_start
            source_end = source_start + chunk_end - position
            result.extend(source[source_start:source_end])
            position = chunk_end
            segment_index += 1
        return bytes(result)

    def close(self) -> None:
        for mapped_file in self._maps:
            mapped_file.close()
        for handle in self._handles:
            handle.close()
        self._maps.clear()
        self._handles.clear()

    def __enter__(self) -> "MappedByteCorpus":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def encode_text(text: str) -> list[int]:
    """Encode text as UTF-8 byte IDs in the range 0-255."""
    return list(text.encode("utf-8"))


def decode_tokens(token_ids: list[int] | torch.Tensor) -> str:
    """Decode UTF-8 byte IDs, replacing incomplete or invalid byte sequences."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return bytes(token_ids).decode("utf-8", errors="replace")


def load_raw_corpus(raw_dir: str | Path) -> tuple[str, list[Path]]:
    """Read and join all UTF-8 .txt files beneath a raw-data directory."""
    raw_path = Path(raw_dir)
    files = sorted(raw_path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {raw_path}")

    documents = [file.read_text(encoding="utf-8").strip() for file in files]
    documents = [document for document in documents if document]
    if not documents:
        raise ValueError(f"All .txt files in {raw_path} are empty")
    return "\n\n".join(documents), files


class CausalLMDataset(Dataset):
    """Create fixed-length, next-token prediction examples from token IDs."""

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must contain more tokens than block_size")
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.token_ids[index : index + self.block_size]
        target_ids = self.token_ids[index + 1 : index + self.block_size + 1]
        return input_ids, target_ids


class MappedCausalLMDataset(Dataset):
    """Create next-token examples from a range of a memory-mapped corpus."""

    def __init__(
        self,
        corpus: MappedByteCorpus,
        start: int,
        end: int,
        block_size: int,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if start < 0 or end > corpus.byte_length or end - start <= block_size:
            raise ValueError("corpus range must contain more tokens than block_size")
        self.corpus = corpus
        self.start = start
        self.end = end
        self.block_size = block_size

    def __len__(self) -> int:
        return self.end - self.start - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        sample = self.corpus.read(
            self.start + index,
            self.start + index + self.block_size + 1,
        )
        token_ids = torch.frombuffer(bytearray(sample), dtype=torch.uint8).to(torch.long)
        return token_ids[:-1], token_ids[1:]


class BufferedShuffleSampler(Sampler[int]):
    """Shuffle a full dataset pass using a fixed-size index buffer."""

    def __init__(self, data_source: Dataset, buffer_size: int = 100_000) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        self.num_samples = len(data_source)
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[int]:
        buffer = list(range(min(self.buffer_size, self.num_samples)))
        next_index = len(buffer)
        while next_index < self.num_samples:
            slot = random.randrange(len(buffer))
            yield buffer[slot]
            buffer[slot] = next_index
            next_index += 1
        random.shuffle(buffer)
        yield from buffer

    def __len__(self) -> int:
        return self.num_samples
