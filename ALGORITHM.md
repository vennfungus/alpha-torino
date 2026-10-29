# Algorithm Guide

This document explains how the project works internally: how text becomes training examples, what the model architecture is, how training works, and how text is generated. It is intended for readers who want to understand the algorithm rather than only run the commands. File and function names refer to the actual source code.

## Overview

The project implements a small **decoder-only Transformer language model** trained from scratch with self-supervision. The learning task is *next-token prediction*: given a sequence of tokens, predict the token that follows at each position. No labels are used — the targets are simply the same byte stream shifted by one position.

The full pipeline is:

1. **Corpus construction** — sorted `.txt` files are joined into one virtual UTF-8 byte stream (`src/dataset.py`, `MappedByteCorpus`).
2. **Tokenisation** — the byte stream is treated directly as token IDs in the range 0–255 (`encode_text`, byte-level tokeniser).
3. **Example formation** — fixed-length windows are cut from the stream; inputs and targets are the same window shifted by one token (`CausalLMDataset`, `MappedCausalLMDataset`).
4. **Modelling** — a decoder-only Transformer estimates a probability distribution over the next token (`src/model.py`, `CausalLanguageModel`).
5. **Training** — cross-entropy minimisation with AdamW, early stopping on validation loss (`src/train.py`).
6. **Generation** — autoregressive sampling: predict one token, append it, repeat (`src/inference.py`, `generate`).

## Tokenisation: UTF-8 bytes as tokens

The tokeniser is deliberately trivial: every byte of the UTF-8 encoding of the text is one token. This gives a fixed vocabulary of 256 IDs (`vocab_size: 256` in `configs/model_config.yaml`).

- **Encoding** (`encode_text`) is `text.encode("utf-8")` viewed as a list of integers.
- **Decoding** (`decode_tokens`) converts IDs back to bytes and decodes with `errors="replace"`, so incomplete or invalid byte sequences (which can occur when generation stops mid-character) do not crash the program.

Because UTF-8 is a variable-width encoding, a single character such as `é` or a Vietnamese diacritic becomes two byte tokens. The model therefore sees a finer-grained sequence than a word- or subword-level model would, and must learn to compose multi-byte characters — this costs sequence length but removes the need to train a tokeniser and handles any Unicode input.

## Corpus handling without loading it into RAM

`MappedByteCorpus` builds one *virtual* byte stream out of all `.txt` files found recursively beneath the data directory, in sorted order, separated by a blank line (`\n\n`):

- Each file is opened with `mmap` (`ACCESS_READ`), so the operating system pages data in on demand rather than the process copying whole files into memory.
- Leading and trailing whitespace is trimmed from each file (`_trim_bounds`), walking UTF-8 character boundaries carefully so a multi-byte character is never split.
- Each file's UTF-8 validity is verified up front with an incremental decoder (`_validate_utf8`) in 1 MiB chunks.
- The `_segments` list records, for every file, where it begins and ends in the *virtual* stream. `read(start, end)` uses `bisect_right` over the segment starts to translate a virtual range into per-file memory-map slices.

Training samples are therefore read straight from the memory maps as small byte ranges, which is what allows corpora larger than RAM to be used.

## Building training examples

A causal language model needs pairs *(input, target)* where the target at each position is the next token. Two dataset classes provide this:

- `MappedCausalLMDataset(corpus, start, end, block_size)` — the memory-efficient path used for real training. For index `i` it reads `block_size + 1` consecutive bytes from the virtual corpus and splits them into:
  - `input_ids  = token_ids[:-1]` (positions `i … i + block_size − 1`)
  - `target_ids = token_ids[1:]`  (positions `i + 1 … i + block_size`)

  so every position of the window yields one supervised prediction from a single read.
- `CausalLMDataset` — the same construction over an in-memory tensor, used by the tests and small experiments.

The number of examples is `length_of_stream − block_size`: one example per possible starting offset. The corpus is split positionally — the first 90 % of the byte stream trains and the last 10 % validates. Note this is a positional holdout, not a document-level split, so text of similar style can appear on both sides of the boundary.

### Shuffling with a bounded buffer

Because materialising a full permutation of millions of example indices would need a lot of memory, training shuffles indices with `BufferedShuffleSampler`: it keeps a reservoir-style buffer of `buffer_size` pending indices (100 000 by default), repeatedly emits one index chosen uniformly at random from the buffer and refills that slot with the next unused index. This gives good local randomness with `O(buffer_size)` memory instead of `O(dataset_size)`.

Validation uses `RandomSubsetSampler`, which draws a fixed number of unique indices via `random.sample` — again avoiding a full permutation — so each epoch validates on a random subset (10 000 examples by default) rather than the whole split.

## Model architecture

`CausalLanguageModel` is a compact decoder-only Transformer (the GPT-style design), parameterised by `configs/model_config.yaml`:

| Parameter    | Default | Meaning                                     |
|--------------|---------|---------------------------------------------|
| `vocab_size` | 256     | Number of distinct tokens (one per byte)    |
| `block_size` | 128     | Maximum context length the model can attend over |
| `d_model`    | 128     | Width of the hidden representations         |
| `n_heads`    | 4       | Attention heads per block                   |
| `n_layers`   | 4       | Number of Transformer blocks                |
| `dropout`    | 0.1     | Dropout probability inside each block       |

The forward pass (`forward`) works as follows:

1. **Embeddings.** Each token ID is mapped to a `d_model`-dimensional vector by a learned token embedding table. A learned *positional* embedding — a separate table of `block_size` vectors indexed by position 0…`block_size−1` — is added to it. Without positional information, attention would be order-blind and the model could not distinguish "dog bites man" from "man bites dog".
2. **Causal masking.** A boolean upper-triangular mask (`torch.triu(..., diagonal=1)`) forbids each position from attending to any later position. Attention to *future* tokens would leak the answer the model is supposed to predict. This mask is what makes the model *causal* (autoregressive): position `t` sees only positions `0…t`.
3. **Transformer blocks.** The masked sequence passes through `n_layers` identical blocks, built here from `nn.TransformerEncoderLayer` with `norm_first=True` (pre-norm). Each block contains:
   - **Multi-head self-attention.** The layer computes queries, keys and values from the hidden states and, for each position, produces a weighted average of value vectors, where the weights are softmax-normalised dot products between that position's query and all *permitted* (earlier) keys. With `n_heads = 4` and `d_model = 128`, each head operates on `d_model / n_heads = 32` dimensions, letting different heads specialise in different relationships (local syntax, repeated characters, and so on) in parallel.
   - **Feed-forward network.** A two-layer MLP that expands each position's vector to `4 × d_model = 512` dimensions, applies a non-linearity, and projects back to `d_model`. This is where much of the per-token transformation capacity lives.
   - **Residual connections, layer normalisation and dropout** around both sub-layers, which stabilise optimisation and regularise the small model.
4. **Output head.** After a final layer norm (`final_norm`), a linear layer without bias (`lm_head`) projects each position's hidden vector to `vocab_size = 256` logits — one unnormalised score per possible next byte.

The implementation reuses PyTorch's `nn.TransformerEncoder`, which keeps the code short; semantically it is a standard decoder-only stack with bidirectional attention disabled by the causal mask.

## Training objective

Given logits of shape `(batch, block_size, vocab_size)` and targets of the same token-length shape, the loss is **cross-entropy** averaged over every position of every sequence in the batch:

```
L = − (1 / N) · Σ  log p(target_t | tokens_0…t)
```

computed in one flattened call to `nn.functional.cross_entropy` (`logits.reshape(-1, vocab_size)` against `targets.reshape(-1)`). Minimising it is maximum-likelihood estimation: the model increases the probability of the bytes that actually occur next in the corpus and decreases the probability of everything else. Because every position of every window contributes a training signal, one `block_size`-token window provides `block_size` supervised predictions.

Training details (`src/train.py`, tuned via `configs/train_config.yaml`):

- **Early stopping** monitors validation loss: training stops after `early_stopping_patience` (default 3) epochs without an improvement of at least `early_stopping_min_delta`, and the weights from the *best* validation epoch are restored before the checkpoint is saved. This protects against the small model overfitting or memorising a limited corpus.
- On CUDA, training uses bfloat16 mixed precision when the GPU supports it, plus pinned-memory batches and parallel data-loading workers, to speed up stepping without changing the algorithm.

## Text generation

Generation (`generate` in `src/inference.py`) is **autoregressive sampling**:

1. The prompt is encoded to byte IDs and used as the initial context.
2. At each step, the model is run on the last `block_size` tokens of the context (a sliding window — older tokens drop out of reach), and only the logits at the **final** position are kept, since that position's distribution is the prediction for the next token.
3. The logits are divided by `temperature`. Temperature rescales the distribution before the softmax: values below 1 sharpen it (more deterministic, more repetitive output), values above 1 flatten it (more varied, potentially less coherent output). The command-line default is `0.8`.
4. The softmaxed distribution is sampled once with `torch.multinomial`, and the chosen token is appended to the context.
5. Steps 2–4 repeat for `max_new_tokens` steps, then the whole ID sequence — prompt plus generated bytes — is decoded back to text.

Because generation re-runs the full forward pass on the context window at every step (there is no KV cache in this implementation), generation cost grows with context length but stays bounded by `block_size`.

## Complexity summary

- **Parameters** are dominated by the embedding tables (`256 × d_model` and `block_size × d_model`) and the per-block matrices; with the defaults the model is on the order of hundreds of thousands of parameters — small enough to train on a CPU.
- **Attention** within a block is `O(block_size²)` in sequence length, which is why `block_size: 128` keeps training cheap.
- **Memory** for the data pipeline is `O(corpus_files + buffer_size)` rather than `O(corpus_bytes)`, thanks to memory mapping and the bounded shuffle buffer.

## Pointers into the code

| Concept                     | Where to look                                            |
|-----------------------------|----------------------------------------------------------|
| Virtual byte-stream corpus  | `src/dataset.py` — `MappedByteCorpus`                    |
| Byte tokeniser              | `src/dataset.py` — `encode_text`, `decode_tokens`        |
| Next-token examples         | `src/dataset.py` — `CausalLMDataset`, `MappedCausalLMDataset` |
| Bounded shuffle             | `src/dataset.py` — `BufferedShuffleSampler`              |
| Model and causal mask       | `src/model.py` — `CausalLanguageModel.forward`           |
| Loss and early stopping     | `src/train.py`                                           |
| Autoregressive sampling     | `src/inference.py` — `generate`                          |
