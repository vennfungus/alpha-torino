# Mini Causal Language Model

A small autoregressive language model project built with PyTorch. The model is trained from scratch on the supplied data; it does not fine-tune a large pre-trained model.

## Project structure

- `data/raw/`: original text data.
- `data/processed/`: cleaned or tokenised data.
- `data/external/`: data from external sources.
- `notebooks/`: notebooks for data exploration and model experiments.
- `src/`: dataset, model, training, evaluation and inference code.
- `configs/`: model and training configuration.
- `checkpoints/`: saved model weights; contents are excluded from Git.
- `tests/`: automated tests.

## Installation

Python 3.10 or later is required. Create a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Preparing data

Place one or more UTF-8 encoded `.txt` files in `data/raw/`. You can organise them into subdirectories. For example:

```text
data/raw/
├── emails.txt
└── articles/
    └── notes.txt
```

Use text representative of the style and subject matter you want the model to learn, such as your emails, notes or articles. The trained model will reflect patterns in its input data, including its wording, punctuation and formatting. It learns statistical patterns from the examples; it does not become the author or know things that are not present in the data.

No labels or pre-tokenised files are needed. This is self-supervised training: the program reads the `.txt` files, joins documents with a blank line, encodes the text as UTF-8 byte IDs, and creates next-token prediction targets automatically.

The byte-level tokenizer has a fixed vocabulary of 256 values, so no separate tokenizer training is needed. It is simple and supports all UTF-8 characters, but is less efficient than a subword tokenizer. The corpus must contain more bytes than `block_size`. With a small dataset, the model may memorise passages and produce limited or repetitive text.

### Building a useful dataset

There is no fixed file count that guarantees good results. As a practical starting point, aim for around **100,000 words** of carefully selected text; **200,000–500,000 words** gives this small model more examples to learn from. These are experimentation targets, not quality guarantees. A few dozen files can be enough to test the pipeline, but a larger, varied collection is more useful for assessing the writing style.

For example, you could collect **100–300 documents averaging 500–1,500 words each**. Documents do not need to be the same length, and a document can be a complete email, article, set of notes or other coherent piece. Prefer complete examples over fragments without context.

Choose roughly **5–8 recurring subject areas** if you want the model to write across several areas. Give the most space to the kinds of writing you expect to generate: as a starting balance, use about **60–70%** for the main use case, **20–30%** for other common topics, and the remainder for related material. If you only need one kind of writing, concentrate on that rather than adding unrelated topics. Consistent examples are more useful than broad but irrelevant coverage.

Before adding files to `data/raw/`:

1. Gather text that you wrote, or that you have permission to use, and that reflects the output you want.
2. Remove exact duplicates, boilerplate that appears in every document, corrupted text and material unrelated to the intended use.
3. Keep natural wording, punctuation, paragraphs and formatting; these are all patterns the model can learn.
4. Save as plain-text `.txt` files encoded as UTF-8. Use subfolders to organise documents if helpful; the training script searches them recursively.
5. Exclude passwords, private correspondence, personal details and other sensitive information you do not want the model to reproduce. Keep the raw dataset private and backed up appropriately.

No manual labels or train/validation folders are required. The current training script reads all `.txt` files, sorts and joins their contents with blank lines, then uses the first 90% of the resulting byte stream for training and the last 10% for validation when both portions are longer than `block_size`. This is a simple holdout by corpus position, not a random or document-level split; closely related passages may therefore appear in both portions. The script does not currently create a processed dataset or deduplicate files for you, so curate the raw files before training.

## Training

After adding data to `data/raw/`, run this command from the project root:

```powershell
python -m src.train
```

The command reads `configs/model_config.yaml` and `configs/train_config.yaml`, creates a validation split when the corpus is large enough, and saves the weights to `checkpoints/model.pt`. You can reduce `block_size`, `d_model`, `n_layers` or `batch_size` in the YAML files to experiment on a machine with limited resources. By default, `device: auto` selects CUDA when available and otherwise uses the CPU.

## Generating text

```powershell
python -m src.inference --prompt "Dear colleague," --max-new-tokens 200
```

Use `--checkpoint` to select another checkpoint and `--temperature` to adjust the randomness of generation.

## Testing

```powershell
python -m pytest
```
