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
