# Mini Causal Language Model and QA

This repository contains two small, separate training pipelines built around the same byte-level causal Transformer:

- **Text generation (`*_gen`)** trains on plain text using next-byte prediction.
- **Question answering (`*_qa`)** trains on context/question/answer records and applies loss only to answer bytes.

Both models are trained from scratch. The QA pipeline is useful for learning and experimenting with supervised QA; this small byte-level model is not a substitute for fine-tuning a capable pretrained model when answer quality matters.

## Project structure

- `data/raw/`: source text and QA JSONL files.
- `configs/`: separate `*_gen.yaml` and `*_qa.yaml` model/training settings.
- `src/`: separate generation and QA datasets, trainers, evaluators and inference entry points.
- `scripts/`: dataset download and personal QA split utilities.
- `checkpoints/`: saved model weights.
- `tests/`: automated tests.
- `notebooks/`: generation-model experiments.

## Installation

Python 3.10 or later is required:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt
```

## Text generation

Place UTF-8 `.txt` files in `data/raw/`. The generation pipeline joins them, uses each UTF-8 byte as a token, and trains next-byte prediction. No manual tokenisation is needed.

```powershell
python -m src.train_gen
python -m src.inference_gen --prompt "Dear colleague," --max-new-tokens 200
```

The model and training settings are in `configs/model_config_gen.yaml` and `configs/train_config_gen.yaml`. The checkpoint defaults to `checkpoints/model_gen.pt`.

To create a sample plain-text corpus, run `python scripts/download_tinystories.py`.

## Question answering

QA records use UTF-8 JSONL, one object per line:

```json
{"context":"The capital of France is Paris.","question":"What is the capital of France?","answer":"Paris"}
```

Download SQuAD into document-level train and validation files:

```powershell
python scripts/download_qa.py --max-examples 50000
```

The files are written to `data/raw/train.jsonl` and `data/raw/validation.jsonl`. For your own single JSONL file, split by context/document before training:

```powershell
python scripts/split_qa_dataset.py --input data/raw/my_qa.jsonl --output-dir data/raw/qa_personal
```

Then train using those files:

```powershell
python -m src.train_qa --train-file data/raw/qa_personal/train.jsonl --validation-file data/raw/qa_personal/validation.jsonl
```

For downloaded SQuAD data, the default paths are already configured:

```powershell
python -m src.train_qa
python -m src.evaluate_qa
python -m src.inference_qa --context "The capital of France is Paris." --question "What is the capital of France?"
```

QA model and training settings live in `configs/model_config_qa.yaml` and `configs/train_config_qa.yaml`; the checkpoint defaults to `checkpoints/model_qa.pt`. The QA dataset formats a `Context / Question / Answer` prompt, truncates long contexts to fit the configured byte-level context window, and masks the prompt and padding so that cross-entropy is calculated only on the answer and its end delimiter. Keep the question short enough to fit the model's `block_size`.

The QA downloader uses SQuAD, an English dataset. For Vietnamese QA, prepare Vietnamese examples in the same JSONL shape. Since the model is trained from scratch with a small byte vocabulary, results are intended for experimentation rather than production-quality QA.

## Tests

```powershell
python -m pytest
```
