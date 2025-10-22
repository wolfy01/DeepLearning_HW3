# Spoken SQuAD – BERT-Based Extractive Question Answering

## Overview
This project fine-tunes a **BERT-based question answering (QA)** model on the **Spoken SQuAD** dataset to analyze robustness under **ASR (Automatic Speech Recognition) errors** and increasing **noise** levels.

Pipeline:
- Load and align Spoken SQuAD datasets (clean and noisy WER sets)
- Tokenize question–context pairs with sliding windows
- Fine-tune **bert-base-uncased** for extractive QA
- Evaluate using **Exact Match (EM)** and **F1** scores

Reference dataset: [Chia-Hsuan-Lee/Spoken-SQuAD](https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD)

---

## Dataset Information

**Source:** SQuAD texts converted to speech (Google TTS) and transcribed with CMU Sphinx.

| Split            | Description        | Size    | Avg. WER |
|------------------|--------------------|---------|----------|
| Train            | Clean ASR          | 37,111  | 22.77%   |
| Test (WER22)     | Baseline ASR       | 5,351   | 22.73%   |
| WER44            | White noise (V1)   | 17,841  | 44%      |
| WER54            | White noise (V2)   | 17,841  | 54%      |

Each item includes **context** (ASR transcription), **question** (text), and **answer** (span indices).

---

## Model & Setup

- **Model:** `BertForQuestionAnswering` (Hugging Face, `bert-base-uncased`)
- **Tokenizer:** `BertTokenizerFast`
- **Frameworks:** PyTorch, Transformers, Accelerate

### Training Configuration

| Parameter                | Value        |
|--------------------------|--------------|
| Optimizer                | AdamW        |
| Learning Rate            | 2e-5         |
| Scheduler                | Linear warmup/decay |
| Epochs                   | 5            |
| Batch Size               | 8            |
| Gradient Accumulation    | 4 steps      |
| Max Sequence Length      | 384          |
| Stride                   | 128          |

**Loss:** Cross-entropy over start/end positions.  
**Progress:** Tracked with `tqdm` per epoch.

---

## Evaluation Metrics

- **Exact Match (EM):** exact span equality after normalization  
- **F1:** token-level overlap  
- **Normalization:** lowercasing, punctuation/articles removal, whitespace fix

### Results

| Dataset                | Exact Match (EM) | F1 Score |
|------------------------|------------------|----------|
| Standard (WER22)       | **51.83%**       | **64.57%** |
| WER44 (Moderate Noise) | 10.20%           | 19.41%   |
| WER54 (High Noise)     | 6.64%            | 15.42%   |

**Insight:** Clean ASR text yields strong performance; higher WER severely degrades span extraction accuracy—consistent with the Spoken SQuAD study.

---

## File Structure
├── HW3_SpokenSQuAD_BERT.ipynb # Main notebook
├── README.md # This file
├── spoken_train-v1.1.json # Train set
├── spoken_test-v1.1.json # Validation (WER22)
├── spoken_test-v1.1_WER44.json # Validation (WER44)
├── spoken_test-v1.1_WER54.json # Validation (WER54)
├── model/ # Saved fine-tuned model
└── tokenizer/ # Saved tokenizer

---

## How to Run

### 1) Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers accelerate fuzzywuzzy tqdm

2️⃣ Run the Notebook
Open and execute all cells in: HW3_SpokenSQuAD_BERT.ipynb

3️⃣ Evaluate Performance
evaluate(model, val_dataset)          # WER22 (clean ASR)
evaluate(model, val_dataset_wer44)    # WER44 (noisy ASR)
evaluate(model, val_dataset_wer54)    # WER54 (noisy ASR)

