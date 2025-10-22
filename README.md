🧠 Spoken SQuAD – BERT-Based Extractive Question Answering
📘 Overview

This project fine-tunes a BERT-based question answering (QA) model on the Spoken SQuAD dataset to evaluate its performance on noisy ASR-transcribed data.
The main objective is to analyze how automatic speech recognition (ASR) errors and background noise affect the model’s comprehension and extraction capabilities.

The pipeline includes:

Preparing and aligning the Spoken SQuAD dataset.

Tokenizing context–question pairs using sliding windows.

Fine-tuning BERT (bert-base-uncased) on clean and noisy transcriptions.

Evaluating robustness across increasing WER (Word Error Rate) levels.

📂 Dataset Information

Dataset: Spoken SQuAD

Source: Text from SQuAD converted to speech using Google TTS and transcribed via CMU Sphinx.

Split	Description	Size	Avg. WER
Train	Clean transcriptions	37,111	22.77%
Test (WER22)	Baseline ASR	5,351	22.73%
WER44	White noise V1	17,841	44%
WER54	White noise V2	17,841	54%

Each entry contains:

context: ASR transcription of a spoken passage

question: text-based query

answer: answer span within the context

⚙️ Model Architecture

Base Model: BertForQuestionAnswering (bert-base-uncased)

Tokenizer: BertTokenizerFast

Framework: PyTorch + Hugging Face Transformers

Device Handling: Managed via Accelerator (supports GPU/mixed precision)

🧩 Training Configuration
Parameter	Value
Optimizer	AdamW
Learning Rate	2e-5
Scheduler	Linear decay with warmup
Epochs	5
Batch Size	8
Gradient Accumulation	4 steps
Max Sequence Length	384
Stride	128

Loss Function: Cross-entropy between predicted and true start–end indices.
Progress Tracking: Implemented via tqdm with epoch-level average loss reporting.

🧮 Evaluation Metrics

Exact Match (EM): Percentage of predictions exactly matching the ground-truth span.

F1 Score: Token-level overlap between prediction and ground truth.

Normalization: Removes punctuation, lowercases, and ignores articles.

📊 Results
Dataset	Exact Match (EM)	F1 Score
Standard (WER22)	51.94%	64.89%
WER44	10.11%	19.89%
WER54	6.75%	15.43%

Insight:
Performance on clean ASR text is strong, while noisy transcriptions significantly reduce comprehension accuracy—consistent with findings from the original Spoken SQuAD paper.

🧱 File Structure
├── HW3_SpokenSQuAD_BERT.ipynb     # Main notebook
├── README.md                      # Project overview and documentation
├── spoken_train-v1.1.json         # Training dataset
├── spoken_test-v1.1.json          # Validation (WER22)
├── spoken_test-v1.1_WER44.json    # Validation (WER44)
├── spoken_test-v1.1_WER54.json    # Validation (WER54)
├── model/                         # Saved fine-tuned model
└── tokenizer/                     # Saved tokenizer

🚀 How to Run
pip install torch torchvision torchaudio
pip install transformers accelerate fuzzywuzzy tqdm

2️⃣ Run the Notebook
Open and execute all cells in: HW3_SpokenSQuAD_BERT.ipynb

3️⃣ Evaluate Performance

The notebook automatically evaluates:
evaluate(model, val_dataset)          # Clean ASR text
evaluate(model, val_dataset_wer44)    # Noisy ASR (WER44)
evaluate(model, val_dataset_wer54)    # Noisy ASR (WER54)

