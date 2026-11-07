# Self-Consistency Prompting on LogiQA

Code and data for the paper "Self-Consistency Prompting: Analyzing Internal Agreement in Large Language Model Responses on English Logical Reasoning Tasks".

## Contents

| File | What it is |
|---|---|
| `prethesis.py` | Runs the experiment: queries a model through OpenRouter, five generations per LogiQA test question, and saves the results. |
| `Result/` | Saved generations for Llama 3.3 70B, DeepSeek V4 Flash, and Qwen3-Next 80B at temperatures 0.0, 0.3, 0.7, and 1.0 (`results_<model>_v3_T<temp>_N5.csv`, or the batch folders `output_batches_...`). |
| `category_labels.xlsx` | Reasoning-category labels for all 651 questions (sheet `final_dataset`), the adjudication log, and the inter-annotator agreement summary. Annotators are anonymized as A, B, and C. |
| `answer_key_review_verdicts.csv` | Verdicts for the 32 questions on which all 15 baseline generations agreed on an answer that differs from the official key. |
| `analysis_statistics.py` | Reproduces every number reported in the paper. Output is printed and saved to `analysis_output.txt`. |
| `make_figures.py` | Reproduces the four result figures in `figures/`. |

## Reproducing the analysis

```
pip install -r requirements.txt
python analysis_statistics.py
python make_figures.py
```

Both scripts search the current folder and its subfolders for the result files, so they can be run from the repository root. No API key is needed.

Scoring rule: the answer selected by majority vote is the plurality answer. A tie for first place is scored incorrect. A question with no parsed generation receives agreement 0 and is scored incorrect, so every model is evaluated on all 651 questions.

## Re-running the experiment

Running `prethesis.py` requires an OpenRouter API key and incurs API costs. See `python prethesis.py -h` for all options.

The LogiQA test split is downloaded from https://github.com/lgw863/LogiQA-dataset.
