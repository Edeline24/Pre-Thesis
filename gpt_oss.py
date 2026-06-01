from dotenv import load_dotenv
load_dotenv()


import os
import json
import time
import re
import urllib.request
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import requests
import random
from math import log2
import numpy as np

random.seed(42)
np.random.seed(42)

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME          = "openai/gpt-oss-120b:free"  
N_GENERATIONS       = 5
START_IDX           = 500 # 0 V, 100 V, 200 V, 300 V , 400 V, 500 V, 600
END_IDX             = 600 # 100 V, 200 V, 300 V, 400 V, 500 V, 600 V,  None
BATCH_SIZE          = 40   # Number of questions per batch file
TEMPERATURE         = 0.7
TOP_P               = 0.9
MAX_TOKENS          = 350  
OUTPUT_CSV          = "results_final.csv"
OUTPUT_DIR          = "output_charts"
BATCH_DIR           = "output_batches"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

# --- Core Functions ---
def load_logiqa(start_idx=0, end_idx=None):
    url = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt"
    print("Downloading LogiQA dataset from GitHub...")

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            raw = response.read().decode("utf-8")
    except Exception as e:
        print(f"Download error: {e}")
        return []

    lines = raw.split("\n")
    questions = []
    experimental_groups = ["group_1", "group_2", "group_3"]
    i = 0

    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        if i + 6 >= len(lines):
            break
        try:
            ans_letter = lines[i].strip().lower()
            if ans_letter not in ["a", "b", "c", "d"]:
                i += 1
                continue

            correct_idx = ord(ans_letter) - ord("a")
            context     = lines[i+1].strip()
            question    = lines[i+2].strip()
            opt_a = re.sub(r"^A\.\s*", "", lines[i+3].strip())
            opt_b = re.sub(r"^B\.\s*", "", lines[i+4].strip())
            opt_c = re.sub(r"^C\.\s*", "", lines[i+5].strip())
            opt_d = re.sub(r"^D\.\s*", "", lines[i+6].strip())

            option_str   = f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
            options_list = [opt_a, opt_b, opt_c, opt_d]

            if any(len(opt.strip()) < 2 for opt in options_list):
                i += 8
                continue

            qid = len(questions)
            questions.append({
                "id"                 : qid,
                "context"            : context,
                "question"           : question,
                "options"            : option_str,
                "correct_answer"     : chr(65 + correct_idx),
                "experimental_group" : experimental_groups[len(questions) % 3],
            })
            i += 8
        except (ValueError, IndexError):
            i += 1
            continue

    print(f"Loaded {len(questions)} total dataset questions.\n")
    for idx, q in enumerate(questions):
        q["id"] = idx
    questions = questions[start_idx:end_idx]
    return questions


PROMPT_TEMPLATE = """Read the question carefully. Explain your reasoning step by step and then choose the correct answer from the options. Please provide the reasoning before the final answer.

Context:

{context}

Question:

{question}

Options:

{options}

After your reasoning, write your final answer exactly like this:

Final Answer: A

(use only the letter A, B, C, or D)"""


def calculate_entropy(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0
    counter = Counter(valid)
    total   = len(valid)
    entropy = 0
    for count in counter.values():
        p = count / total
        entropy -= p * log2(p)
    return round(entropy, 3)


def build_prompt(q):
    return PROMPT_TEMPLATE.format(
        context  = q["context"],
        question = q["question"],
        options  = q["options"],
    )


# --- OpenRouter API call ---
def call_gpt(prompt):
    headers = {
        "Authorization" : f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type"  : "application/json",
        # Optional: helps OpenRouter identify your app in the dashboard
        #"HTTP-Referer"  : "https://github.com/your-project",
        "X-Title"       : "LogiQA Self-Consistency Experiment",
    }
    payload = {
        "model"       : MODEL_NAME,
        "messages"    : [{"role": "user", "content": prompt}],
        "temperature" : TEMPERATURE,
        "top_p"       : TOP_P,
        "max_tokens"  : MAX_TOKENS,
    }

    for attempt in range(3):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers = headers,
                json    = payload,
                timeout = 60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            print(f"  HTTP error {status} (attempt {attempt+1}): {e}")

            # Rate limit — parse Retry-After header if present
            if status == 429:
                retry_after = int(e.response.headers.get("Retry-After", 30))
                print(f"  Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after + 2)
                continue

            time.sleep(5)

        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(5)

    return None


def extract_answer(text):
    if not text:
        return None
    match = re.search(r"Final\s*Answer\s*:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        match = re.search(r"\b([A-D])\b", line)
        if match:
            return match.group(1).upper()
    return None


def calculate_agreement(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0, None
    counter             = Counter(valid)
    most_common, count  = counter.most_common(1)[0]
    agreement           = count / len(valid)
    return round(agreement, 3), most_common


# --- Batch Pipeline logic ---
def process_batch(batch_questions, batch_id, batch_file_path):
    """Runs experiment for a single block of questions."""
    batch_results = []

    for q in tqdm(batch_questions, desc=f"Processing Batch {batch_id}"):
        prompt  = build_prompt(q)
        answers = []
        texts   = []

        for _ in range(N_GENERATIONS):
            text   = call_gpt(prompt)        
            answer = extract_answer(text)
            answers.append(answer)
            texts.append(text or "")
            time.sleep(random.uniform(0.5, 1.2))

        if all(a is None for a in answers):
            continue

        agreement, majority_answer = calculate_agreement(answers)
        is_correct = (majority_answer == q["correct_answer"]) if majority_answer else False
        entropy    = calculate_entropy(answers)

        batch_results.append({
            "question_id"        : q["id"],
            "experimental_group" : q["experimental_group"],
            "correct_answer"     : q["correct_answer"],
            "all_answers"        : json.dumps(answers),
            "majority_answer"    : majority_answer,
            "agreement_ratio"    : agreement,
            "is_correct"         : is_correct,
            "high_agreement"     : agreement >= 0.8,
            "reasoning_texts"    : json.dumps(texts),
            "prompt"             : prompt,
            "entropy"            : entropy,
            "batch_id"           : batch_id,
        })

    if batch_results:
        df_batch = pd.DataFrame(batch_results)
        df_batch.to_csv(batch_file_path, index=False)
        print(f"-> Batch {batch_id} saved successfully ({len(df_batch)} rows).")


def merge_all_batches():
    """Finds all batch CSVs, merges them cleanly, and handles sorting."""
    all_files = glob.glob(os.path.join(BATCH_DIR, "batch_*.csv"))
    if not all_files:
        print("No batch files found to merge!")
        return pd.DataFrame()

    print(f"Found {len(all_files)} batch files. Merging...")
    df_list     = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    if "question_id" in combined_df.columns:
        combined_df = combined_df.sort_values(by="question_id").reset_index(drop=True)

    return combined_df


# --- Analytics and Reporting ---
def analyze_results(df):
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy    : {df['is_correct'].mean():.3f}")
    print(f"Mean Agreement Ratio: {df['agreement_ratio'].mean():.3f}")
    print(f"Total Questions     : {len(df)}")

    print("\n--- Accuracy per Experimental Group ---")
    type_stats = df.groupby("experimental_group").agg(
        accuracy       = ("is_correct", "mean"),
        mean_agreement = ("agreement_ratio", "mean"),
        count          = ("question_id", "count")
    ).round(3)
    print(type_stats)

    high_agree = df[df["high_agreement"]]
    low_agree  = df[~df["high_agreement"]]

    if len(high_agree) > 0:
        print(f"\n--- High Agreement (>=0.8): {len(high_agree)} questions --- Accuracy: {high_agree['is_correct'].mean():.3f}")
    if len(low_agree) > 0:
        print(f"--- Low Agreement (<0.8): {len(low_agree)} questions --- Accuracy: {low_agree['is_correct'].mean():.3f}")

    return type_stats


def plot_results(df, type_stats):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLM Self-Consistency Analysis — LogiQA (GPT-4o-mini via OpenRouter)",
                 fontsize=14, fontweight="bold")

    # Graph 1
    ax = axes[0, 0]
    colors = df["is_correct"].map({True: "steelblue", False: "tomato"}).fillna("gray")
    ax.scatter(df["agreement_ratio"], df["is_correct"].astype(int),
               c=colors, alpha=0.6, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Agreement Ratio")
    ax.set_ylabel("Correct (1) / Wrong (0)")
    ax.set_title("Agreement Ratio vs Correctness")

    # Graph 2
    ax = axes[0, 1]
    type_stats["accuracy"].plot(
        kind="bar", ax=ax,
        color=["#4C72B0", "#DD8452", "#55A868"][:len(type_stats)],
        edgecolor="white"
    )
    ax.set_xlabel("Experimental Group")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Experimental Group")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=0)

    # Graph 3
    ax = axes[1, 0]
    df["agreement_ratio"].hist(bins=10, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(0.8, color="red", linestyle="--", label="Threshold (0.8)")
    ax.set_xlabel("Agreement Ratio")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Distribution of Agreement Ratios")
    ax.legend()

    # Graph 4
    ax = axes[1, 1]
    agree_acc = df.groupby("high_agreement")["is_correct"].mean()
    labels = ["Low (<0.8)", "High (>=0.8)"]
    values = [agree_acc.get(False, 0), agree_acc.get(True, 0)]
    ax.bar(labels, values, color=["#DD8452", "#4C72B0"], edgecolor="white")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: High vs Low Agreement")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "thesis_charts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nCharts saved -> {path}")


# --- Main Execution Runtime ---
if __name__ == "__main__":
    start_time = time.time()

    all_questions = load_logiqa(START_IDX, END_IDX)
    if not all_questions:
        print("ERROR: Failed to load dataset context.")
        exit(1)

    print(f"Targeting: {len(all_questions)} questions split into chunks of {BATCH_SIZE}.\n")

    for i in range(0, len(all_questions), BATCH_SIZE):
        chunk = all_questions[i: i + BATCH_SIZE]
        if not chunk:
            continue

        real_start_id = chunk[0]["id"]
        real_end_id   = chunk[-1]["id"]
        batch_id      = f"{real_start_id}_to_{real_end_id}"
        batch_file    = os.path.join(BATCH_DIR, f"batch_{batch_id}.csv")

        if os.path.exists(batch_file):
            print(f"Batch {batch_id} already exists. Skipping processing stage.")
            continue

        process_batch(chunk, batch_id, batch_file)

    # --- Merge Phase ---
    final_df = merge_all_batches()
    if not final_df.empty:
        final_df.to_csv(OUTPUT_CSV, index=False)
        final_df.to_json("results_full.json", orient="records", indent=2)
        print(f"\nMaster compilation records written to: {OUTPUT_CSV}")

        type_stats = analyze_results(final_df)
        plot_results(final_df, type_stats)
    else:
        print("Error: Aggregation failed. No processable rows compiled.")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime metric: {elapsed/60:.2f} minutes.")