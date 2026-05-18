import os, json, time, re, urllib.request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"

MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]

N_GENERATIONS   = 3
N_QUESTIONS     = None   # semua ~651 soal
TEMPERATURE     = 0.7
TOP_P           = 0.9
MAX_TOKENS      = 300
OUTPUT_CSV      = "results.csv"
OUTPUT_DIR      = "output_charts"
CHECKPOINT_FILE = "checkpoint.json"
DELAY           = 1.5   # delay per model (lebih pendek karena beda model)
MAX_WORKERS     = 3     # 3 thread paralel, 1 per model

os.makedirs(OUTPUT_DIR, exist_ok=True)
client = Groq(api_key=GROQ_API_KEY)
_last_req = {m: 0.0 for m in MODELS}

# ─── DATASET ──────────────────────────────────────────────────────────────────
def load_logiqa(n_questions=None):
    url = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt"
    print("Downloading LogiQA dataset...")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        print(f"Download error: {e}"); return []

    lines = raw.strip().split("\n")
    questions, rtypes = [], ["deductive", "inductive", "conditional"]
    i = 0
    while i < len(lines):
        if n_questions and len(questions) >= n_questions: break
        if lines[i].strip() == "": i += 1; continue
        if i + 6 >= len(lines): break
        try:
            ans = lines[i].strip().lower()
            if ans not in ["a","b","c","d"]: i += 1; continue
            opts = "\n".join([
                f"A. {lines[i+3].strip().lstrip('A.').strip()}",
                f"B. {lines[i+4].strip().lstrip('B.').strip()}",
                f"C. {lines[i+5].strip().lstrip('C.').strip()}",
                f"D. {lines[i+6].strip().lstrip('D.').strip()}",
            ])
            questions.append({
                "id": len(questions),
                "context": lines[i+1].strip(),
                "question": lines[i+2].strip(),
                "options": opts,
                "correct_answer": chr(65 + ord(ans) - ord("a")),
                "reasoning_type": rtypes[len(questions) % 3],
            })
            i += 8
        except: i += 1

    print(f"Loaded {len(questions)} questions.\n")
    return questions

# ─── PROMPT ───────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Context: {context}

Question: {question}

Options:
{options}

Think step by step, then answer:
Final Answer: [A/B/C/D]"""

def build_prompt(q):
    return PROMPT_TEMPLATE.format(
        context=q["context"], question=q["question"], options=q["options"]
    )

# ─── API CALL (per model, thread-safe) ────────────────────────────────────────
def call_model(prompt, model):
    for attempt in range(5):
        elapsed = time.time() - _last_req[model]
        if elapsed < DELAY:
            time.sleep(DELAY - elapsed)
        try:
            _last_req[model] = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "decommissioned" in err:
                print(f"\n[{model}] DECOMMISSIONED!"); return None
            wait_match = re.search(r"try again in\s+([\d.]+)m([\d.]+)s", err)
            if wait_match:
                wait = float(wait_match.group(1))*60 + float(wait_match.group(2))
                print(f"\n[{model}] Rate limit, tunggu {wait:.0f}s...")
                time.sleep(wait + 2)
            elif "429" in err:
                time.sleep(15 * (attempt + 1))
            else:
                time.sleep(5)
    return None

# ─── EXTRACT ANSWER ───────────────────────────────────────────────────────────
def extract_answer(text):
    if not text: return None
    m = re.search(r"Final Answer[:\s]+([A-D])", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    for line in reversed([l.strip() for l in text.split("\n") if l.strip()]):
        m = re.search(r"\b([A-D])\b", line)
        if m: return m.group(1).upper()
    return None

def calculate_agreement(answers):
    valid = [a for a in answers if a is not None]
    if not valid: return 0.0, None
    c = Counter(valid)
    top, cnt = c.most_common(1)[0]
    return round(cnt / N_GENERATIONS, 3), top

# ─── PROCESS 1 QUESTION (parallel generations) ────────────────────────────────
def process_question(q):
    prompt = build_prompt(q)

    # Kirim 3 generasi secara PARALEL ke 3 model berbeda
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(call_model, prompt, MODELS[i]): i for i in range(N_GENERATIONS)}
        answers = [None] * N_GENERATIONS
        for future in as_completed(futures):
            idx = futures[future]
            text = future.result()
            answers[idx] = extract_answer(text)

    agree, majority = calculate_agreement(answers)
    is_correct = (majority == q["correct_answer"]) if majority else False

    return {
        "question_id"    : q["id"],
        "reasoning_type" : q["reasoning_type"],
        "correct_answer" : q["correct_answer"],
        "all_answers"    : json.dumps(answers),
        "majority_answer": majority,
        "agreement_ratio": agree,
        "is_correct"     : is_correct,
        "high_agreement" : agree >= 0.8,
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_experiment(questions):
    # Checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f: results = json.load(f)
        done_ids = {r["question_id"] for r in results}
        print(f"Resume: {len(done_ids)} soal sudah selesai.")
    else:
        results, done_ids = [], set()

    pending = [q for q in questions if q["id"] not in done_ids]
    print(f"Pending: {len(pending)} soal\n")

    for q in tqdm(pending, desc="Processing", unit="soal"):
        row = process_question(q)
        results.append(row)

        with open(CHECKPOINT_FILE, "w") as f: json.dump(results, f)

        status = "✓" if row["is_correct"] else "✗"
        tqdm.write(f"  Q{q['id']:03d} | {row['reasoning_type']:12s} | "
                   f"Agree:{row['agreement_ratio']:.2f} | "
                   f"Ans:{row['all_answers']}→{row['majority_answer']} | "
                   f"Correct:{q['correct_answer']} | {status}")

    if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    return pd.DataFrame(results)

# ─── ANALISIS ─────────────────────────────────────────────────────────────────
def analyze_and_plot(df):
    print(f"\nOverall Accuracy : {df['is_correct'].mean():.3f}")
    print(f"Mean Agreement   : {df['agreement_ratio'].mean():.3f}")

    type_stats = df.groupby("reasoning_type").agg(
        accuracy=("is_correct","mean"),
        mean_agreement=("agreement_ratio","mean"),
        count=("question_id","count")
    ).round(3)
    print("\n--- Per Reasoning Type ---")
    print(type_stats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LLM Self-Consistency — LogiQA ({len(df)} soal)", fontsize=14, fontweight="bold")

    ax = axes[0,0]
    colors = df["is_correct"].map({True:"steelblue", False:"tomato"})
    ax.scatter(df["agreement_ratio"], df["is_correct"].astype(int), c=colors, alpha=0.4, s=20)
    ax.set_xlabel("Agreement Ratio"); ax.set_ylabel("Correct"); ax.set_title("Agreement vs Correctness")

    ax = axes[0,1]
    type_stats["accuracy"].plot(kind="bar", ax=ax, color=["#4C72B0","#DD8452","#55A868"], edgecolor="white")
    ax.set_ylim(0,1); ax.set_title("Accuracy per Reasoning Type")

    ax = axes[1,0]
    df["agreement_ratio"].hist(bins=10, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(0.8, color="red", linestyle="--", label="0.8"); ax.legend()
    ax.set_title("Distribusi Agreement Ratio")

    ax = axes[1,1]
    acc = df.groupby("high_agreement")["is_correct"].mean()
    ax.bar(["Low (<0.8)","High (≥0.8)"], [acc.get(False,0), acc.get(True,0)],
           color=["#DD8452","#4C72B0"], edgecolor="white")
    ax.set_ylim(0,1); ax.set_title("Accuracy: High vs Low Agreement")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "thesis_charts.png"), dpi=150, bbox_inches="tight")
    df.drop(columns=["all_answers"], errors="ignore").to_csv(OUTPUT_CSV, index=False)
    print(f"\nChart → {OUTPUT_DIR}/thesis_charts.png")
    print(f"CSV   → {OUTPUT_CSV}")

# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = load_logiqa(N_QUESTIONS)
    if not questions: exit(1)
    df = run_experiment(questions)
    if len(df) > 0:
        analyze_and_plot(df)
        print(f"\n✓ Done! {len(df)} soal selesai.")