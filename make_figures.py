"""
Reproduces the four result figures of the paper from the saved generations.

Usage (from the repository root):
    python make_figures.py            # searches the current folder and subfolders
    python make_figures.py path/to/results

Uses the merged files results_<model>_v3_T<temp>_N5.csv, or merges the batch
folders output_batches_<model>_v3_T<temp>_N5/ when a merged file is missing.
Needs category_labels.xlsx. Figures are written to ./figures/.
Scoring rule as in the paper: a question with no parsed generation has
agreement 0 and counts as incorrect (651 questions per model).
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.linewidth": 0.8})
rng = np.random.default_rng(42)
os.makedirs("figures", exist_ok=True)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."


def find_one(name):
    hits = glob.glob(os.path.join(ROOT, "**", name), recursive=True)
    if not hits:
        sys.exit(f"Missing {name} (searched under {ROOT})")
    return hits[0]


labels = (pd.read_excel(find_one("category_labels.xlsx"), sheet_name="final_dataset")[["item_number", "gold_label"]]
          .rename(columns={"item_number": "question_id", "gold_label": "cat"}))
MODELS = [("llama_3_3_70b_instruct", "Llama-3.3-70B", "#4C72B0"),
          ("deepseek_v4_flash_0731", "DeepSeek-V4-Flash", "#DD8452"),
          ("qwen3_next_80b_a3b_instruct", "Qwen3-Next-80B", "#55A868")]
CATS = ["Constraint-Based Reasoning", "Inference Generation", "Argument Weakening",
        "Argument Strengthening", "Assumption Identification", "Others"]
SHORT = ["Constraint-Based", "Inference Gen.", "Arg. Weakening", "Arg. Strength.", "Assump. Ident.", "Others"]
W, BAR = 3.5, 0.27


def load(model, temp):
    stem = f"{model}_v3_T{temp}_N5"
    merged = glob.glob(os.path.join(ROOT, "**", f"results_{stem}.csv"), recursive=True)
    if merged:
        d = pd.read_csv(merged[0])
    else:
        batches = glob.glob(os.path.join(ROOT, "**", f"output_batches_{stem}", "batch_*.csv"), recursive=True)
        if not batches:
            sys.exit(f"Missing results for {stem}")
        d = pd.concat([pd.read_csv(f) for f in batches], ignore_index=True)
        d["_ok"] = d["n_valid_parses"] > 0
        d = (d.sort_values(["question_id", "_ok"], kind="mergesort")
               .drop_duplicates("question_id", keep="last").drop(columns="_ok"))
    d = d.merge(labels, on="question_id")
    d["ok"] = d["is_correct"].astype(str).str.lower().eq("true")
    return d


def legend(ax):
    ax.legend(ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.99),
              handlelength=1.0, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Accuracy by agreement level
levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
x = np.arange(len(levels))
fig, ax = plt.subplots(figsize=(W, 2.3))
for j, (m, name, color) in enumerate(MODELS):
    d = load(m, "0.7")
    acc, cnt = [], []
    for v in levels:
        s = d["agreement_over_n"].round(1) == v
        cnt.append(int(s.sum()))
        acc.append(d.loc[s, "ok"].mean() if s.sum() else 0)
    bars = ax.bar(x + (j - 1) * BAR, acc, BAR, color=color, label=name)
    for b, k in zip(bars, cnt):
        if k:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(k),
                    ha="center", va="bottom", rotation=90, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.1f}" for v in levels])
ax.set_xlabel("Agreement ratio")
ax.set_ylabel("Majority-vote accuracy")
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.5, 1.0])
legend(ax)
save(fig, "fig1_accuracy_by_agreement")

# Accuracy by category
x = np.arange(len(CATS))
fig, ax = plt.subplots(figsize=(W, 2.4))
for j, (m, name, color) in enumerate(MODELS):
    d = load(m, "0.7")
    ax.bar(x + (j - 1) * BAR, [d.loc[d["cat"] == c, "ok"].mean() for c in CATS], BAR, color=color, label=name)
ax.set_xticks(x)
ax.set_xticklabels(SHORT, rotation=30, ha="right")
ax.set_ylabel("Majority-vote accuracy")
ax.set_ylim(0, 1.0)
legend(ax)
save(fig, "fig2_accuracy_by_category")

# Unanimous errors by category, with unanimous/all-error counts
fig, ax = plt.subplots(figsize=(W, 2.6))
for j, (m, name, color) in enumerate(MODELS):
    d = load(m, "0.7")
    err = d[~d["ok"]]
    pct, txt = [], []
    for c in CATS:
        g = err[err["cat"] == c]
        u = int((g["agreement_over_n"] == 1).sum())
        pct.append(100 * u / len(g))
        txt.append(f"{u}/{len(g)}")
    bars = ax.bar(x + (j - 1) * BAR, pct, BAR, color=color, label=name)
    for b, t in zip(bars, txt):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, t,
                ha="center", va="bottom", rotation=90, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(SHORT, rotation=30, ha="right")
ax.set_ylabel("Unanimous errors (% of errors)")
ax.set_ylim(0, 130)
ax.set_yticks([0, 20, 40, 60, 80, 100])
legend(ax)
save(fig, "fig3_consistent_errors_by_category")

# Temperature: accuracy with bootstrap CI (top) and unanimous share (bottom)
temps, tvals = ["0", "0.3", "0.7", "1"], [0.0, 0.3, 0.7, 1.0]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.3), sharex=True)
for j, (m, name, color) in enumerate(MODELS):
    acc, lo, hi, una = [], [], [], []
    for t in temps:
        d = load(m, t)
        o = d["ok"].values
        acc.append(o.mean())
        boot = [o[rng.integers(0, len(o), len(o))].mean() for _ in range(2000)]
        lo.append(acc[-1] - np.percentile(boot, 2.5))
        hi.append(np.percentile(boot, 97.5) - acc[-1])
        una.append((d["agreement_over_n"] == 1).mean() * 100)
    a1.errorbar(np.array(tvals) + (j - 1) * 0.012, acc, yerr=[lo, hi], color=color,
                marker="o", ms=3.5, capsize=2.5, lw=1.2, label=name)
    a2.plot(tvals, una, color=color, marker="o", ms=3.5, lw=1.2)
a1.set_ylabel("Accuracy")
a1.set_ylim(0.5, 0.9)
a1.set_yticks([0.5, 0.7, 0.9])
a2.set_ylabel("Unanimous (%)")
a2.set_xlabel("Temperature")
a2.set_xticks(tvals)
a2.set_ylim(70, 90)
a2.set_yticks([70, 80, 90])
legend(a1)
save(fig, "fig4_temperature")
print("Figures written to ./figures/")
