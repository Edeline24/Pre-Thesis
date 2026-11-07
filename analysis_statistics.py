"""
Reproduces every number reported in the paper from the saved generations.
No API calls are made.

Usage (from the repository root):
    python analysis_statistics.py            # searches the current folder and subfolders
    python analysis_statistics.py path/to/results

It looks for the merged result files results_<model>_v3_T<temp>_N5.csv (as
written by prethesis.py). If a merged file is missing but its batch folder
output_batches_<model>_v3_T<temp>_N5/ exists, the batches are merged in memory.
It also needs category_labels.xlsx and answer_key_review_verdicts.csv.

Scoring rule (as in the paper): a question with no parsed generation has
agreement 0 and counts as incorrect, so every model is scored on 651 questions.
The output is printed and saved to analysis_output.txt.
"""
import glob
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import (binomtest, chi2, chi2_contingency, fisher_exact,
                         mannwhitneyu, pointbiserialr, ttest_1samp, ttest_rel,
                         wilcoxon)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
N = 5
SEED = 42
N_BOOT = 2000
BASE = 0.7
TEMPS = [0.0, 0.3, 0.7, 1.0]
MODELS = {"Llama": "llama_3_3_70b_instruct",
          "DeepSeek": "deepseek_v4_flash_0731",
          "Qwen": "qwen3_next_80b_a3b_instruct"}
ARG_CATS = ["Argument Strengthening", "Argument Weakening", "Assumption Identification"]
CB = "Constraint-Based Reasoning"

out = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out.append(line)


def find_one(name):
    hits = glob.glob(os.path.join(ROOT, "**", name), recursive=True)
    if not hits:
        sys.exit(f"Missing {name} (searched under {ROOT})")
    return hits[0]


def tstr(t):
    return f"{t:g}"


def load_run(slug, temp):
    stem = f"{slug}_v3_T{tstr(temp)}_N5"
    merged = glob.glob(os.path.join(ROOT, "**", f"results_{stem}.csv"), recursive=True)
    if merged:
        df = pd.read_csv(merged[0])
    else:
        batches = glob.glob(os.path.join(ROOT, "**", f"output_batches_{stem}", "batch_*.csv"), recursive=True)
        if not batches:
            sys.exit(f"Missing results for {stem}: neither results_{stem}.csv nor output_batches_{stem}/ found")
        df = pd.concat([pd.read_csv(f) for f in batches], ignore_index=True)
        df["_ok"] = df["n_valid_parses"] > 0
        df = (df.sort_values(["question_id", "_ok"], kind="mergesort")
                .drop_duplicates("question_id", keep="last").drop(columns="_ok"))
    df = df.sort_values("question_id").reset_index(drop=True)
    if len(df) != 651:
        sys.exit(f"{stem}: expected 651 questions, found {len(df)}")
    df["answers"] = df["all_answers"].apply(json.loads)
    df["tied"] = df["tied_answers"].apply(json.loads)
    for col in ["is_correct", "is_tie"]:
        df[col] = df[col].astype(str).str.lower().eq("true")
    df["ok"] = df["is_correct"]
    df["ar"] = df["agreement_over_n"]
    df["frac"] = [sum(a == k for a in ans) / N for ans, k in zip(df["answers"], df["correct_answer"])]
    return df


labels = (pd.read_excel(find_one("category_labels.xlsx"), sheet_name="final_dataset")
          [["item_number", "gold_label"]].rename(columns={"item_number": "question_id", "gold_label": "cat"}))
verdicts = pd.read_csv(find_one("answer_key_review_verdicts.csv"))

R = {}
for m, slug in MODELS.items():
    for t in TEMPS:
        R[(m, t)] = load_run(slug, t).merge(labels, on="question_id", how="left")
B = {m: R[(m, BASE)] for m in MODELS}


def auroc(score, y):
    y = np.asarray(y, bool)
    s = np.asarray(score, float)
    return mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())


def boot(stat_fn, n_items, rng):
    stats = [stat_fn(rng.integers(0, n_items, n_items)) for _ in range(N_BOOT)]
    return np.percentile(stats, [2.5, 97.5])


def pct(x):
    return f"{100 * x:.1f}"


def unanimous_error_share(d, keep=None):
    k = np.ones(len(d), bool) if keep is None else np.asarray(keep)
    err = k & ~d["ok"].values
    return ((d["ar"].values == 1) & err).sum() / err.sum()


# ---------------------------------------------------------------- Abstract / IV-D
say("=== Abstract and IV-D: unanimous errors at T = 0.7 ===")
for m in MODELS:
    say(f"  {m}: {pct(unanimous_error_share(B[m]))}% of errors are unanimous")

# ---------------------------------------------------------------- III-F
say("\n=== III-F: two-vote plurality without a tie, per run ===")
for (m, t), d in R.items():
    n2 = 0
    for ans in d["answers"]:
        v = [a for a in ans if a is not None]
        if not v:
            continue
        c = Counter(v)
        top = max(c.values())
        if top == 2 and sum(1 for k in c.values() if k == top) == 1:
            n2 += 1
    say(f"  {m} T={tstr(t)}: {n2}")

# ---------------------------------------------------------------- IV-A
say("\n=== IV-A: parsing, truncation, unparsed questions ===")
for (m, t), d in R.items():
    gens = N * len(d)
    trunc = int(d["n_truncated"].sum())
    say(f"  {m} T={tstr(t)}: parsed {pct(d['n_valid_parses'].sum() / gens)}% of generations, "
        f"truncated {trunc} ({pct(trunc / gens)}%), unparsed questions {int((d['n_valid_parses'] == 0).sum())}")

q = B["Qwen"]
say(f"  Qwen baseline accuracy: {pct(q['ok'].mean())}% on 651, "
    f"{pct(q.loc[q['n_valid_parses'] > 0, 'ok'].mean())}% excluding unparsed questions")
matched = set.intersection(*[set(B[m].loc[B[m]["n_valid_parses"] > 0, "question_id"]) for m in MODELS])
say(f"  Matched set ({len(matched)} questions): " +
    ", ".join(f"{m} {pct(B[m].loc[B[m]['question_id'].isin(matched), 'ok'].mean())}%" for m in MODELS))

say("\n=== IV-A: ties per condition and alternative tie policies ===")
max_change = 0.0
for (m, t), d in R.items():
    tie = d["is_tie"].values
    k = np.array([max(len(x), 1) for x in d["tied"]])
    key_in = np.array([key in x for key, x in zip(d["correct_answer"], d["tied"])])
    frac_credit = (d["ok"].values + (tie & key_in) / k).mean()
    optimistic = (d["ok"].values | (tie & key_in)).mean()
    max_change = max(max_change, frac_credit - d["ok"].mean(), optimistic - d["ok"].mean())
    say(f"  {m} T={tstr(t)}: ties {int(tie.sum())} (key among tied: {int((tie & key_in).sum())}) | "
        f"accuracy strict {pct(d['ok'].mean())}, 1/k credit {pct(frac_credit)}, ties correct {pct(optimistic)}")
say(f"  Largest accuracy change under the alternative policies: {100 * max_change:.1f} points")
for m in MODELS:
    d = B[m]
    tie = d["is_tie"].values
    key_in = np.array([key in x for key, x in zip(d["correct_answer"], d["tied"])])
    say(f"  {m} baseline AUROC: strict {auroc(d['ar'], d['ok']):.2f}, "
        f"ties counted correct {auroc(d['ar'], d['ok'].values | (tie & key_in)):.2f}")

say("\n=== IV-A: questions where all 15 baseline generations agree against the key ===")
disputed = []
for qid in range(651):
    pooled = sum((B[m]["answers"][qid] for m in MODELS), [])
    if None not in pooled and len(set(pooled)) == 1 and pooled[0] != B["Llama"]["correct_answer"][qid]:
        disputed.append(qid)
say(f"  {len(disputed)} questions, identical to the verdict file: "
    f"{sorted(disputed) == sorted(verdicts['question_id'].tolist())}")
say("  " + ", ".join(f"{v}: {n}" for v, n in verdicts["verdict"].value_counts().items()))

# ---------------------------------------------------------------- IV-B and main results table
say("\n=== IV-B: main results at T = 0.7 ===")
rng = np.random.default_rng(SEED)
for m in MODELS:
    d = B[m]
    ok = d["ok"].values
    ar = d["ar"].values
    acc_ci = boot(lambda i: ok[i].mean(), len(d), rng)
    auc_ci = boot(lambda i: auroc(ar[i], ok[i]), len(d), rng)
    diff = ok.astype(float) - d["frac"].values
    err = ~ok
    say(f"  {m}: accuracy {pct(ok.mean())} [CI {pct(acc_ci[0])}-{pct(acc_ci[1])}] | "
        f"single-sample {pct(d['frac'].mean())} | voting gain {100 * diff.mean():+.1f} "
        f"(paired t p={ttest_1samp(diff, 0).pvalue:.2g}) | unanimous {pct((ar == 1).mean())}% | "
        f"AUROC {auroc(ar, ok):.2f} [CI {auc_ci[0]:.2f}-{auc_ci[1]:.2f}] | "
        f"consistent errors {int(((ar == 1) & err).sum())} ({100 * unanimous_error_share(d):.0f}% of errors)")

say("  Model comparisons over the four runs:")
avg = {m: np.mean([R[(m, t)]["ok"].values for t in TEMPS], axis=0) for m in MODELS}
runacc = {m: np.array([R[(m, t)]["ok"].mean() * 100 for t in TEMPS]) for m in MODELS}
for a, b in itertools.combinations(MODELS, 2):
    say(f"    {a} vs {b}: Wilcoxon p={wilcoxon(avg[a], avg[b]).pvalue:.2g}, "
        f"run-level paired t p={ttest_rel(runacc[a], runacc[b]).pvalue:.2g}")
say("  SD of accuracy over the four temperature runs (points): " +
    ", ".join(f"{m} {runacc[m].std(ddof=1):.2f}" for m in MODELS))

say("  Stability: every subset of 3 of the 5 generations at T = 0.7")
for m in MODELS:
    d = B[m]
    accs, aucs = [], []
    for sub in itertools.combinations(range(N), 3):
        corr, agr = [], []
        for ans, key in zip(d["answers"], d["correct_answer"]):
            v = [ans[i] for i in sub if ans[i] is not None]
            if not v:
                corr.append(False)
                agr.append(0.0)
                continue
            c = Counter(v)
            top = max(c.values())
            winners = [x for x, n in c.items() if n == top]
            corr.append(len(winners) == 1 and winners[0] == key)
            agr.append(top / 3)
        accs.append(np.mean(corr))
        aucs.append(auroc(agr, corr))
    say(f"    {m}: largest accuracy change {100 * max(abs(np.array(accs) - d['ok'].mean())):.1f} points, "
        f"mean AUROC {np.mean(aucs):.3f} vs {auroc(d['ar'], d['ok']):.3f} at N = 5")

# ---------------------------------------------------------------- IV-C
say("\n=== IV-C: agreement as an indicator ===")
for m in MODELS:
    d = B[m]
    lv = d["ar"].round(1)
    parts = [f"AR {v:.1f}: {pct(d.loc[lv == v, 'ok'].mean())}% (n={int((lv == v).sum())})"
             for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if (lv == v).any()]
    say(f"  {m}: " + ", ".join(parts))
rs = [pointbiserialr(d["ok"], d["ar"]) for d in R.values()]
say(f"  Point-biserial r across all runs: {min(r[0] for r in rs):.2f} to {max(r[0] for r in rs):.2f}, "
    f"largest p = {max(r[1] for r in rs):.1g}")
for m in MODELS:
    full = B[m][B[m]["n_valid_parses"] == N]
    say(f"  {m}: fully parsed questions {len(full)}, AUROC {auroc(full['ar'], full['ok']):.2f}")
for m in MODELS:
    d = B[m]
    low = d["ar"].round(1) <= 0.4
    say(f"  {m}: questions with AR <= 0.4 that contain a truncated generation: "
        f"{int((low & (d['n_truncated'] > 0)).sum())} of {int(low.sum())}")

# ---------------------------------------------------------------- IV-D
say("\n=== IV-D: consistent errors ===")
shares = {(m, t): unanimous_error_share(d) for (m, t), d in R.items()}
say(f"  Unanimous-error share across all runs: {100 * min(shares.values()):.1f}% to {100 * max(shares.values()):.1f}%")
for m in MODELS:
    d = B[m]
    err = ~d["ok"].values
    say(f"  {m}: relaxed share (AR >= 0.8) {100 * ((d['ar'].values >= 0.8) & err).sum() / err.sum():.1f}%")
ex21 = verdicts.loc[verdicts["verdict"].str.contains("better supported"), "question_id"]
ex32 = verdicts["question_id"]
for m in MODELS:
    d = B[m]
    a = unanimous_error_share(d, ~d["question_id"].isin(ex21).values)
    b = unanimous_error_share(d, ~d["question_id"].isin(ex32).values)
    say(f"  {m}: excluding the 21 better-supported questions {100 * a:.1f}%, excluding all 32 {100 * b:.1f}%")
d = B["DeepSeek"]
key_in = [key in x for key, x, tie in zip(d["correct_answer"], d["tied"], d["is_tie"]) if tie]
say(f"  DeepSeek ties at T = 0.7: {len(key_in)}, correct option among the tied answers in {sum(key_in)}")
say(f"  Q286 ({B['Llama'].loc[286, 'cat']}, key {B['Llama'].loc[286, 'correct_answer']}): " +
    ", ".join(f"{m} {B[m].loc[286, 'answers']}" for m in MODELS))

# ---------------------------------------------------------------- IV-E
say("\n=== IV-E: reasoning categories ===")
for m in MODELS:
    d = B[m]
    p = chi2_contingency(pd.crosstab(d["cat"], d["ok"]))[1]
    acc = d.groupby("cat")["ok"].mean().mul(100).round(0).astype(int).to_dict()
    say(f"  {m}: accuracy by category {acc}, chi-square p={p:.2g}")
    err = d[~d["ok"]]
    counts = {c: f"{int((g['ar'] == 1).sum())}/{len(g)}" for c, g in err.groupby("cat")}
    say(f"    unanimous errors / errors: {counts}")
    aucs = {c: round(auroc(g["ar"], g["ok"]), 2) for c, g in d.groupby("cat")}
    say(f"    AUROC by category: {aucs}")
    u = err["ar"] == 1
    cb = err["cat"] == CB
    ag = err["cat"].isin(ARG_CATS)
    table = [[int((u & cb).sum()), int((~u & cb).sum())], [int((u & ag).sum()), int((~u & ag).sum())]]
    say(f"    Fisher (Constraint-Based vs argument-based): {pct(table[0][0] / sum(table[0]))}% vs "
        f"{pct(table[1][0] / sum(table[1]))}%, p={fisher_exact(table)[1]:.2g}")

# ---------------------------------------------------------------- IV-F
say("\n=== IV-F: temperature ===")
for m in MODELS:
    mat = np.column_stack([R[(m, t)]["ok"].values for t in TEMPS]).astype(int)
    k = mat.shape[1]
    col, row, tot = mat.sum(0), mat.sum(1), mat.sum()
    q_stat = (k - 1) * (k * (col ** 2).sum() - tot ** 2) / (k * tot - (row ** 2).sum())
    a, b = mat[:, 0].astype(bool), mat[:, -1].astype(bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    say(f"  {m}: accuracy " + ", ".join(f"T={tstr(t)} {pct(mat[:, i].mean())}" for i, t in enumerate(TEMPS)) +
        f" | max change {100 * (mat.mean(0).max() - mat.mean(0).min()):.1f} | Cochran Q p={chi2.sf(q_stat, k - 1):.2f}"
        f" | McNemar T0 vs T1 p={binomtest(n01, n01 + n10).pvalue:.2f}")
    say("    unanimous % " + ", ".join(f"T={tstr(t)} {pct((R[(m, t)]['ar'] == 1).mean())}" for t in TEMPS) +
        f" | unanimous-error share T0 {pct(shares[(m, 0.0)])}% -> T1 {pct(shares[(m, 1.0)])}%"
        f" | AUROC T0 {auroc(R[(m, 0.0)]['ar'], R[(m, 0.0)]['ok']):.3f}"
        f" -> T1 {auroc(R[(m, 1.0)]['ar'], R[(m, 1.0)]['ok']):.3f}")

with open("analysis_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\nSaved -> analysis_output.txt")
