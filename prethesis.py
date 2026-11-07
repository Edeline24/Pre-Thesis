"""
Self-consistency experiment on LogiQA.

All configuration (model, temperature, N, etc.) is passed as command-line
arguments -- there are no numbers to edit inside this file. Run
`python prethesis.py -h` for the full list, or see "Usage" below.

Features: batched execution, fault tolerance, auto-resume, automatic merging.

Changes from the previous version:
  * Options are parsed by their letter prefix, not by line position.
    Previously 19 questions had their option lines in A/C/B/D order, so
    options B and C were swapped, and 9 questions had prefixes without a
    period ("A No dangshen"), so the letter leaked into the option text.
  * The len(opt) < 2 filter was removed. It discarded 22 questions whose
    options are genuinely single characters (F, G, 4, 5) -- almost all of
    them assignment/ordering puzzles, so the dropped items were concentrated
    in a single reasoning category.
  * question_id is now the index over the full 651-question set, matching
    the numbering used in the labeling files.
  * MAX_TOKENS raised to 6000 (v3). At 1500, long constraint puzzles were cut
    off before the answer. finish_reason is recorded so any truncation is
    visible instead of masquerading as an answer.
  * extract_answer no longer guesses a random letter from the last line.
  * Questions whose generations all fail to parse are still written as a
    row (not silently skipped), and are retried on the next run.
  * Configuration moved from source constants to CLI arguments, so three
    people can use the same file without overwriting each other's edits.

Usage (PowerShell on Windows)
-----------------------------
Set the API key once per terminal session:
    $env:OPENROUTER_API_KEY="sk-or-..."
(macOS/Linux bash: export OPENROUTER_API_KEY="sk-or-...")

Check the setup first (1 API call):
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct --check

Baseline (used for the main results table):
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct

Temperature sweep (only --temperature/--n change; each combination gets its
own folder and files automatically, so runs never collide):
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct --temperature 0.3
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct --temperature 1.0
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct --n 10

Other models (teammates only change --model):
    python prethesis.py --model deepseek/deepseek-v4-flash-0731 --disable-reasoning
    python prethesis.py --model qwen/qwen3-next-80b-a3b-instruct

Pin an OpenRouter provider (recommended, so the generations of a question are
not served by different backends/quantizations):
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct --provider deepinfra

Small pilot, then the full run (the first 20 are not repeated):
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct --end 20
    python prethesis.py --model meta-llama/llama-3.3-70b-instruct

If the run stops midway (power, network, Ctrl+C): rerun the EXACT SAME
command. The configuration is locked in run_config.json inside the batch
folder; different arguments are rejected rather than mixed in.
"""

import os
import sys
import json
import time
import re
import argparse
import urllib.request
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from openai import OpenAI
import random
from math import log2
import numpy as np

random.seed(42)
np.random.seed(42)

# On Windows, a console or redirected output using cp1252 cannot print some
# characters; replace them instead of crashing at the end of a paid run.
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Self-consistency experiment on LogiQA. Every value below "
                     "has a default; the only REQUIRED argument is --model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Exact model ID, e.g. meta-llama/llama-3.3-70b-instruct "
                        "(REQUIRED -- the only thing that differs per person)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--n", type=int, default=5, dest="n_generations",
                   help="Number of generations per question (default 5)")
    p.add_argument("--max-tokens", type=int, default=6000,
                   help="Default 6000. At 1500, Qwen was cut off on 7 of 20 pilot "
                        "questions -- all multi-step assignment/ordering puzzles, "
                        "so truncation would bias one reasoning category. Billing "
                        "is per token actually generated, so a high cap only costs "
                        "more on the responses that need it.")
    p.add_argument("--start", type=int, default=0, dest="start_idx")
    p.add_argument("--end", type=int, default=None, dest="end_idx",
                   help="None = through the last question (651)")
    p.add_argument("--batch-size", type=int, default=50)

    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="Change this when not using OpenRouter, e.g. "
                        "https://api.deepseek.com")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="Name of the environment variable holding the API key "
                        "(default OPENROUTER_API_KEY)")
    p.add_argument("--provider", default=None,
                   help="Pin OpenRouter to a single provider (e.g. deepinfra, "
                        "fireworks), so the run does not switch backends or "
                        "quantizations mid-experiment. Ignored when --base-url "
                        "is not OpenRouter.")
    p.add_argument("--extra-body", default=None,
                   help='Extra JSON for the request body, e.g. '
                        '\'{"reasoning": {"enabled": false}}\' for DeepSeek. '
                        "Merged with --provider when both are given.")
    p.add_argument("--disable-reasoning", action="store_true",
                   help='Shortcut for --extra-body \'{"reasoning": {"enabled": false}}\'. '
                        "Use this instead of typing JSON in PowerShell, which "
                        "can strip the inner quotes.")
    p.add_argument("--run-tag", default="v3",
                   help="Bump this when the parser/prompt/defaults change, so old "
                        "results are not mixed in (default v3; v2 used max_tokens=1500)")
    p.add_argument("--check", action="store_true",
                   help="Make 1 API call and stop: checks connectivity, answer "
                        "format, and whether reasoning is really disabled. "
                        "Run this first whenever you switch models.")
    return p.parse_args()


ARGS = parse_args()

# --- Configuration (all from ARGS, nothing hardcoded) ---
API_KEY = os.getenv(ARGS.api_key_env)
if not API_KEY:
    raise ValueError(f"Environment variable {ARGS.api_key_env} is not set.")

MODEL_NAME    = ARGS.model
N_GENERATIONS = ARGS.n_generations
START_IDX     = ARGS.start_idx
END_IDX       = ARGS.end_idx
BATCH_SIZE    = ARGS.batch_size

TEMPERATURE   = ARGS.temperature
TOP_P         = ARGS.top_p
MAX_TOKENS    = ARGS.max_tokens

# extra_body is sent to the API as-is (provider-specific features, e.g.
# reasoning control or OpenRouter provider pinning). --provider and
# --extra-body are merged once here so call_model doesn't care where they
# came from.
EXTRA_BODY = json.loads(ARGS.extra_body) if ARGS.extra_body else {}
if ARGS.disable_reasoning:
    EXTRA_BODY.setdefault("reasoning", {})["enabled"] = False
if ARGS.provider and "openrouter.ai" in ARGS.base_url:
    EXTRA_BODY.setdefault("provider", {})["only"] = [ARGS.provider]

# All output names are derived from model + run-tag + sampling config, so no
# file names need to be changed by hand. Each T/N combination automatically
# gets its own folder, so temperature sweeps never overwrite each other or
# resume from the wrong run.
_SLUG         = re.sub(r"[^a-zA-Z0-9]+", "_", MODEL_NAME.split("/")[-1]).strip("_")
_CFG          = f"T{TEMPERATURE:g}_N{N_GENERATIONS}"
_STEM         = f"{_SLUG}_{ARGS.run_tag}_{_CFG}"
OUTPUT_CSV    = f"results_{_STEM}.csv"
OUTPUT_JSON   = f"results_{_STEM}.json"
OUTPUT_DIR    = f"output_charts_{_STEM}"
BATCH_DIR     = f"output_batches_{_STEM}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)
print(f"Run: {MODEL_NAME} | T={TEMPERATURE} top_p={TOP_P} N={N_GENERATIONS} "
      f"max_tokens={MAX_TOKENS} | questions {START_IDX}-{END_IDX or 'end'}")
print(f"Endpoint: {ARGS.base_url} (key from ${ARGS.api_key_env})")
if EXTRA_BODY:
    print(f"extra_body: {json.dumps(EXTRA_BODY)}")
print(f"Output -> {OUTPUT_CSV}\n")

# --- Core Functions ---

DATASET_SIZE = None

OPTION_STRICT = re.compile(r"^\s*([A-D])\s*[\.\)\?:]\s*(.*)$")
OPTION_LOOSE  = re.compile(r"^\s*([A-D])\s+(.*)$")


def parse_option_block(raw_lines):
    """Return a {letter: text} dict from the 4 option lines.

    Three strategies are tried in order: a prefix with punctuation, a prefix
    without punctuation (accepted only if the four letters are exactly
    A/B/C/D, so option text that happens to start with the word "A ..." is
    not truncated), and finally line position. The third strategy is flagged
    so it can be audited.
    """
    for pattern in (OPTION_STRICT, OPTION_LOOSE):
        hits = [pattern.match(line) for line in raw_lines]
        if all(hits) and sorted(m.group(1) for m in hits) == list("ABCD"):
            return {m.group(1): m.group(2).strip() for m in hits}, False
    positional = {}
    for letter, line in zip("ABCD", raw_lines):
        positional[letter] = re.sub(rf"^{letter}\s*[\.\)\?:]?\s*", "", line).strip()
    return positional, True


def load_logiqa(start_idx=0, end_idx=None):
    url = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt"
    cache = "logiqa_test.txt"

    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            raw = f.read()
    else:
        print("Downloading LogiQA dataset from GitHub...")
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                raw = response.read().decode("utf-8")
        except Exception as e:
            print(f"Download error: {e}")
            return []
        with open(cache, "w", encoding="utf-8") as f:
            f.write(raw)

    lines = raw.split("\n")
    questions = []
    i = 0
    n_positional = 0

    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue

        if i + 6 >= len(lines):
            break

        ans_letter = lines[i].strip().lower()
        if ans_letter not in ["a", "b", "c", "d"]:
            i += 1
            continue

        context  = lines[i + 1].strip()
        question = lines[i + 2].strip()
        opts, fell_back = parse_option_block([lines[i + 3 + k].strip() for k in range(4)])
        n_positional += int(fell_back)

        # id = position in the full 651-question list, with no filtering
        qid = len(questions)
        questions.append({
            "id"             : qid,
            "context"        : context,
            "question"       : question,
            "options"        : "\n".join(f"{L}. {opts[L]}" for L in "ABCD"),
            "correct_answer" : ans_letter.upper(),
            "option_parse_fallback": fell_back,
            "has_empty_option": any(not opts[L].strip() for L in "ABCD"),
        })
        i += 8

    global DATASET_SIZE
    DATASET_SIZE = len(questions)
    print(f"Loaded {len(questions)} total dataset questions.")
    if n_positional:
        print(f"  note: {n_positional} questions had options read by line position "
              f"(column option_parse_fallback = True)")
    empties = sum(q["has_empty_option"] for q in questions)
    if empties:
        print(f"  note: {empties} questions have an empty option after parsing")
    print()

    return questions[start_idx:end_idx]


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
    total = len(valid)
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

client = OpenAI(
    base_url=ARGS.base_url,
    api_key=API_KEY,
)


class ApiFailure(RuntimeError):
    """The API could not be reached. Deliberately stops the run -- it is NOT
    recorded as a failed generation, because a network outage is not model
    behavior, and recording it would permanently corrupt that question's
    agreement score."""


MAX_ATTEMPTS = 6
FATAL_STATUS = {400, 401, 402, 403, 404, 422}  # bad key/model/credits: retrying is pointless


def _suggested_wait(err_text):
    """Wait time suggested by the server in the error message (Groq/OpenAI format)."""
    m = re.search(r"try again in ([\dm\.s]+)", err_text)
    if not m:
        return None
    mins = re.search(r"(\d+)m", m.group(1))
    secs = re.search(r"([\d\.]+)s", m.group(1))
    return (int(mins.group(1)) * 60 if mins else 0) + (float(secs.group(1)) if secs else 0) + 5


def call_model(prompt):
    """Return (text, finish_reason, usage). Raises ApiFailure if the API keeps
    failing after several attempts -- progress is already saved, so just rerun
    the same command to continue."""
    kwargs = dict(
        model       = MODEL_NAME,
        messages    = [{"role": "user", "content": prompt}],
        temperature = TEMPERATURE,
        top_p       = TOP_P,
        max_tokens  = MAX_TOKENS,
    )
    if EXTRA_BODY:
        kwargs["extra_body"] = EXTRA_BODY

    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(**kwargs)
            if not getattr(response, "choices", None):
                raise RuntimeError(f"response has no choices: {response}")
            choice = response.choices[0]
            msg = choice.message

            usage = {}
            u = getattr(response, "usage", None)
            if u is not None:
                usage["prompt_tokens"] = getattr(u, "prompt_tokens", None)
                usage["completion_tokens"] = getattr(u, "completion_tokens", None)
                details = getattr(u, "completion_tokens_details", None)
                usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", None) if details else None
            reasoning_text = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            usage["has_reasoning_text"] = bool(reasoning_text)

            return msg.content or "", getattr(choice, "finish_reason", None), usage

        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in FATAL_STATUS:
                raise ApiFailure(f"HTTP {status} -- retrying will not fix this "
                                 f"(check the API key, model name, credits, or --extra-body): {e}") from e
            wait = _suggested_wait(str(e)) or min(60, 2 ** (attempt + 1))
            print(f"  API error (attempt {attempt+1}/{MAX_ATTEMPTS}): {e} -- waiting {wait:.0f}s")
            time.sleep(wait)

    raise ApiFailure(f"API failed {MAX_ATTEMPTS} times in a row: {last_err}")

# "final answer" must be followed by ':' / '-' / 'is'. Otherwise a passing
# mention mid-reasoning ("toward the final answer by ruling out D") in a
# truncated response would be read as answer D.
FINAL_LINE  = re.compile(r"final\s*answer\s*(?::|-|\bis\b:?)\s*(.*)", re.IGNORECASE)
# Case rules for the answer letter:
#  * LEAD_LETTER and BARE_LINE accept lowercase ("Final Answer: b", a line
#    that is just "(c)"). There the letter provably stands alone -- it must
#    be followed by end-of-line or punctuation -- so it cannot be part of a
#    word ("Based", "After") or the article in "A careful reading...".
#  * LONE_LETTER (searching a sentence for one standalone letter) stays
#    UPPERCASE-only. Made case-insensitive, it would count the article "a"
#    in "Based on a careful reading, the answer is C" and return None.
LEAD_LETTER = re.compile(r"^[\s\*\(\[]*(?:option\s*)?([A-D])(?=\s*(?:$|[\.\)\]\*:,;]))", re.IGNORECASE)
LONE_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
BARE_LINE   = re.compile(r"\**\s*(?:answer\s*[:\-]?\s*)?\**\s*[\(\[]?([A-D])[\)\]]?\s*\.?\s*\**", re.IGNORECASE)


def extract_answer(text):
    if not text:
        return None

    # 1) The "Final Answer" line required by the prompt. Scan from the last
    #    occurrence backwards, so a model that corrects itself is read correctly.
    for m in reversed(list(FINAL_LINE.finditer(text))):
        rest = m.group(1)
        lead = LEAD_LETTER.match(rest)              # "B", "**B**", "(B)", "Option B"
        if lead:
            return lead.group(1).upper()
        letters = set(LONE_LETTER.findall(rest))    # "Based on ..., the answer is C"
        if len(letters) == 1:
            return letters.pop()
        # zero or several standalone letters on that line -> ambiguous, keep looking

    # 2) Narrow fallback: the last line whose ENTIRE content is a single letter,
    #    e.g. "B", "**B**", "(B)", "Answer: C".
    for line in reversed([l.strip() for l in text.strip().split("\n") if l.strip()]):
        m = BARE_LINE.fullmatch(line)
        if m:
            return m.group(1).upper()

    # No valid answer marker -> None. Never guess.
    return None

def calculate_agreement(answers):
    """Return (agreement_ratio, majority_answer, tied_answers).

    A tie for first place (e.g. A A B B C, or A A B B + one unparsed) means the
    model reached NO consensus, so majority_answer is None. Counter.most_common
    would otherwise silently pick whichever answer was generated first, letting
    generation order decide the result.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0, None, []
    counts    = Counter(valid)
    top       = max(counts.values())
    winners   = sorted(a for a, c in counts.items() if c == top)
    agreement = round(top / len(valid), 3)
    if len(winners) > 1:
        return agreement, None, winners
    return agreement, winners[0], []

# --- Batch Pipeline logic ---

def _retry_write(write_fn, path):
    """Run write_fn, waiting instead of crashing if Windows has the file
    locked -- typically because it is open in Excel or being synced by
    OneDrive. Nothing is lost while waiting."""
    for attempt in range(120):
        try:
            return write_fn()
        except PermissionError:
            if attempt == 0:
                print(f"\n  Cannot write {path} -- is it open in Excel, or being "
                      f"synced by OneDrive? Close it; retrying every 5s...")
            time.sleep(5)
    return write_fn()   # last try: let the error surface


def _progress_path():
    return os.path.join(BATCH_DIR, "_progress.json")


def _load_progress():
    """Per-generation checkpoint, separate from the per-question CSV.

    Format: {question_id: {"answers": [...], "texts": [...], "finishes": [...], "usages": [...]}}
    Only questions that have NOT yet reached N_GENERATIONS appear here -- once
    complete, the entry is removed and its result lives in the batch CSV.
    """
    path = _progress_path()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(progress):
    # Write to a temp file then rename, so a process killed mid-write never
    # leaves a half-written JSON that would break the next run.
    path = _progress_path()
    tmp = path + ".tmp"

    def _write():
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(progress, f)
        os.replace(tmp, path)
    _retry_write(_write, path)


def process_batch(batch_questions, batch_id, batch_file_path):
    """Runs experiment for a single block of questions.

    Two-level resume:
      1. Across questions: questions with at least one parsed answer are
         skipped (ties included -- they are finished, just without consensus).
      2. Within a question: each generation is checkpointed to
         _progress.json as soon as it finishes. If the process dies on
         generation 3 of 5, the next run continues from generation 4 --
         instead of repeating all five and paying for the tokens twice.
    """

    if os.path.exists(batch_file_path):
        existing_df   = pd.read_csv(batch_file_path)
        # A question counts as done once it has any parsed answer. Using
        # majority_answer here would re-query (and re-bill) every tie forever.
        done_rows     = existing_df[existing_df["n_valid_parses"] > 0]
        processed_ids = set(done_rows["question_id"])
        batch_results = done_rows.to_dict("records")
        retry = len(existing_df) - len(done_rows)
        if retry:
            print(f"  Batch {batch_id}: retrying {retry} previously failed questions.")
    else:
        processed_ids = set()
        batch_results = []

    progress = _load_progress()

    for q in tqdm(batch_questions, desc=f"Processing Batch {batch_id}"):

        if q["id"] in processed_ids:
            continue

        qid_key = str(q["id"])
        prompt  = build_prompt(q)

        partial  = progress.get(qid_key, {})
        answers  = partial.get("answers", [])
        texts    = partial.get("texts", [])
        finishes = partial.get("finishes", [])
        usages   = partial.get("usages", [])
        usages  += [{}] * (len(answers) - len(usages))  # older checkpoint format
        if answers:
            print(f"  Question {q['id']}: resuming at generation {len(answers)+1} "
                  f"({len(answers)}/{N_GENERATIONS} already done in a previous session)")

        for _ in range(len(answers), N_GENERATIONS):
            text, finish, usage = call_model(prompt)   # ApiFailure -> run stops
            answers.append(extract_answer(text))
            texts.append(text or "")
            finishes.append(finish or "")
            usages.append(usage)
            # Checkpoint AFTER EVERY GENERATION, not after all five -- this is
            # what limits a crash to losing only the unsaved generation,
            # not the whole question.
            progress[qid_key] = {"answers": answers, "texts": texts,
                                 "finishes": finishes, "usages": usages}
            _save_progress(progress)
            time.sleep(random.uniform(0.5, 1.2))

        agreement, majority_answer, tied = calculate_agreement(answers)
        is_correct = (majority_answer == q["correct_answer"]) if majority_answer else False
        entropy    = calculate_entropy(answers)
        n_valid    = sum(a is not None for a in answers)
        # agreement_ratio  = top-answer votes / PARSED answers
        # agreement_over_n = top-answer votes / N (unparsed answers count as
        #                    not agreeing). This matches "consensus of the N
        #                    responses" and is the PRIMARY metric: it is what
        #                    high_agreement and "unanimous" are based on.
        n_top = max(Counter(a for a in answers if a is not None).values(), default=0)
        agreement_over_n = round(n_top / N_GENERATIONS, 3)

        def _sum(key):
            vals = [u.get(key) for u in usages if u.get(key) is not None]
            return int(sum(vals)) if vals else None

        # The row is written even if every generation failed. Silently dropped
        # questions make coverage look complete when it isn't, and the ones
        # dropped tend to be the longest questions -- a bias, not random noise.
        result_row = {
            "question_id"    : q["id"],
            "correct_answer" : q["correct_answer"],
            "all_answers"    : json.dumps(answers),
            "majority_answer": majority_answer,
            "is_tie"         : bool(tied),
            "tied_answers"   : json.dumps(tied),
            "agreement_ratio": agreement,
            "agreement_over_n": agreement_over_n,
            "is_correct"     : is_correct,
            # Based on agreement_over_n: with the parsed-only ratio, a question
            # where 1 of 5 generations parsed would count as 100% agreement.
            "high_agreement" : agreement_over_n >= 0.8,
            "reasoning_texts": json.dumps(texts),
            "prompt"         : prompt,
            "entropy"        : entropy,
            "batch_id"       : batch_id,
            "n_valid_parses" : n_valid,
            "n_truncated"    : sum(f == "length" for f in finishes),
            "finish_reasons" : json.dumps(finishes),
            "parse_failed"   : n_valid == 0,
            "option_parse_fallback": q["option_parse_fallback"],
            "prompt_tokens"    : _sum("prompt_tokens"),
            "completion_tokens": _sum("completion_tokens"),
            "reasoning_tokens" : _sum("reasoning_tokens"),
            "n_with_reasoning_text": sum(bool(u.get("has_reasoning_text")) for u in usages),
        }

        batch_results.append(result_row)
        if n_valid:
            processed_ids.add(q["id"])

        # question finished (successfully or as a total failure) -> remove it
        # from the per-generation checkpoint; its result now lives in the batch CSV
        progress.pop(qid_key, None)
        _save_progress(progress)

        # SAVE AFTER EACH QUESTION
        _retry_write(lambda: pd.DataFrame(batch_results).to_csv(batch_file_path, index=False),
                     batch_file_path)

    if batch_results:
        df_batch = pd.DataFrame(batch_results)
        _retry_write(lambda: df_batch.to_csv(batch_file_path, index=False), batch_file_path)
        ok = int(df_batch["n_valid_parses"].gt(0).sum())
        print(f"-> Batch {batch_id} saved ({len(df_batch)} rows, {ok} with a valid answer).")


def merge_all_batches():
    """Finds all batch CSVs, merges them cleanly, and handles sorting."""
    all_files = glob.glob(os.path.join(BATCH_DIR, "batch_*.csv"))
    if not all_files:
        print("No batch files found to merge!")
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} batch files. Merging...")
    df_list = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # If a question appears in more than one file (e.g. leftovers from an
    # older run with different batch boundaries), keep one -- preferring the
    # row with a valid answer.
    combined_df["_has_answer"] = combined_df["n_valid_parses"] > 0
    combined_df = (combined_df
                   .sort_values(["question_id", "_has_answer"], kind="mergesort")
                   .drop_duplicates("question_id", keep="last")
                   .drop(columns="_has_answer")
                   .reset_index(drop=True))
    return combined_df

# --- Analytics and Reporting ---

def analyze_results(df):
    """Print parsing diagnostics, then metrics. Returns the cleaned df (only
    questions with a valid answer) for plotting."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    if "n_valid_parses" in df.columns:
        total_gen = len(df) * N_GENERATIONS
        valid_gen = int(df["n_valid_parses"].sum())
        print(f"\nGenerations with valid answer: {valid_gen}/{total_gen} ({valid_gen/total_gen:.1%})")
        trunc = int(df["n_truncated"].sum()) if "n_truncated" in df.columns else 0
        print(f"Truncated generations       : {trunc}/{total_gen} ({trunc/total_gen:.1%})")
        if total_gen and trunc / total_gen > 0.05:
            print("  ^ MAX_TOKENS is still too small. Raise it before using these numbers.")
        failed = int(df["parse_failed"].sum())
        if failed:
            print(f"Questions with no answer    : {failed} (excluded from metrics)")
        if "completion_tokens" in df.columns:
            print(f"Total tokens input/output   : {int(df['prompt_tokens'].fillna(0).sum()):,} / "
                  f"{int(df['completion_tokens'].fillna(0).sum()):,}")
            rt = df["reasoning_tokens"].fillna(0).sum()
            rtx = int(df["n_with_reasoning_text"].fillna(0).sum())
            print(f"Reasoning tokens (must be 0): {int(rt):,}"
                  f"{'  <-- REASONING IS STILL ON' if rt > 0 or rtx > 0 else ''}")
        df = df[~df["parse_failed"]].copy()

    if "is_tie" in df.columns:
        n_tie = int(df["is_tie"].sum())
        print(f"Tied questions (no majority): {n_tie} (counted as incorrect; "
              f"see is_tie / tied_answers)")

    print(f"\nOverall Accuracy    : {df['is_correct'].mean():.3f}")
    print(f"Mean Agreement (over N, primary): {df['agreement_over_n'].mean():.3f}")
    print(f"Mean Agreement (over parsed)    : {df['agreement_ratio'].mean():.3f}")
    print(f"Total Questions     : {len(df)}")

    high_agree = df[df["high_agreement"]]
    low_agree  = df[~df["high_agreement"]]

    if len(high_agree) > 0:
        print(f"\n--- High Agreement (>=0.8): {len(high_agree)} questions --- Accuracy: {high_agree['is_correct'].mean():.3f}")
    if len(low_agree) > 0:
        print(f"--- Low Agreement (<0.8): {len(low_agree)} questions --- Accuracy: {low_agree['is_correct'].mean():.3f}")

    # unanimous = all N generations parsed AND identical
    unanimous = df["agreement_over_n"] == 1.0
    consistent_wrong = df[unanimous & (~df["is_correct"].astype(bool))]
    print(f"\nUnanimous but wrong: {len(consistent_wrong)} of "
          f"{int(unanimous.sum())} unanimous questions")

    return df

def plot_results(df, type_stats):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LLM Self-Consistency Analysis — LogiQA ({MODEL_NAME})",
                 fontsize=14, fontweight="bold")

    # Graph 1
    ax = axes[0, 0]
    colors = df["is_correct"].map({True: "steelblue", False: "tomato"}).fillna("gray")
    ax.scatter(df["agreement_over_n"], df["is_correct"].astype(int), c=colors, alpha=0.6, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Agreement (over N)")
    ax.set_ylabel("Correct (1) / Wrong (0)")
    ax.set_title("Agreement Ratio vs Correctness")

    # Graph 2
    ax = axes[0, 1]
    df["entropy"].hist(
        bins=10,
        ax=ax,
        edgecolor="white"
    )
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Distribution of Entropy")

    # Graph 3
    ax = axes[1, 0]
    df["agreement_over_n"].hist(bins=10, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(0.8, color="red", linestyle="--", label="Threshold (0.8)")
    ax.set_xlabel("Agreement (over N)")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Distribution of Agreement Ratios")
    ax.legend()

    # Graph 4
    ax = axes[1, 1]
    agree_acc = df.groupby("high_agreement")["is_correct"].mean()
    labels    = ["Low (<0.8)", "High (≥0.8)"]
    values    = [agree_acc.get(False, 0), agree_acc.get(True, 0)]
    ax.bar(labels, values, color=["#DD8452", "#4C72B0"], edgecolor="white")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: High vs Low Agreement")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "thesis_charts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nCharts saved -> {path}")


# --- Main Execution Runtime ---

def lock_run_config():
    """Write run_config.json to the batch folder on the first run; on later
    runs (resume) the configuration must be identical. The folder name only
    encodes model/T/N, so without this check, resuming with a different
    --max-tokens, --top-p, or --extra-body would silently mix two
    configurations into one result -- e.g. half the DeepSeek questions with
    reasoning enabled."""
    import hashlib
    cfg = {
        "model": MODEL_NAME, "base_url": ARGS.base_url,
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "n_generations": N_GENERATIONS, "max_tokens": MAX_TOKENS,
        "extra_body": EXTRA_BODY, "batch_size": BATCH_SIZE,
        "prompt_sha": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:12],
    }
    path = os.path.join(BATCH_DIR, "run_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diff = {k: (old.get(k), cfg[k]) for k in cfg if old.get(k) != cfg[k]}
        if diff:
            print("STOP: configuration differs from the existing run in this folder:")
            for k, (a, b) in diff.items():
                print(f"  {k}: previously {a!r}, now {b!r}")
            print("Use exactly the same arguments to resume, or a new --run-tag "
                  "if you are intentionally changing the configuration.")
            sys.exit(1)
    else:
        cfg["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def run_check():
    """A single API call to sanity-check the setup before a full run."""
    q = load_logiqa(0, 1)[0]
    print(f"Sample question id 0 (key: {q['correct_answer']}):")
    try:
        text, finish, usage = call_model(build_prompt(q))
    except ApiFailure as e:
        print(f"\nFAILED: API error -- {e}")
        print("Usually the API key, --base-url, --model name, or --extra-body format.")
        return
    answer = extract_answer(text)
    rtok = usage.get("reasoning_tokens")
    print(f"\nfinish_reason      : {finish}")
    print(f"prompt/output tokens: {usage.get('prompt_tokens')} / {usage.get('completion_tokens')}")
    print(f"reasoning tokens   : {rtok if rtok is not None else 'not reported by provider'}")
    print(f"reasoning text     : {'PRESENT' if usage.get('has_reasoning_text') else 'none'}")
    print(f"extracted answer   : {answer}")
    print(f"\n--- last 300 characters of the response ---\n{text[-300:]}")

    print("\n--- Verdict ---")
    problems = []
    if finish == "length":
        problems.append("finish_reason='length': the answer was cut off. Raise --max-tokens.")
    if answer is None:
        problems.append("no 'Final Answer: X' could be parsed. Inspect the response above.")
    if (rtok or 0) > 0 or usage.get("has_reasoning_text"):
        problems.append("the model is STILL doing hidden reasoning. Add "
                        "--disable-reasoning, or choose a non-thinking variant.")
    if problems:
        for pr in problems:
            print("FAILED:", pr)
    else:
        print("OK: answer parsed, not truncated, no hidden reasoning.")
        if rtok is None:
            print("    (the provider did not report reasoning_tokens; also confirm from "
                  "the model documentation that this is a non-thinking variant)")


if __name__ == "__main__":
    if ARGS.check:
        run_check()
        sys.exit(0)

    start_time = time.time()
    all_questions = load_logiqa(START_IDX, END_IDX)

    if not all_questions:
        print("ERROR: Failed to load dataset context.")
        exit(1)

    lock_run_config()
    print(f"Targeting: {len(all_questions)} questions split into chunks of {BATCH_SIZE}.\n")

    # Batch boundaries are anchored to absolute ids (0-49, 50-99, ...), NOT to
    # the position within --start/--end. So a --end 20 pilot writes to the
    # same batch_0_to_49 as the full run, and the full run simply resumes it.
    chunks = {}
    for q in all_questions:
        chunks.setdefault(q["id"] // BATCH_SIZE, []).append(q)

    try:
        for k in sorted(chunks):
            lo = k * BATCH_SIZE
            hi = min(lo + BATCH_SIZE, DATASET_SIZE) - 1
            batch_id   = f"{lo}_to_{hi}"
            batch_file = os.path.join(BATCH_DIR, f"batch_{batch_id}.csv")
            process_batch(chunks[k], batch_id, batch_file)
    except ApiFailure as e:
        print(f"\nRUN STOPPED: {e}")
        print("All completed generations are saved. Rerun the exact same "
              "command to continue.")
        sys.exit(1)

    # --- Merge Phase ---
    final_df = merge_all_batches()

    if not final_df.empty:
        _retry_write(lambda: final_df.to_csv(OUTPUT_CSV, index=False), OUTPUT_CSV)
        _retry_write(lambda: final_df.to_json(OUTPUT_JSON, orient="records", indent=2), OUTPUT_JSON)
        print(f"\nMaster compilation records written to: {OUTPUT_CSV}")
        
        clean_df = analyze_results(final_df)
        plot_results(clean_df, None)
    else:
        print("Error: Aggregation failed. No processable rows compiled.")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime metric: {elapsed/60:.2f} minutes.")