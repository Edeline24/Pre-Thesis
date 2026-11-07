"""
Stage 1 (Claude Code variant): label one item per `claude -p` invocation.

Each invocation is a fresh process with a clean context, so items stay
independent. Uses the Claude Code subscription instead of API credits.

Output format is identical to the API version, so postprocess.py works
unchanged.

Requires: Claude Code installed and logged in (`claude` on PATH).

Usage:
    python run_labeling_cc.py --dataset logiqa.jsonl \
        --prompt logiqa_labeling_prompt_v4.md --limit 5     # pilot
    python run_labeling_cc.py --dataset logiqa.jsonl \
        --prompt logiqa_labeling_prompt_v4.md               # full run

Stop it any time with Ctrl-C. Rerun the same command and it picks up where
it left off.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL = "sonnet"     # "opus" burns subscription limits much faster
EFFORT = "medium"    # fixed explicitly so every item uses the same setting,
                      # rather than relying on a local default that could
                      # differ between your machine and a collaborator's
TIMEOUT = 180        # seconds per item


def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("## PROMPT")
    end = text.index("Passage:\n{passage}")
    return text[start + len("## PROMPT"):end].strip()


def load_dataset(path: Path) -> list[dict]:
    """Adjust field names to match your LogiQA file."""
    items = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append({
                "item_id": row.get("id", f"logiqa_{i:04d}"),
                "passage": row["context"],
                "question": row["question"],
                "options": "\n".join(
                    f"{chr(65 + j)}. {o}"
                    for j, o in enumerate(row["options"])
                ),
            })
    return items


def already_done(raw_path: Path) -> set[tuple[str, int]]:
    done = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("parsed"):          # only count real successes
                        done.add((rec["item_id"], rec["run_index"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def call_claude_code(prompt_text: str, model: str, effort: str) -> tuple[str, str | None]:
    """One headless invocation. Fresh context, no tools, no file access."""
    cmd = [
        "claude", "-p", prompt_text,
        "--model", model,
        "--effort", effort,
        "--output-format", "json",
        "--allowedTools", "",        # pure text task; deny all tools
        "--max-turns", "1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",           # avoid Windows' cp1252 mojibake
            errors="replace",           # never crash on a stray bad byte
            timeout=TIMEOUT,
            stdin=subprocess.DEVNULL,   # don't wait on stdin
        )
    except subprocess.TimeoutExpired:
        return "", "timeout"

    if proc.returncode != 0:
        return proc.stdout, f"exit {proc.returncode}: {proc.stderr[:400]}"

    # --output-format json wraps the reply; fall back to raw stdout
    try:
        envelope = json.loads(proc.stdout)
        return envelope.get("result", proc.stdout), None
    except json.JSONDecodeError:
        return proc.stdout, None


def extract_json(text: str):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None, "no JSON object found"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("raw_responses.jsonl"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT, choices=["low", "medium", "high"])
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="pause between calls, eases rate limiting")
    args = ap.parse_args()

    system_prompt = load_prompt(args.prompt)
    items = load_dataset(args.dataset)

    end = args.start + args.limit if args.limit else len(items)
    items = items[args.start:end]

    done = already_done(args.out)
    remaining = [
        (it, r) for it in items for r in range(args.runs)
        if (it["item_id"], r) not in done
    ]
    print(f"{len(remaining)} calls to make "
          f"({len(done)} already on disk)", file=sys.stderr)

    with args.out.open("a", encoding="utf-8") as out:
        for n, (item, run_index) in enumerate(remaining, 1):
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Passage:\n{item['passage']}\n\n"
                f"Question:\n{item['question']}\n\n"
                f"Options:\n{item['options']}"
            )

            text, call_error = call_claude_code(full_prompt, args.model, args.effort)
            if call_error:
                # the CLI's own error envelope (e.g. a 429 rate-limit message)
                # is valid JSON and would otherwise be mistaken for a label
                parsed, parse_error = None, call_error
            else:
                parsed, parse_error = extract_json(text) if text else (None, "empty")

            record = {
                "item_id": item["item_id"],
                "run_index": run_index,
                "model": f"claude-code:{args.model}",
                "effort": args.effort,
                "temperature": None,          # not exposed by the CLI
                "prompt_file": str(args.prompt),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "raw_text": text,
                "parsed": parsed,
                "parse_error": call_error or parse_error,
                "usage": None,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            status = "ok" if parsed else f"FAIL ({record['parse_error']})"
            print(f"[{n}/{len(remaining)}] {item['item_id']} {status}",
                  file=sys.stderr)

            if call_error and "exit" in str(call_error):
                # usually a rate limit; back off rather than burning the queue
                print("  backing off 60s", file=sys.stderr)
                time.sleep(60)
            else:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()