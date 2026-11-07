## PROMPT

You are labeling logical reasoning questions from the LogiQA dataset for a
research thesis.

Each item contains a passage, a question, and answer options. Instead of forcing
the item into exactly one category, estimate how strongly the item's TASK
matches EACH of six categories.

Treat each item independently. Do not let any previous item influence this one.

### What the scores mean

Each score is a TASK-MATCH SCORE: a judgment of how well the reading task
demanded by this item corresponds to that category's description.

A score is NOT a statistical probability. A score of 0.80 means the item's task
corresponds strongly to that category. It does NOT mean there is an 80 percent
chance the item's true label is that category. Do not reason about a
hypothetical true label at all. Reason about how well the demanded task fits
each category description.

### Scoring rules

- Assign a score from 0.00 to 1.00 to each of the six categories.
- The six scores must sum to 1.00. Use two decimal places.
- Do not treat the six scores as independent probabilities. They form a
  normalized representation of relative task-match across the six categories.
  A high score for one category does not require all other categories to be
  irrelevant, and a non-zero score should be assigned only when there is
  genuine task-level evidence for that category.
- Score the TASK the question demands, not the difficulty of the item and not
  whether a kind of reasoning appears somewhere in the solution process.
- A secondary category may receive a substantial score whenever the question
  genuinely supports two readings of the task, even when one reading is clearly
  stronger. Overlap is an expected outcome, not a failure to decide.
- Conversely, when the task is unmistakable, let one category dominate heavily.
  Do not spread scores to appear cautious.
- Use the QUESTION STEM as the primary basis. Use the options and passage only
  to resolve which task is being demanded.
- No individual word or phrase determines a score. Expressions such as
  "conclude," "derive," "infer," "can be concluded," "true," or "above
  statement" occur across several categories and are never sufficient on their
  own. Ask what the reader must actually DO to reach an answer.
- Do NOT score an item as Constraint-Based Reasoning merely because solving it
  involves deduction or logical reasoning.

### Categories

**1. Constraint-Based Reasoning**

Requires combining multiple explicit rules, conditions, relationships,
orderings, assignments, selections, or truth-value constraints to determine what
must, can, or cannot be true.

Typical tasks:
- determining a possible or impossible arrangement
- determining an ordering or sequence
- determining who or what must/cannot be selected
- solving assignments or matching problems
- applying multiple "if...then..." conditions
- eliminating possibilities through multiple constraints

The key characteristic is a structured constraint-solving process, usually
involving elimination or case analysis.

**2. Argument Strengthening**

Asks which statement, if true, would most strengthen, support, or increase the
credibility of an existing argument or conclusion.

A strengthener makes the conclusion more credible or better supported but is not
necessarily required for the argument to work.

**3. Argument Weakening**

Asks which statement, if true, would most weaken, undermine, challenge, or cast
doubt on an existing argument or conclusion.

The answer may provide contradictory evidence, a counterexample, an alternative
explanation, or another consideration that reduces support for the conclusion.

**4. Assumption Identification**

Asks which unstated premise, condition, or assumption the argument depends on
for its conclusion to be supported.

An assumption is something the argument requires to be true. If the assumption
is false, the connection between evidence and conclusion is substantially
damaged.

Important distinction:
- Assumption = the argument NEEDS this unstated condition.
- Strengthening = the information merely makes the argument MORE convincing,
  but the argument could still stand without it.

**5. Inference Generation**

Asks what can reasonably be concluded, inferred, or is most strongly supported
by the passage, without primarily requiring systematic constraint-solving.

If reaching an answer requires combining multiple explicit constraints to
determine what can or must be true, weight should shift toward Constraint-Based
Reasoning.

**6. Others**

Use when the question does not fit categories 1-5.

Examples:
- identifying a flaw in an argument
- resolving a paradox
- identifying the main point
- explaining a discrepancy
- identifying a reasoning error or technique
- other reasoning tasks not captured by categories 1-5

Note: a question that discusses doubt, skepticism, or credibility is not
automatically Others. If it specifically asks which statement most WEAKENS the
conclusion, that is Argument Weakening. Reserve Others for tasks categories 1-5
genuinely cannot describe.

### task_evidence

Quote verbatim the fragment of the question stem that identifies the task.

The fragment is documentation for a human reviewer, not the basis of the
decision. Quoting "can be concluded" does not by itself justify a high score for
Inference Generation, because that phrasing also appears on constraint-solving
items. Choose the fragment that best shows WHAT THE READER MUST DO, and let the
justification explain why that fragment indicates that task in this particular
item.

If no fragment of the stem is decisive on its own and the task only becomes
clear from the options or passage, quote the most task-revealing fragment
available and say so in the justification.

### Worked examples

The scores below illustrate the range of the scale. They are not targets. A
dominant category may legitimately sit anywhere from roughly 0.45 to above 0.90
depending on how cleanly the task fits.

Read Examples 3 and 5 together. Their question stems are near-identical in
wording, yet Example 3 is dominated by Constraint-Based Reasoning while Example
5 splits between Inference Generation and Constraint-Based Reasoning. The
difference comes entirely from what the passage and options require the reader
to do, not from the stem. Treat this pair as evidence that stem wording alone
never determines the scores.

**Example 1 — task unmistakable**

Passage: A company claims that switching to a four-day work week increased
productivity, since total output rose in the six months after the policy began.

Question: "Which of the following, if true, would most weaken the argument?"

```json
{
  "scores": {
    "Constraint-Based Reasoning": 0.01,
    "Argument Strengthening": 0.01,
    "Argument Weakening": 0.90,
    "Assumption Identification": 0.05,
    "Inference Generation": 0.02,
    "Others": 0.01
  },
  "primary_label": "Argument Weakening",
  "task_evidence": "which of the following, if true, would most weaken the argument",
  "justification": "The reader must supply new information that reduces support for a stated conclusion, which is the defining task of this category and is not shared by any other.",
  "notes": null
}
```

**Example 2 — task clear, one adjacent category still visible**

Passage: A city plans to reduce rush-hour congestion by expanding its subway
network, arguing that this expansion alone will resolve the delays.

Question: "Which of the following is an assumption required by the argument?"

```json
{
  "scores": {
    "Constraint-Based Reasoning": 0.01,
    "Argument Strengthening": 0.08,
    "Argument Weakening": 0.00,
    "Assumption Identification": 0.85,
    "Inference Generation": 0.05,
    "Others": 0.01
  },
  "primary_label": "Assumption Identification",
  "task_evidence": "an assumption required by the argument",
  "justification": "The reader must identify a condition the argument depends on, since the link from evidence to conclusion collapses if that condition fails; a small weight remains on Strengthening because a necessary condition also happens to support the conclusion.",
  "notes": null
}
```

**Example 3 — dominant but with a genuine secondary reading**

Passage: Office A is directly above Office B. Office C is exactly two floors
below Office A.

Question: "Based on the above statements, which of the following can be
derived?"

```json
{
  "scores": {
    "Constraint-Based Reasoning": 0.68,
    "Argument Strengthening": 0.01,
    "Argument Weakening": 0.01,
    "Assumption Identification": 0.00,
    "Inference Generation": 0.27,
    "Others": 0.03
  },
  "primary_label": "Constraint-Based Reasoning",
  "task_evidence": "which of the following can be derived",
  "justification": "Each option can only be accepted or rejected by composing the two explicit positional relations and discarding placements that violate them, which is constraint-solving; the stem itself is neutral between the two readings, so Inference Generation retains real weight.",
  "notes": null
}
```

**Example 4 — task outside categories 1-5**

Passage: The manager argues that the new policy succeeded because complaints
dropped after it was introduced.

Question: "The reasoning above is most vulnerable to criticism on the grounds
that it..."

```json
{
  "scores": {
    "Constraint-Based Reasoning": 0.00,
    "Argument Strengthening": 0.01,
    "Argument Weakening": 0.18,
    "Assumption Identification": 0.05,
    "Inference Generation": 0.04,
    "Others": 0.72
  },
  "primary_label": "Others",
  "task_evidence": "most vulnerable to criticism on the grounds that",
  "justification": "The reader must name a defect internal to the reasoning rather than introduce outside information, which distinguishes this from Argument Weakening even though both involve undermining the conclusion.",
  "notes": null
}
```

**Example 5 — boundary case; same stem wording as Example 3, different outcome**

Passage: A research team reports that among the surveyed households, every
household that owns a car also subscribes to the streaming service, and no
household without broadband owns a car. Most surveyed households have broadband.

Question: "Which of the following can be concluded from the above?"

```json
{
  "scores": {
    "Constraint-Based Reasoning": 0.42,
    "Argument Strengthening": 0.01,
    "Argument Weakening": 0.01,
    "Assumption Identification": 0.02,
    "Inference Generation": 0.46,
    "Others": 0.08
  },
  "primary_label": "Inference Generation",
  "task_evidence": "can be concluded from the above",
  "justification": "The split is driven by the options, not the stem: two options can only be settled by chaining the explicit conditionals, which is constraint-solving, while the remaining options require weighing an unquantified claim about most households, which is interpretive. The stem is worded almost identically to Example 3 yet scores differently, confirming that the wording carries no weight on its own.",
  "notes": "Boundary case between Inference Generation and Constraint-Based Reasoning; overlap arises from mixed option types."
}
```

### Procedure

1. Read the question stem and determine what the reader must do to answer.
2. Consider each of the six categories in turn and how well the task matches it.
3. Assign a score to every category. Verify the six scores sum to 1.00.
4. Set primary_label to the category with the highest score.
5. Quote the task-revealing fragment of the stem verbatim as task_evidence.
6. Write one or two sentences of justification describing the required reading
   task. If a secondary category received a substantial score, say why.
7. Use notes only when a human reviewer needs to know something not captured by
   the scores. Otherwise return null.

Do not compute margins, do not apply thresholds, and do not decide whether the
item is clear or ambiguous. That is handled downstream.

### Output format

Return ONLY a single valid JSON object. No markdown code fences, no text before
or after the JSON.

```json
{
  "scores": {
    "Constraint-Based Reasoning": 0.00,
    "Argument Strengthening": 0.00,
    "Argument Weakening": 0.00,
    "Assumption Identification": 0.00,
    "Inference Generation": 0.00,
    "Others": 0.00
  },
  "primary_label": "<one of the six category names>",
  "task_evidence": "<verbatim fragment of the question stem>",
  "justification": "<one or two sentences>",
  "notes": "<short note for the human reviewer, or null>"
}
```

primary_label must be exactly one of:
- "Constraint-Based Reasoning"
- "Argument Strengthening"
- "Argument Weakening"
- "Assumption Identification"
- "Inference Generation"
- "Others"

Passage:
{passage}

Question:
{question}

Options:
{options}
