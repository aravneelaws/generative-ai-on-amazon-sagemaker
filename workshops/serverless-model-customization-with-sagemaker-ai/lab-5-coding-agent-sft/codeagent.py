"""
Data access and prompt construction for the coding-agent SFT lab.

Like `contractnli.py` in Lab 1, this module holds only the mechanical parts: loading
the datasets and rendering the prompt. Anything that constitutes the lesson (calling a
model, executing its code, scoring it) stays in the notebooks.

Two datasets, two roles:

  TRAIN / VAL   bigcode/self-oss-instruct-sc2-exec-filter-50k
                An execution-filtered, self-generated (StarCoder2) instruction corpus,
                released under ODC-BY. We take a shuffled subset and format each row as a
                prompt/completion pair: `instruction` -> prompt, `response` -> completion.

  TEST          openai_humaneval (HumanEval, MIT)
                164 hand-written problems, each with a canonical unit-test suite. Using
                it as the held-out set makes pass@1 a real execution measurement rather
                than a string comparison, and it never overlaps the training data.

The prompt is defined once here and used by every notebook (train, evaluate, serve), so
the trained prompt and the inference prompt share one template (evaluation and serving add
only the `/no_think` switch, see NO_THINK). A second copy pasted into a notebook is how
train/serve skew gets introduced.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset

BIGCODE_DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
HUMANEVAL_DATASET = "openai_humaneval"


# ---------------------------------------------------------------- 1. the prompt

# The single template every notebook fills in. "code only, no explanation" keeps the
# output short and runnable, which is exactly what pass@1 executes and what makes the
# base-vs-fine-tuned delta visible.
INSTRUCTION = """You are a Python coding assistant. Write a correct, self-contained \
solution to the task below. Respond with Python code only, inside a single ```python \
code block, with no explanation before or after.

TASK:
{task}"""

# Qwen3 is a reasoning model: by default it opens a <think> block and, on a coding task,
# can spend the whole generation budget there before writing any code. `/no_think` at the
# end of the prompt switches that off. It goes on evaluation and serving prompts, not on
# training prompts (the BigCode completions already answer directly), the same convention
# Lab 1 uses. Both models are evaluated on the same string, so the comparison stays fair.
NO_THINK = "/no_think"


def build_prompt(task: str, no_think: bool = True) -> str:
    """Render the single-string prompt for one coding task.

    The only prompt any notebook stores. Training records pass `no_think=False`; the
    HumanEval test records and the endpoint check keep the default, so the base and
    fine-tuned models are always evaluated on byte-identical prompts.
    """
    body = INSTRUCTION.format(task=task.strip())
    return f"{body}\n\n{NO_THINK}" if no_think else body


# ---------------------------------------------------------------- 2. train / val data


def load_subset(n: int, seed: int = 42) -> Dataset:
    """The shuffled BigCode training subset: `n` rows, deterministic under `seed`."""
    ds = load_dataset(BIGCODE_DATASET, split="train")
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


def completion_for(row: Dict) -> str:
    """The target solution for one BigCode row.

    The dataset's `response` is execution-verified working code, usually with a short
    explanation and tests around it. We keep it as-is: it already carries the shape we
    want the model to learn (reasoning then a fenced solution). Trimming to only the
    fenced block is possible here if you prefer terser completions.
    """
    return row["response"].strip()


def split(ds: Dataset, val: int) -> Tuple[Dataset, Dataset]:
    """Split the BigCode subset into (train, val). Disjoint, deterministic.

    HumanEval is the test set (see `load_humaneval`), so this only carves a validation
    slice off the training subset.
    """
    val = min(val, len(ds) // 5)
    return ds.select(range(val, len(ds))), ds.select(range(val))


# ---------------------------------------------------------------- 3. test data (HumanEval)


def load_humaneval() -> Dataset:
    """The 164-problem HumanEval benchmark.

    Each row carries:
      task_id            e.g. "HumanEval/0"
      prompt             a function signature + docstring the model completes
      canonical_solution the reference body (not shown to the model)
      test               a `check(candidate)` unit-test function
      entry_point        the function name the tests call
    """
    return load_dataset(HUMANEVAL_DATASET, split="test")


def humaneval_task(row: Dict) -> str:
    """The task text shown to the model: the signature + docstring to complete.

    HumanEval prompts are themselves a function header with a docstring, which is a
    natural coding instruction, so we pass it straight through `build_prompt`.
    """
    return row["prompt"]


def humaneval_reference(row: Dict) -> str:
    """The gold `response` field for the test record.

    For HumanEval the "reference answer" that travels in the dataset is the canonical
    solution appended to the prompt. The scorer does not compare against it directly (it
    runs the unit tests instead), but the genqa format requires a `response` field.
    """
    return row["prompt"] + row["canonical_solution"]
