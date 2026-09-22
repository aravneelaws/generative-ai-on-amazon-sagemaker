"""
Custom scorer for the managed evaluation pipeline: execution-based pass@1.

This is the deterministic alternative to LLM-as-a-Judge. For each HumanEval record it
extracts the Python code the model generated, runs it together with that problem's unit
tests, and reports whether the tests passed.

It must be a single self-contained file. The pipeline uploads it and executes it in its
own container, where neither the notebook nor `codeagent.py` exists, hence the small
amount of duplication.

Emits three metrics:

    pass_at_1       1.0 if the model's code passes the problem's unit tests, else 0.0
    executes        1.0 if the code ran to completion without raising, else 0.0
    syntax_valid    1.0 if the extracted code parsed as Python, else 0.0

plus `aggregate_reward_score` = pass_at_1, the single number the pipeline ranks models by.

SECURITY: the model's code is untrusted. Each record runs in a separate subprocess with
a wall-clock timeout and no arguments, isolated from the scorer process. This is adequate
for a workshop; for a larger or repeated run, add stricter sandboxing (resource limits,
no network, a locked-down container).

FIELD MAPPING: each test record (built in notebook 1) carries the problem's unit tests
and entry point alongside the usual genqa fields, so the scorer has everything it needs
to execute:

    query          the prompt shown to the model         (genqa)
    response       the canonical solution                (genqa, not used for scoring)
    model_response the model's generated output          (what we execute)
    test           the HumanEval `check(candidate)` body
    entry_point    the function name the tests call
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

# Per-record wall-clock budget for running the generated code plus its tests.
EXEC_TIMEOUT_SECONDS = 15


def _batch(event: Any) -> List[Dict[str, Any]]:
    if isinstance(event, list):
        return event
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        parsed = json.loads(body) if isinstance(body, str) else body
        if isinstance(parsed, list):
            return parsed
    return [event] if isinstance(event, dict) else []


def _model_response(record: Dict[str, Any]) -> str:
    """The model's answer. Only fields that hold model output are read; `response`
    carries the gold solution and is deliberately excluded, reading it would execute the
    reference against its own tests and score a perfect 1.0 on every record."""
    for key in ("model_response", "generated_text", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def extract_code(text: str) -> str:
    """Pull the Python source out of a model response.

    Prefers a fenced ```python block; falls back to the largest fenced block; falls back
    to the whole text. Strips a leading <think>...</think> reasoning trace first.
    """
    body = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)

    fences = re.findall(r"```(?:python|py)?\s*(.*?)```", body, re.DOTALL)
    if fences:
        # the longest fenced block is almost always the solution, not a usage snippet
        return max(fences, key=len).strip()
    return body.strip()


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _run(program: str) -> Dict[str, bool]:
    """Run one self-contained program in an isolated subprocess.

    Returns {"executes": bool, "passed": bool}:
      passed    the unit tests ran and none failed (clean exit 0)
      executes  the candidate code loaded and the tests ran, whether they passed or not.
                A wrong answer (AssertionError) still counts as "executes"; an import or
                runtime error before/at the assertion, or a timeout, does not.

    The harness distinguishes the two by exit code: it exits 0 on pass, 1 on a test
    AssertionError (ran but wrong), and 2 on any other exception (did not execute).
    """
    harness = (
        f"{program}\n"
        "import sys as _sys\n"
        "try:\n"
        "    _CHECK()\n"
        "except AssertionError:\n"
        "    _sys.exit(1)\n"
        "except Exception:\n"
        "    _sys.exit(2)\n"
        "_sys.exit(0)\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(harness)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {"executes": False, "passed": False}
    passed = result.returncode == 0
    executes = result.returncode in (0, 1)   # ran the tests; 1 means wrong answer
    return {"executes": executes, "passed": passed}


def score_record(record: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Score one HumanEval problem: does the model's code pass the unit tests?"""
    code = extract_code(_model_response(record))
    entry_point = record.get("entry_point", "")
    test_src = record.get("test", "")

    syntax_valid = 1.0 if code and _syntax_ok(code) else 0.0

    executes = 0.0
    pass_at_1 = 0.0
    if syntax_valid and test_src and entry_point:
        # HumanEval's `test` defines `check(candidate)`. Bind the model's function to it
        # via `_CHECK`, which the harness in `_run` calls and classifies by exit code.
        program = (
            f"{code}\n\n"
            f"{test_src}\n\n"
            f"_CHECK = lambda: check({entry_point})\n"
        )
        outcome = _run(program)
        executes = 1.0 if outcome["executes"] else 0.0
        pass_at_1 = 1.0 if outcome["passed"] else 0.0

    def metric(name, value, kind="Metric"):
        return {"name": name, "value": round(float(value), 4), "type": kind}

    return {
        "id": str(record.get("id") or record.get("task_id") or index),
        "aggregate_reward_score": round(pass_at_1, 4),
        "metrics_list": [
            metric("pass_at_1", pass_at_1, "Reward"),
            metric("executes", executes),
            metric("syntax_valid", syntax_valid),
        ],
    }


def lambda_handler(event: Any, context: Any) -> Dict[str, Any]:
    try:
        results = [score_record(r, i) for i, r in enumerate(_batch(event))]
        return {"statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(results)}
    except Exception as error:  # pragma: no cover
        return {"statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(error)})}
