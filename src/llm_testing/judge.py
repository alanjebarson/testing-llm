"""Judge step: an LLM grades each actor response PASS/FAIL, with a rule-based fallback."""

from __future__ import annotations

import json
import re
from string import Template

import pandas as pd

from .client import ChatClient
from .config import read_text
from .rules import evaluate_by_content

# JSON Schema for OpenAI structured outputs (strict mode needs additionalProperties: false).
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string", "enum": ["PASS", "FAIL"]},
        "reason": {"type": "string"},
    },
    "required": ["result", "reason"],
    "additionalProperties": False,
}


def parse_verdict(text: str) -> tuple[str, str] | None:
    """Pull {"result", "reason"} out of judge output. None if it can't be trusted."""
    if not text:
        return None
    clean = re.sub(r"```(?:json)?", "", text).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(clean[start : end + 1])
            result = str(data.get("result", "")).upper()
            if result in ("PASS", "FAIL"):
                return result, str(data.get("reason", "")).strip() or "No reason given"
        except json.JSONDecodeError:
            pass
    # Truncated/malformed JSON: accept only an explicit, unambiguous result field.
    m = re.search(r'"result"\s*:\s*"(PASS|FAIL)"', clean, re.IGNORECASE)
    if m:
        r = re.search(r'"reason"\s*:\s*"([^"]*)', clean)
        return m.group(1).upper(), (r.group(1) if r else "Reason not returned") + " [partial JSON]"
    return None


def run_judge(responses: pd.DataFrame, client: ChatClient, settings: dict) -> pd.DataFrame:
    cfg = settings["judge"]
    template = Template(read_text(cfg["prompt_file"]))

    rows = []
    total = len(responses)
    for i, row in enumerate(responses.to_dict("records"), 1):
        finish = str(row.get("Finish_Reason", ""))
        verdict, source = None, "llm"

        if str(row["Response"]).startswith("ERROR") or finish == "length":
            source = "rules"  # no point asking the judge about a broken answer
        else:
            prompt = template.safe_substitute(
                category=row["Category"],
                expected_behavior=row["Expected_Behavior"],
                prompt=row["Prompt"],
                response=row["Response"],
            )
            res = client.generate(cfg["model"], prompt, cfg["generation"], response_schema=VERDICT_SCHEMA)
            verdict = parse_verdict(res.text) if res.ok else None
            if verdict is None:
                source = "rules"

        if verdict is None:
            verdict = evaluate_by_content(row["Category"], row["Expected_Behavior"], row["Response"], finish)

        result, reason = verdict
        print(f"[{i}/{total}] {row['Test_ID']} {result:4} ({source}) {reason[:70]}")
        rows.append({**row, "Result": result, "Judge_Reason": reason, "Judge_Source": source})
    return pd.DataFrame(rows)
