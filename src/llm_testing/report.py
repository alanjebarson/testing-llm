"""Summaries for a judged results sheet."""

from __future__ import annotations

import pandas as pd


def summarize(results: pd.DataFrame, fail_rate_threshold: float) -> dict:
    total = len(results)
    failed = int((results["Result"] == "FAIL").sum())
    fail_rate = failed / total if total else 0.0
    by_cat = (
        results.assign(passed=results["Result"].eq("PASS"))
        .groupby("Category")["passed"]
        .agg(passed="sum", total="count")
        .reset_index()
    )
    return {
        "total": total,
        "passed": total - failed,
        "failed": failed,
        "fail_rate": fail_rate,
        "rules_fallbacks": int((results.get("Judge_Source", pd.Series(dtype=str)) == "rules").sum()),
        "fine_tuning_needed": fail_rate > fail_rate_threshold,
        "by_category": by_cat,
    }


def print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Passed {s['passed']}/{s['total']}   Failed {s['failed']}   Fail rate {s['fail_rate']:.0%}")
    print(s["by_category"].to_string(index=False))
    if s["rules_fallbacks"]:
        print(f"Note: {s['rules_fallbacks']} verdict(s) came from keyword rules, not the judge model.")
    print(f"Fine-tuning needed? {'YES' if s['fine_tuning_needed'] else 'NO'}")
    print("=" * 60)
