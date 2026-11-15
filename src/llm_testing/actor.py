"""Actor step: send each test prompt to the model under test and record its answer."""

from __future__ import annotations

import pandas as pd

from .client import ChatClient
from .config import read_text


def run_actor(cases: pd.DataFrame, client: ChatClient, settings: dict) -> pd.DataFrame:
    cfg = settings["actor"]
    system_prompt = read_text(cfg["system_prompt_file"]) if cfg["use_role_prompt"] else None

    rows = []
    total = len(cases)
    for i, case in enumerate(cases.to_dict("records"), 1):
        print(f"[{i}/{total}] {case['Test_ID']} ({case['Category']})")
        res = client.generate(cfg["model"], str(case["Prompt"]), cfg["generation"], system_prompt)
        flag = "ERROR" if not res.ok else ("TRUNCATED" if res.truncated else "ok")
        print(f"   {flag} {res.latency_s:.1f}s  {res.text[:80]!r}")
        rows.append(
            {
                **case,
                "Response": res.text if res.ok else f"ERROR: {res.error}",
                "Finish_Reason": res.finish_reason,
                "Latency_s": round(res.latency_s, 2),
            }
        )
    return pd.DataFrame(rows)
