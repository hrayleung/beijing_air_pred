#!/usr/bin/env python3
"""
Update PROJECT_REPORT.md with the latest baseline results tables.

Reads:
  - baseline/results/metrics_overall.csv
  - baseline/results/metrics_per_pollutant.csv

Writes:
  - Updates the section between:
      <!-- BASELINE_RESULTS_START -->
      <!-- BASELINE_RESULTS_END -->
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(ROOT, "PROJECT_REPORT.md")
OVERALL_CSV = os.path.join(ROOT, "baseline", "results", "metrics_overall.csv")
PER_POLLUTANT_CSV = os.path.join(ROOT, "baseline", "results", "metrics_per_pollutant.csv")

START_MARKER = "<!-- BASELINE_RESULTS_START -->"
END_MARKER = "<!-- BASELINE_RESULTS_END -->"


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _to_float(x: Optional[str]) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:  # nan
        return ""
    return f"{x:.{digits}f}"


def _build_baseline_results_md() -> str:
    if not os.path.exists(OVERALL_CSV):
        raise FileNotFoundError(f"Missing {OVERALL_CSV}. Run baselines first.")
    if not os.path.exists(PER_POLLUTANT_CSV):
        raise FileNotFoundError(f"Missing {PER_POLLUTANT_CSV}. Run baselines first.")

    overall_rows = _read_csv_rows(OVERALL_CSV)
    per_rows = _read_csv_rows(PER_POLLUTANT_CSV)

    # CO per-horizon MAE from per-pollutant csv
    co_by_model: Dict[str, Dict[str, float]] = {}
    for r in per_rows:
        if r.get("pollutant") != "CO":
            continue
        model = r.get("model", "")
        co_by_model[model] = {
            "CO_MAE": _to_float(r.get("MAE")),
            "CO_MAE_h1": _to_float(r.get("MAE_h1")),
            "CO_MAE_h6": _to_float(r.get("MAE_h6")),
            "CO_MAE_h12": _to_float(r.get("MAE_h12")),
            "CO_MAE_h24": _to_float(r.get("MAE_h24")),
        }

    # Merge into overall table
    merged = []
    for r in overall_rows:
        model = r.get("model", "")
        row = {
            "model": model,
            "MAE": _to_float(r.get("MAE")),
            "macro_MAE": _to_float(r.get("macro_MAE")),
            "MAE_h1": _to_float(r.get("MAE_h1")),
            "MAE_h6": _to_float(r.get("MAE_h6")),
            "MAE_h12": _to_float(r.get("MAE_h12")),
            "MAE_h24": _to_float(r.get("MAE_h24")),
        }
        row.update(co_by_model.get(model, {}))
        merged.append(row)

    merged.sort(key=lambda x: (x["MAE"] if x["MAE"] == x["MAE"] else 1e30))

    lines: List[str] = []
    lines.append("### 5.4 Latest baseline results (auto-generated)")
    lines.append("")
    lines.append("This section is generated from:")
    lines.append(f"- `{os.path.relpath(OVERALL_CSV, ROOT)}`")
    lines.append(f"- `{os.path.relpath(PER_POLLUTANT_CSV, ROOT)}`")
    lines.append("")
    lines.append("Regenerate after reruns with:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/update_project_report.py")
    lines.append("```")
    lines.append("")
    lines.append("**Overall + macro MAE (masked)**")
    lines.append("")
    lines.append("| model | MAE | macro_MAE | MAE_h1 | MAE_h6 | MAE_h12 | MAE_h24 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in merged:
        lines.append(
            "| {model} | {MAE} | {macro_MAE} | {MAE_h1} | {MAE_h6} | {MAE_h12} | {MAE_h24} |".format(
                model=r["model"],
                MAE=_fmt(r["MAE"]),
                macro_MAE=_fmt(r["macro_MAE"]),
                MAE_h1=_fmt(r["MAE_h1"]),
                MAE_h6=_fmt(r["MAE_h6"]),
                MAE_h12=_fmt(r["MAE_h12"]),
                MAE_h24=_fmt(r["MAE_h24"]),
            )
        )
    lines.append("")
    lines.append("**CO focus (masked MAE, raw units)**")
    lines.append("")
    lines.append("| model | CO_MAE | CO_MAE_h1 | CO_MAE_h6 | CO_MAE_h12 | CO_MAE_h24 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in merged:
        if "CO_MAE" not in r:
            continue
        lines.append(
            "| {model} | {CO_MAE} | {CO_MAE_h1} | {CO_MAE_h6} | {CO_MAE_h12} | {CO_MAE_h24} |".format(
                model=r["model"],
                CO_MAE=_fmt(r.get("CO_MAE", float("nan"))),
                CO_MAE_h1=_fmt(r.get("CO_MAE_h1", float("nan"))),
                CO_MAE_h6=_fmt(r.get("CO_MAE_h6", float("nan"))),
                CO_MAE_h12=_fmt(r.get("CO_MAE_h12", float("nan"))),
                CO_MAE_h24=_fmt(r.get("CO_MAE_h24", float("nan"))),
            )
        )

    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _replace_section(text: str, replacement: str) -> str:
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"Missing markers in {REPORT_PATH}: {START_MARKER} / {END_MARKER}")
    pre, rest = text.split(START_MARKER, 1)
    _, post = rest.split(END_MARKER, 1)
    return pre + START_MARKER + "\n\n" + replacement + "\n" + END_MARKER + post


def main() -> None:
    md = _build_baseline_results_md()
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    out = _replace_section(text, md)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Updated {REPORT_PATH}")


if __name__ == "__main__":
    main()
