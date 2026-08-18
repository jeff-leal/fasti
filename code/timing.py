from __future__ import annotations

import csv
import os
import platform
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))

REGISTER_DIR = REPO / "reports" / "timing"
CSV_PATH = REGISTER_DIR / "pipeline_timing.csv"
MD_PATH = REGISTER_DIR / "pipeline_timing.md"

FIELDS = ["run_started", "stage", "corpus", "step", "seconds",
          "n_docs", "n_chunks", "host", "cores", "note"]

CORPUS_NAMES = {"us": "United States", "br": "Brazil", "both": "Both corpora"}

MD_HEADER = (
    "# Pipeline timing register\n"
    "\n"
    "Wall time of every stage of the pipeline, one section per run, oldest first.\n"
    "The machine-readable copy of the same rows is `pipeline_timing.csv`, which is\n"
    "the file to read when the stages are combined. Both files are append-only and\n"
    "are written by `code/timing.py`; nothing else records a run time.\n"
)


def _hms(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _num(v) -> str:
    return f"{v:,}" if isinstance(v, (int, float)) else ""


def log_run(stage, timing, scale=None, started=None, note=""):
    """Append one run of one stage to the register.  Returns the markdown block."""
    started = started or datetime.now()
    stamp = started.strftime("%Y-%m-%d %H:%M:%S")
    scale = scale or {}
    host, cores = platform.node(), os.cpu_count()

    rows = []
    for corpus, steps in timing.items():
        size = scale.get(corpus, {})
        for step, secs in steps.items():
            rows.append({"run_started": stamp, "stage": stage, "corpus": corpus,
                         "step": step, "seconds": round(float(secs), 1),
                         "n_docs": size.get("n_docs", ""),
                         "n_chunks": size.get("n_chunks", ""),
                         "host": host, "cores": cores, "note": note})

    REGISTER_DIR.mkdir(parents=True, exist_ok=True)
    fresh = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)

    block = _markdown(stage, timing, scale, stamp, host, cores, note)
    prev = (MD_PATH.read_text(encoding="utf-8").rstrip() + "\n\n"
            if MD_PATH.exists() else MD_HEADER + "\n")
    MD_PATH.write_text(prev + block, encoding="utf-8")
    return block


def _markdown(stage, timing, scale, stamp, host, cores, note):
    corpora = list(timing)
    names = [CORPUS_NAMES.get(c, c) for c in corpora]

    steps = []
    for per_corpus in timing.values():
        for step in per_corpus:
            if step not in steps:
                steps.append(step)

    total = {c: sum(timing[c].values()) for c in corpora}

    lines = [f"## {stage} · {stamp} · {host}, {cores} cores", ""]
    if note:
        lines += [note, ""]
    lines += ["| step | " + " | ".join(names) + " |",
              "|---" + "|---:" * len(corpora) + "|"]
    for step in steps:
        lines.append("| " + step + " | "
                     + " | ".join(_hms(timing[c].get(step, 0)) for c in corpora) + " |")
    lines.append("| **total** | "
                 + " | ".join(f"**{_hms(total[c])}**" for c in corpora) + " |")
    lines.append("")

    if scale:
        lines += ["| corpus | documents | chunks | seconds per 1k chunks |",
                  "|---|---:|---:|---:|"]
        for c, name in zip(corpora, names):
            size = scale.get(c, {})
            chunks = size.get("n_chunks") or 0
            rate = f"{1000 * total[c] / chunks:.1f}" if chunks else ""
            # print the chunk count only if the stage reported one; a blank cell says
            # "not a quantity this stage carries", where a 0 would read as a real count
            lines.append(f"| {name} | {_num(size.get('n_docs'))} | "
                         f"{_num(size.get('n_chunks'))} | {rate} |")
        lines.append("")

    lines += [f"Wall clock over the whole stage: **{_hms(sum(total.values()))}**.", ""]
    return "\n".join(lines)


def read_register():
    """The register as a dataframe, for combining stages."""
    import pandas as pd
    return pd.read_csv(CSV_PATH)
