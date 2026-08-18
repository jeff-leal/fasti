from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console defaults to cp1252
except Exception:
    pass

# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
CMP = REPO / "data" / "cmp_stance_dim"

SAMPLE_CSV = PROC / "stratified_stance_sample.csv"
RUN_TAG = "full_gpt41_20260629"
MERGED_CSV = CMP / f"stance_dim_merged_{RUN_TAG}.csv"
LABELS_CSV = CMP / f"stance_dim_labels_{RUN_TAG}.csv"
CHECKPOINT_JSONL = CMP / f"stance_dim_checkpoint_{RUN_TAG}.jsonl"

OUT_FRAME = PROC / "stance_nli_training_frame.csv"
OUT_CONC = PROC / "stance_nli_training_concordant.csv"
OUT_REPORT = PROC / "stance_nli_training_report.md"

# --------------------------------------------------------------------------- #
DIM_LABELS = {"D01": "Economy", "D02": "Society"}
STANCE_LABELS = {"S01": "State", "S02": "Market", "S03": "Progressive",
                 "S04": "Conservative", "S05": "Neutral"}
ST_TO_LRN = {"S01": "Left", "S03": "Left", "S02": "Right", "S04": "Right", "S05": "Neutral"}
GPT_STANCE_TO_LRN = {"State": "Left", "Progressive": "Left",
                     "Market": "Right", "Conservative": "Right", "Neutral": "Neutral"}
ECON_POLE = {"Left": "State", "Right": "Market", "Neutral": "Neutral"}
SOC_POLE = {"Left": "Progressive", "Right": "Conservative", "Neutral": "Neutral"}
LRN_ORDER = ["Left", "Right", "Neutral"]

_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


# --------------------------------------------------------------------------- #
def load_gpt_full() -> tuple[pd.DataFrame, str]:
    """Return [sid, gpt_dimension, gpt_stance, gpt_lrn] + source name."""
    if MERGED_CSV.exists():
        m = pd.read_csv(MERGED_CSV, dtype={"sid": str})
        if {"gpt_dimension", "gpt_stance"}.issubset(m.columns) and m["gpt_stance"].notna().any():
            out = m.loc[m["gpt_stance"].notna(), ["sid", "gpt_dimension", "gpt_stance"]].copy()
            out["gpt_lrn"] = out["gpt_stance"].map(GPT_STANCE_TO_LRN)
            return out.dropna(subset=["gpt_lrn"]), MERGED_CSV.name

    if LABELS_CSV.exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"id": str}).rename(columns={"id": "sid"})
        lab["gpt_dimension"] = lab["dim"].map(DIM_LABELS)
        lab["gpt_stance"] = lab["st"].map(STANCE_LABELS)
        lab["gpt_lrn"] = lab["st"].map(ST_TO_LRN)
        return lab.dropna(subset=["gpt_lrn"])[["sid", "gpt_dimension", "gpt_stance", "gpt_lrn"]], LABELS_CSV.name

    if CHECKPOINT_JSONL.exists():
        recs: dict[str, dict] = {}
        with CHECKPOINT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("schema_version") != "stance_dimension_v02":
                    continue
                for c in rec.get("codes", []):
                    recs[str(c["id"])] = {"gpt_dimension": DIM_LABELS.get(c.get("dim")),
                                          "gpt_stance": STANCE_LABELS.get(c.get("st")),
                                          "gpt_lrn": ST_TO_LRN.get(c.get("st"))}
        out = pd.DataFrame([{"sid": k, **v} for k, v in recs.items()])
        return out.dropna(subset=["gpt_lrn"]), CHECKPOINT_JSONL.name + "  (LIVE CHECKPOINT — incomplete)"

    raise FileNotFoundError(f"No GPT predictions. Looked for:\n  {MERGED_CSV}\n  {LABELS_CSV}\n  {CHECKPOINT_JSONL}")


def balance(s: pd.Series, order, title: str) -> None:
    s = s.dropna()
    n = len(s)
    vc = s.value_counts().reindex(order).fillna(0).astype(int)
    _log(f"\n{title}  (n={n:,})")
    for k in order:
        c = int(vc[k])
        _log(f"  {k:<14} {c:>7,}  ({(100*c/n if n else 0):5.1f}%)")
    if n and vc.min() > 0:
        _log(f"  imbalance (max/min) = {vc.max()/vc.min():.2f}x")


# --------------------------------------------------------------------------- #
def main() -> int:
    _log("# NLI training frame — multi-task concordance filter\n")

    samp = pd.read_csv(SAMPLE_CSV, dtype={"sid": str})
    # The sample carries empty placeholder prediction columns; drop them so the
    # real GPT predictions land in clean names (no _x/_y suffix collision).
    samp = samp.drop(columns=["gpt_dimension", "gpt_stance", "gpt_lrn"], errors="ignore")
    _log(f"Sample rows                  : {len(samp):,}")

    gpt, src = load_gpt_full()
    gpt = gpt.drop_duplicates(subset="sid", keep="last")
    _log(f"GPT predictions source       : {src}")
    _log(f"GPT-labelled sids            : {len(gpt):,}")

    df = samp.merge(gpt, on="sid", how="inner")
    _log(f"Joined (sample & gpt)        : {len(df):,}")

    # Premise text: require non-empty original `text`. text_en kept as-is (may be empty).
    df["text"] = df["text"].astype("string")
    df["text_en"] = df.get("text_en", pd.Series(index=df.index, dtype="string")).astype("string")
    before = len(df)
    df = df[df["text"].str.strip().fillna("").ne("")].copy()
    if before - len(df):
        _log(f"Dropped empty-premise rows   : {before - len(df):,}")
    n_en = int(df["text_en"].str.strip().fillna("").ne("").sum())
    _log(f"Rows with English MT (text_en): {n_en:,}  ({100*n_en/len(df):.1f}%)")

    # Coder agreement flags.
    df["gold_lrn"] = df["stance"]
    df["stance_conc"] = df["gold_lrn"] == df["gpt_lrn"]
    df["topic_conc"] = df["dimension"] == df["gpt_dimension"]
    df["full_conc"] = df["stance_conc"] & df["topic_conc"]

    # ---- Stance concordance report (T4) ----------------------------------- #
    n_join = len(df)
    n_stance = int(df["stance_conc"].sum())
    _log(f"\n## T4 stance concordance (gpt L/R/N == gold L/R/N)")
    _log(f"Concordant                   : {n_stance:,}  ({100*n_stance/n_join:.1f}%)")
    _log(f"Dropped disagreements        : {n_join-n_stance:,}  ({100*(n_join-n_stance)/n_join:.1f}%)")
    conf = (pd.crosstab(df["gold_lrn"], df["gpt_lrn"])
              .reindex(index=LRN_ORDER, columns=LRN_ORDER).fillna(0).astype(int))
    _log("\nGold (rows) x GPT (cols) — L/R/N:")
    _log(conf.to_string())
    _log(f"\nTopic concordance (Economy/Society) : {int(df['topic_conc'].sum()):,} "
         f"({100*df['topic_conc'].mean():.1f}%)")
    _log(f"Full concordance (both axes)        : {int(df['full_conc'].sum()):,} "
         f"({100*df['full_conc'].mean():.1f}%)")

    # ---- Per-task labels --------------------------------------------------- #
    df["topic_label"] = np.where(df["topic_conc"], df["dimension"], np.nan)
    df["gen_label"] = np.where(df["stance_conc"], df["gold_lrn"], np.nan)
    econ_mask = df["full_conc"] & (df["dimension"] == "Economy")
    soc_mask = df["full_conc"] & (df["dimension"] == "Society")
    df["econ_label"] = np.where(econ_mask, df["gold_lrn"].map(ECON_POLE), np.nan)
    df["soc_label"] = np.where(soc_mask, df["gold_lrn"].map(SOC_POLE), np.nan)

    # ---- Per-task balances ------------------------------------------------- #
    _log("\n## Per-task training labels (clean, agreed)")
    balance(df["topic_label"], ["Economy", "Society"], "T1 topic")
    balance(df["econ_label"], ["State", "Market", "Neutral"], "T2 econ pole")
    balance(df["soc_label"], ["Progressive", "Conservative", "Neutral"], "T3 soc pole")
    balance(df["gen_label"], LRN_ORDER, "T4 general stance (PRODUCTION)")

    # ---- Projected hypothesis-pair counts (orig / EN / both) --------------- #
    # Pairs per labelled sentence: binary task -> 2 (1 entail + 1 not) ; trinary -> 3.
    has_en = df["text_en"].str.strip().fillna("").ne("")
    _log("\n## Projected hypothesis–sentence pairs  (premises = original + English MT)")
    _log(f"{'task':<16}{'sents':>8}{'pairs/sent':>11}{'orig pairs':>12}{'EN pairs':>11}{'TOTAL':>11}")
    tot_o = tot_e = 0
    for task, col, pps in [("T1 topic", "topic_label", 2), ("T2 econ", "econ_label", 3),
                           ("T3 soc", "soc_label", 3), ("T4 general", "gen_label", 3)]:
        m = df[col].notna()
        s = int(m.sum())
        po = s * pps                       # original-language pairs
        pe = int((m & has_en).sum()) * pps  # English-MT pairs (only where text_en present)
        tot_o += po; tot_e += pe
        _log(f"{task:<16}{s:>8,}{pps:>11}{po:>12,}{pe:>11,}{po+pe:>11,}")
    _log(f"{'ALL TASKS':<16}{'':>8}{'':>11}{tot_o:>12,}{tot_e:>11,}{tot_o+tot_e:>11,}")
    _log("  (original-language pairs, English-translation pairs, and the BOTH/total.")
    _log("   Final counts after the leakage-safe split + optional oversampling are")
    _log("   printed by the training notebook.)")

    # ---- Write frames ------------------------------------------------------ #
    frame_cols = ["sid", "language", "text", "text_en",
                  "topic_label", "econ_label", "soc_label", "gen_label",
                  "dimension", "gpt_dimension", "gold_lrn", "gpt_lrn", "gpt_stance",
                  "manifesto_id", "party", "countryname", "cmp_code"]
    frame_cols = [c for c in frame_cols if c in df.columns]
    keep = df["topic_conc"] | df["stance_conc"]   # any task usable
    frame = df.loc[keep, frame_cols].reset_index(drop=True)
    frame.to_csv(OUT_FRAME, index=False, encoding="utf-8")
    _log(f"\nSaved multi-task frame -> {OUT_FRAME}  ({len(frame):,} rows)")

    # Back-compat: the T4 stance-concordant subset with a plain `label`.
    conc = df.loc[df["stance_conc"]].copy()
    conc["label"] = conc["gold_lrn"]
    conc_cols = [c for c in ["sid", "language", "text", "text_en", "label",
                             "manifesto_id", "party", "countryname", "cmp_code",
                             "dimension", "gpt_stance", "gold_lrn"] if c in conc.columns]
    conc[conc_cols].to_csv(OUT_CONC, index=False, encoding="utf-8")
    _log(f"Saved T4 concordant    -> {OUT_CONC}  ({len(conc):,} rows)")

    OUT_REPORT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"Saved report           -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
