from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BOOK = HERE / "prompts" / "topic_codebook.xlsx"

COLUMNS = ("Macro-Topic", "Code", "Label", "Prompt Label")
RESIDUAL_BLOCK = "Others"          # the workbook's own name for the residual dimension


def _read_book(path: Path = BOOK) -> pd.DataFrame:
    """The workbook, validated hard enough that a bad edit cannot reach a paid run.

    A symbol is part of the prompt, of the response schema, and of every stored
    answer, so a duplicate or a stray lowercase letter has to stop the run here
    rather than surface as a class that silently absorbs another one.
    """
    if not path.exists():
        raise SystemExit(f"topic codebook not found: {path}")
    d = pd.read_excel(path)
    d.columns = [str(c).strip() for c in d.columns]
    missing = [c for c in COLUMNS if c not in d.columns]
    if missing:
        raise SystemExit(f"{path.name}: missing column(s) {missing}. Expected "
                         f"{list(COLUMNS)}")
    d = d[list(COLUMNS)].copy()
    for c in COLUMNS:
        d[c] = d[c].map(lambda s: " ".join(str(s).split()))
    blank = d[(d == "") | (d == "nan")].any(axis=1)
    if blank.any():
        raise SystemExit(f"{path.name}: blank cell(s) in row(s) "
                         f"{[int(i) + 2 for i in d.index[blank]]} (sheet row numbers)")
    for c in ("Code", "Label", "Prompt Label"):
        if d[c].duplicated().any():
            dup = sorted(d.loc[d[c].duplicated(), c])
            raise SystemExit(f"{path.name}: duplicate {c}: {dup}")
    bad = [c for c in d["Code"] if len(c) != 3 or not (c.isascii() and c.isalpha()
                                                      and c.isupper())]
    if bad:
        raise SystemExit(f"{path.name}: Code must be three uppercase ASCII letters; "
                         f"got {bad}")
    if RESIDUAL_BLOCK not in set(d["Macro-Topic"]):
        raise SystemExit(f"{path.name}: no row in the '{RESIDUAL_BLOCK}' dimension; "
                         f"the taxonomy has no residual class")
    return d


BOOK_DF = _read_book()

# Workbook order throughout: it is the author's presentation order.
CODES = list(BOOK_DF["Code"])
LABEL = dict(zip(CODES, BOOK_DF["Label"]))
PROMPT_LABEL = dict(zip(CODES, BOOK_DF["Prompt Label"]))
BLOCK = dict(zip(CODES, BOOK_DF["Macro-Topic"]))
BY_LABEL = {lab: code for code, lab in LABEL.items()}

BLOCKS: dict[str, list[str]] = {}
for _code in CODES:
    BLOCKS.setdefault(BLOCK[_code], []).append(_code)

RESIDUAL = BLOCKS[RESIDUAL_BLOCK]
SUBSTANTIVE = [c for c in CODES if c not in RESIDUAL]

BOOK_SHA256 = hashlib.sha256(BOOK.read_bytes()).hexdigest()


def codebook_block() -> str:
    """The codebook exactly as the classifier is shown it.

    The `Code` and the `Prompt Label`, in workbook order, and nothing else.
    `Macro-Topic` groups figures and is not shown: the classifier picks a code,
    never a dimension.
    """
    return "\n".join(f"{c}  {PROMPT_LABEL[c]}" for c in CODES)


def as_frame() -> pd.DataFrame:
    """The codebook as a table, for a report or a join."""
    return BOOK_DF.rename(columns={"Macro-Topic": "macro_topic", "Code": "code",
                                   "Label": "label",
                                   "Prompt Label": "prompt_label"})


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"{len(CODES)} codes in {len(BLOCKS)} dimensions, from {BOOK.name}")
    print(f"  sha256 {BOOK_SHA256[:16]}...  "
          f"({len(SUBSTANTIVE)} substantive + {len(RESIDUAL)} residual)\n")
    for block, codes in BLOCKS.items():
        print(block.upper())
        for c in codes:
            print(f"  {c}  {LABEL[c]:<32} {PROMPT_LABEL[c]}")
        print()
