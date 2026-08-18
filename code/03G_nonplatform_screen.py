from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))
DATA = REPO / "data"
BR = DATA / "br"
sys.path.insert(0, str(HERE))

CORPUS = BR / "br_manifestos_all.feather"
CONFIRMED = BR / "nonplatform_br.csv"
FLAGS = BR / "nonplatform_screen_br.csv"
CACHE = BR / "nonplatform_features_br.parquet"

RUN_STARTED = datetime.now()


def _row(args):
    """Worker: one document in, one feature row out."""
    import nonplatform as NP
    return NP.features(args[0], args[1])


def load_corpus():
    a = pd.read_feather(CORPUS, columns=["platform_id", "text"])
    a["doc"] = a.platform_id.astype(str)
    return a[["doc", "text"]]


def build_features(procs):
    corp = load_corpus()
    items = list(zip(corp.doc.tolist(), corp.text.tolist()))
    n = len(items)
    t0 = time.perf_counter()
    out = []
    with Pool(processes=procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_row, items, chunksize=25), start=1):
            out.append(r)
            if k % 2000 == 0:
                el = time.perf_counter() - t0
                print(f"  {k:,}/{n:,}  {el:.0f}s  eta {el / k * (n - k):.0f}s", flush=True)
    F = pd.DataFrame(out)
    F.to_parquet(CACHE, index=False)
    print(f"[br] features cached: {len(F):,} documents, {time.perf_counter() - t0:.0f}s")
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true",
                    help="recompute the per-document feature cache")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    import nonplatform as NP

    t0 = time.perf_counter()
    if a.features or not CACHE.exists():
        F = build_features(a.procs)
    else:
        F = pd.read_parquet(CACHE)
        print(f"[br] features read from cache: {len(F):,} documents")
    t_feat = time.perf_counter() - t0

    t1 = time.perf_counter()
    fired = [sorted(NP.arms_from_features(r)) for r in F.to_dict("records")]
    F["arms"] = ["+".join(x) for x in fired]
    F["flag"] = [bool(x) for x in fired]
    t_arm = time.perf_counter() - t1

    out = F.loc[F.flag, ["doc", "arms", "n_words", "n_lines", "n_prop", "first_line"]]
    out = out.sort_values("doc")
    out.to_csv(FLAGS, index=False, encoding="utf-8")
    print(f"[br] flagged {len(out):,} of {len(F):,} documents")
    print(F.loc[F.flag, "arms"].value_counts().to_string())
    print("  arms fired, counting a document once per arm:")
    for k in sorted(NP.ARM_LABEL):
        n = sum(1 for x in fired if k in x)
        if n:
            print(f"    {k}  {NP.ARM_LABEL[k]:<38} {n:>4}")

    # Reconciliation.  The confirmed list is the exclusion the paper applies; the
    # screen must still reproduce it exactly, or a document needs reading.
    ok = True
    if CONFIRMED.exists():
        conf = pd.read_csv(CONFIRMED, dtype={"doc": str})
        flagged, confirmed = set(out.doc), set(conf.doc)
        extra = sorted(flagged - confirmed)
        missing = sorted(confirmed - flagged)
        print(f"\n[br] confirmed exclusions: {len(confirmed):,}")
        print(f"     flagged and confirmed:  {len(flagged & confirmed):,}")
        if extra:
            ok = False
            print(f"     FLAGGED BUT NOT READ:   {len(extra)}  -> read these before "
                  f"they are excluded:\n       " + "\n       ".join(extra))
        if missing:
            print(f"     confirmed, no longer flagged: {len(missing)}  (still excluded; "
                  f"the reading stands)\n       " + "\n       ".join(missing))
        if ok and not missing:
            print("     the screen reproduces the confirmed list exactly")
    else:
        print(f"\n[br] no confirmed list at {CONFIRMED.name}; nothing is excluded yet")

    try:
        from timing import log_run
        log_run("03G nonplatform screen",
                {"br": {"1 features": round(t_feat, 1), "2 arms": round(t_arm, 1)}},
                scale={"br": {"n_docs": int(len(F)), "n_flagged": int(len(out))}},
                started=RUN_STARTED)
    except Exception as exc:      # the register is a record, never a dependency
        print(f"[timing] not recorded: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
