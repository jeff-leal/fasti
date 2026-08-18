import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
IRT = REPO / "data" / "irt"
OUT = REPO / "paper" / "Figures"

sys.path.insert(0, str(HERE))
import paper_style  # noqa: E402
paper_style.apply(base=13)

INK, TEAL = "#222222", "#3E7C8B"
PANELS = [("lda", "LDA", "Log likelihood per token"),
          ("nmf", "NMF", "Relative Frobenius loss"),
          ("stm", "STM", "Bound per document"),
          ("dmr", "DMR", "Log likelihood per token")]
CORPORA = [("us", "US Congress Candidates", INK),
           ("br", "Brazil Mayoral Candidates", TEAL)]


def main() -> int:
    tr = pd.concat([pd.read_csv(IRT / f"topic_quality_trace_{s}.csv")
                    for s in ["lda_nmf", "dmr", "stm"]], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    for ax, (model, title, ylab) in zip(axes.ravel(), PANELS):
        ax2 = ax.twinx()
        lines = []
        for (corpus, lab, color), a in zip(CORPORA, (ax, ax2)):
            d = tr[(tr.model == model) & (tr.corpus == corpus)] \
                .sort_values("iteration")
            lines += a.plot(d.iteration, d.value, color=color, lw=1.8,
                            marker="o", ms=3.5, label=lab)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Iteration")
        ax.tick_params(axis="y", colors=INK)
        ax2.tick_params(axis="y", colors=TEAL)
        ax.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "left"]].set_visible(False)
        ax2.spines["right"].set_color(TEAL)
        if ax is axes[0, 0]:
            ax.legend(lines, [ln.get_label() for ln in lines],
                      frameon=False, loc="lower right")
    fig.tight_layout()
    out = OUT / "fig_topic_quality_convergence.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
