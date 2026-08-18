"""Official paper figure style: Latin Modern Sans body (matches the Latin Modern / Computer Modern
article), math kept in Computer Modern.  Larger type + higher text-to-figure ratio for print.

Usage:
    import paper_style; paper_style.apply()
"""
import os
from pathlib import Path
import matplotlib
import matplotlib.font_manager as fm

# Latin Modern Sans .otf directory. Override with LM_FONT_DIR to reproduce the
# paper's exact typography on another machine; if the files are not found the
# rcParams fall back to TeX Gyre Heros / Arial and figures still render.
_LM = Path(os.getenv(
    "LM_FONT_DIR",
    "C:/Users/leall/AppData/Local/Programs/MiKTeX/fonts/opentype/public/lm"))
for _f in ("lmsans10-regular.otf", "lmsans10-bold.otf", "lmsans10-oblique.otf", "lmsans10-boldoblique.otf"):
    _p = _LM / _f
    if _p.exists():
        fm.fontManager.addfont(str(_p))
FONT = "Latin Modern Sans"


def apply(base=14):
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "TeX Gyre Heros", "Arial"],
        "mathtext.fontset": "cm",          # math (beta_t, alpha_t, ...) stays Computer Modern -> matches the paper
        "axes.unicode_minus": False,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.titlesize": base + 3,
        "axes.linewidth": 0.8,
        # Latin Modern Sans is an OTF with CFF (PostScript) outlines, NOT TrueType.
        # fonttype 42 mis-declares it -> "font type mismatch" -> symbols fail to render in strict
        # viewers.  fonttype 3 embeds CFF-OTF + mathtext as Type-3 glyph procedures -> renders everywhere.
        "pdf.fonttype": 3,
        "savefig.bbox": "tight",
    })
