"""Consistent Traditional Chinese scientific figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

def style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ("Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei") if name in available), None)
    if chinese_font is None:
        raise RuntimeError("Install Microsoft JhengHei or Noto Sans CJK TC to render Chinese figures")
    plt.rcParams.update({"font.family": [chinese_font, "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
