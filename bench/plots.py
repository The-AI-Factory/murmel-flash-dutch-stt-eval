"""Charts of results/results.json in The AI Factory style (warm grey, one teal element = murmel-flash, the finding as
the title, logo bottom-right) -> results/img/*.png at 300 dpi.

    python -m bench.plots
"""
from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
IMG = RES / "img"
ASSETS = ROOT / "assets"

SURFACE, INK, INK_2, GRID, ACCENT, MUTED = "#f5f4f0", "#0c1310", "#4d524f", "#e3e1da", "#009688", "#cdcac1"
ACCENT_DEEP = "#00796c"   # the larger murmel-flash model, when its results are added
for f in ("Archivo-Regular.ttf", "Archivo-Bold.ttf"):
    font_manager.fontManager.addfont(str(ASSETS / "fonts" / f))
MONO = font_manager.FontProperties(fname=str(ASSETS / "fonts" / "JetBrainsMono-Bold.ttf"))
plt.rcParams.update({
    "font.family": "Archivo", "font.size": 12.5, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})
LOGO_RECTS = [(float(x), float(y), float(w), float(h), c, float(o or 1)) for x, y, w, h, c, o in re.findall(
    r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="(#[0-9A-Fa-f]{6})"(?: opacity="([\d.]+)")?',
    (ASSETS / "aif-logo.svg").read_text())]

NAMES = {"murmel-flash-m-v5": "murmel-flash-m-v5", "murmel-flash-l-v5": "murmel-flash-l-v5", "whisper-large-v3": "Whisper large-v3",
         "whisper-large-v3-turbo": "Whisper large-v3 turbo", "parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3",
         "canary-1b-v2": "Canary 1B v2", "Voxtral-Mini-3B-2507": "Voxtral Mini 3B"}


def pct1(v: float) -> str:
    """One decimal, rounded from the two-decimal value in RESULTS.md (half up), so the chart matches the tables."""
    return f"{Decimal(f'{v:.2f}').quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def logo(fig, icon_px=28, margin_px=14, baseline=0.02):
    W, H = fig.get_size_inches() * fig.dpi
    s = icon_px / 28
    text = fig.text(1 - margin_px / W, baseline, "THE AI FACTORY", ha="right", fontproperties=MONO,
                    fontsize=12 * s * 72 / fig.dpi, color=INK)
    bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    left_px = bb.x0 - 11.2 * s - icon_px
    bottom_px = (bb.y0 + bb.y1) / 2 - icon_px / 2
    ax = fig.add_axes([left_px / W, bottom_px / H, icon_px / W, icon_px / H])
    ax.set_xlim(0, 40); ax.set_ylim(40, 0); ax.axis("off")
    for x, y, w, h, c, o in LOGO_RECTS:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, alpha=o, linewidth=0))


def frame(ax, title, subtitle, source):
    fig = ax.figure
    fig.text(0.02, 0.97, title, fontsize=17, fontweight="bold", va="top", color=INK)
    fig.text(0.02, 0.905, subtitle, fontsize=12.5, va="top", color=INK_2)
    fig.text(0.02, 0.02, source, fontsize=10, color=INK_2)
    logo(fig)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=300)
    plt.close(fig)
    print("wrote", out / name)


def load():
    table = json.loads((RES / "results.json").read_text())
    params = {"theaifactorybv/murmel-flash-l-v5": 397.0}   # extra results, not in models.txt
    for line in (ROOT / "models.txt").read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and not line.lstrip().startswith("#"):
            params[p[1]] = float(p[2])
    rows = []
    for model, v in table.items():
        short = model.split("/")[-1]
        rows.append({"model": model, "name": NAMES.get(short, short), "wer": v["avg_wer"], "rtfx": v["rtfx"],
                     "params": params.get(model), "murmel": "murmel-flash-m" in model, "flagship": "murmel-flash-l" in model, "n_sets": len(v["sets"]),
                     "gpu": v["run"].get("hardware", {}).get("gpu", "CPU")})
    return rows


def source(rows):
    gpu = sorted({r["gpu"] for r in rows})[0].replace("NVIDIA ", "")
    gpu = re.sub(r"[- ]?(SXM\d*|PCIE|PCIe)[- ]?", " ", gpu).replace("  ", " ").strip()
    return f"Source: murmel-flash-dutch-stt-eval; open Dutch test sets; batch 1 on one {gpu}"


def scatter(ax, rows, xkey, logx=True):
    for r in rows:
        ax.scatter(r[xkey], r["wer"], s=150 if r["murmel"] else 110,
                   color=ACCENT if r["murmel"] else ACCENT_DEEP if r["flagship"] else MUTED,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(r["name"], (r[xkey], r["wer"]), xytext=(9, -1), textcoords="offset points", va="center",
                    fontsize=11.5, color=INK if r["murmel"] or r["flagship"] else INK_2,
                    fontweight="bold" if r["murmel"] else "normal")
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    lo = min(r["wer"] for r in rows); hi = max(r["wer"] for r in rows)
    ax.set_ylim(max(0, lo - (hi - lo) * 0.25 - 0.5), hi + (hi - lo) * 0.2 + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))


def size_vs_wer(rows, out):
    rows = [r for r in rows if r["params"]]
    m = next(r for r in rows if r["murmel"])
    smaller = sum(r["params"] < m["params"] for r in rows if not r["murmel"])
    better = [r for r in rows if r["wer"] < m["wer"] and not r["murmel"]]
    if not smaller and not better:
        title = f"The smallest model here also makes the fewest errors"
    elif not smaller:
        ratio = min(r["params"] for r in better) / m["params"]
        title = f"Every model with fewer errors is at least {ratio:.0f}× larger than murmel-flash-m-v5"
    else:
        title = f"murmel-flash-m-v5: {m['wer']:.1f}% WER with {m['params']:.0f}M parameters"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.08, right=0.8)
    scatter(ax, rows, "params")
    ticks = [100, 300, 1000, 3000]
    ax.set_xticks(ticks, ["100M", "300M", "1B", "3B"])
    ax.set_xlim(min(r["params"] for r in rows) * 0.7, max(r["params"] for r in rows) * 1.5)
    ax.set_xlabel("Parameters (log scale)", color=INK_2)
    frame(ax, title, "Average word error rate on four open Dutch test sets against model size (lower is better)", source(rows))
    save(fig, "wer-vs-size.png", out)


def wer_bars(rows, out):
    rows = sorted(rows, key=lambda r: r["wer"])
    m = next(r for r in rows if r["murmel"])
    rank = rows.index(m) + 1
    title = ("murmel-flash-m-v5 makes the fewest errors on open Dutch test sets" if rank == 1 else
             f"{rows[0]['name']} is most accurate; murmel-flash-m-v5 ranks {rank} of {len(rows)}")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2.2))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.3, right=0.97)
    y = list(range(len(rows)))[::-1]
    vals = [r["wer"] for r in rows]
    ax.barh(y, vals, color=[ACCENT if r["murmel"] else ACCENT_DEEP if r["flagship"] else MUTED for r in rows], height=0.62,
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.15)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, pct1(v), va="center", fontsize=12, color=INK)
    frame(ax, title, "Average word error rate over FLEURS, Common Voice 17, MLS and VoxPopuli (Dutch test sets)", source(rows))
    save(fig, "wer.png", out)


def speed_vs_wer(rows, out):
    m = next(r for r in rows if r["murmel"])
    faster = [r for r in rows if r["rtfx"] > m["rtfx"] and not r["murmel"]]
    if not faster:
        title = f"murmel-flash-m-v5 is the fastest model here: {m['rtfx']:.0f}× real time"
    elif all(r["wer"] > m["wer"] for r in faster):
        title = (f"Only {faster[0]['name'].split()[0]} is faster than murmel-flash-m-v5, and it makes more errors" if len(faster) == 1
                 else "Every faster model makes more errors than murmel-flash-m-v5")
    else:
        title = f"murmel-flash-m-v5 transcribes {m['rtfx']:.0f}× faster than real time"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.08, right=0.8)
    scatter(ax, rows, "rtfx")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}×"))
    ax.set_xlim(min(r["rtfx"] for r in rows) * 0.7, max(r["rtfx"] for r in rows) * 1.6)
    ax.set_xlabel("Speed: seconds of audio per second of compute (log scale, higher is faster)", color=INK_2)
    frame(ax, title, "Average word error rate against transcription speed, one clip at a time (lower and further right is better)",
          source(rows))
    save(fig, "wer-vs-speed.png", out)


def main(out: Path = IMG):
    rows = load()
    if not any(r["murmel"] for r in rows):
        raise SystemExit("no murmel-flash results to highlight")
    size_vs_wer(rows, out)
    wer_bars(rows, out)
    speed_vs_wer(rows, out)


if __name__ == "__main__":
    main()
