"""Create Tab10 acquisition variants of v9; posterior always stays Tab10 blue.

Run: .venv/bin/python plots/bo_acquisition_colors.py
Uses the saved snapshot without rerunning BO. Includes a comparison PNG.
"""
from pathlib import Path
import os
import subprocess

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "acquisition_colors"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from pypdf import PdfReader

def draw(snapshot: dict, width_pt: float, height_pt: float, accent_color, output_stem: str) -> None:
    """Use TeX points (72.27/in), with fixed canvas size and vector PDF text."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.55,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    blue = plt.get_cmap("tab10").colors[0]
    orange = accent_color
    ink = "black"
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(width_pt / 72.27, height_pt / 72.27),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.14, top=0.94, hspace=0.37)
    t, mu, std = snapshot["theta"], snapshot["mean"], snapshot["std"]
    next_theta, next_ei = float(snapshot["next_theta"]), float(snapshot["next_ei"])
    guides = {}
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("black")
        ax.tick_params(colors=ink)
        ax.set_xlim(t[0], t[-1])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        guides[ax] = ax.axvline(next_theta, ymax=1.0 if ax is top else 0.96,
                               color=orange, lw=1.2, ls=(0, (3, 2)), zorder=1)

    top.fill_between(t, mu - 2 * std, mu + 2 * std, color=blue, alpha=0.17, lw=0)
    top.plot(t, mu, color=blue, lw=1.15)
    top.scatter(snapshot["observed_theta"], snapshot["observed_cost"],
                color=ink, s=14, marker="x", linewidths=0.85, zorder=4,
                clip_on=False)
    top.set_ylabel(r"Cost $J(\theta)$")
    top.set_title("(a) Gaussian process posterior", loc="left", fontsize=9, pad=5)
    top.tick_params(labelbottom=False)
    lo, hi = top.get_ylim()
    top.set_ylim(lo, hi + 0.20 * (hi - lo))

    bottom.fill_between(t, 0, snapshot["ei"], color=orange, alpha=0.12, lw=0)
    bottom.plot(t, snapshot["ei"], color=orange, lw=1.15)
    bottom.plot(next_theta, next_ei, marker="*", ms=8, color=orange,
                markeredgecolor=ink, markeredgewidth=0.4, zorder=5)
    bottom.set_ylim(0, max(next_ei, float(np.max(snapshot["ei"]))) * 1.75)
    bottom.set_xlabel(r"Controller gain $\theta$")
    bottom.set_ylabel(r"$\alpha_n(\theta)=\mathrm{EI}_n(\theta)$", fontsize=9, labelpad=5)
    bottom.set_title("(b) Acquisition function", loc="left", fontsize=9, pad=5)
    bottom.set_xticks([-10, -8, -6, -4, -2])
    bottom.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    # Hide numeric y labels and their scientific-notation offset on both panels.
    for ax in (top, bottom):
        ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.get_offset_text().set_visible(False)
    # Place the callout on the side opposite the selected point.
    text_x = 0.03 if next_theta > (t[0] + t[-1]) / 2 else 0.97
    bottom.annotate(
        r"$\theta_{n+1}=\arg\max_{\theta}\,\alpha_n(\theta)$",
        xy=(next_theta, next_ei), xycoords="data",
        xytext=(text_x, 0.84), textcoords="axes fraction",
        ha="left" if text_x < 0.5 else "right", va="center", fontsize=9,
        arrowprops={"arrowstyle": "->", "color": ink, "lw": 0.65,
                    "connectionstyle": "arc3,rad=-0.15", "shrinkB": 5},
    )
    legend = top.legend(handles=[
        Line2D([], [], color=blue, lw=1.15, label="Mean"),
        Patch(facecolor=blue, alpha=0.17, edgecolor="none", label=r"$\pm 2\sigma$"),
        Line2D([], [], color=ink, marker="x", lw=0, ms=4, label="Observations"),
    ], loc="upper left", ncol=3, borderaxespad=0.35,
        frameon=False, handlelength=1.3, columnspacing=1.2, handletextpad=0.4)

    # Pack the artists into the fixed-width page, rather than cropping the PDF.
    # Keep 2.4 pt outer padding while increasing all text to 9 pt.
    fig.tight_layout(pad=2.4 / 9, h_pad=0.8)
    fig.align_ylabels((top, bottom))
    fig.canvas.draw()
    # Use the rendered legend extent so the guide stays clear after layout.
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    gap_pixels = 4.0 * fig.dpi / 72
    stop_y = top.transAxes.inverted().transform(
        (legend_box.x0, legend_box.y0 - gap_pixels)
    )[1]
    guides[top].set_ydata([0, max(0.0, min(1.0, stop_y))])
    fig.canvas.draw()
    assert legend_box.y0 - top.transAxes.transform((0, stop_y))[1] >= gap_pixels - 1e-6
    from matplotlib.text import Text
    visible_text = [artist for artist in fig.findobj(Text)
                    if artist.get_visible() and artist.get_text()]
    assert all(artist.get_fontsize() == 9 for artist in visible_text)
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    page_width, page_height = fig.get_size_inches()
    margins = {
        "left_pdf_pt": float(tight.x0 * 72),
        "right_pdf_pt": float((page_width - tight.x1) * 72),
        "bottom_pdf_pt": float(tight.y0 * 72),
        "top_pdf_pt": float((page_height - tight.y1) * 72),
        "axes_width_pdf_pt": float(top.get_position().width * page_width * 72),
        "posterior_to_acquisition_height_ratio": float(top.get_position().height / bottom.get_position().height),
        "guide_linewidth_pdf_pt": 1.2,
        "guide_to_legend_gap_pdf_pt": 4.0,
        "bottom_guide_height_fraction": 0.96,
    }
    assert min(v for k, v in margins.items() if k.endswith("_pdf_pt") and k != "axes_width_pdf_pt") > 0

    for transparent, suffix in [(False, "white"), (True, "transparent")]:
        # Do not use bbox_inches='tight': it changes the requested page width.
        folder = OUTPUT / ("transparent" if transparent else "non_transparent")
        fig.savefig(folder / f"{output_stem}_{suffix}.pdf", transparent=transparent,
                    facecolor="none" if transparent else "white",
                    metadata={"Title": "Bayesian optimization for controller tuning",
                              "Subject": f"Acquisition color comparison: {output_stem}"})
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    for folder in ("transparent", "non_transparent"):
        (OUTPUT / folder).mkdir(exist_ok=True)
    qa = OUTPUT / "qa"
    qa.mkdir(exist_ok=True)
    with np.load(HERE / "bo_snapshot.npz") as cached:
        snapshot = dict(cached)
    names = ["orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    colors = plt.get_cmap("tab10").colors[1:]
    records = []
    for name, color in zip(names, colors):
        stem = f"bo_acquisition_{name}"
        draw(snapshot, 219, 200, color, stem)
        for background in ["white", "transparent"]:
            folder = OUTPUT / ("transparent" if background == "transparent" else "non_transparent")
            pdf = folder / f"{stem}_{background}.pdf"
            page = PdfReader(pdf).pages[0]
            assert abs(float(page.mediabox.width) - 219 * 72 / 72.27) < 1e-6
            assert abs(float(page.mediabox.height) - 200 * 72 / 72.27) < 1e-6
            text = page.extract_text()
            assert "Gaussian process posterior" in text and "arg max" in text
            command = ["pdftocairo", "-scale-to", "900", "-singlefile", "-png"]
            if background == "transparent":
                command.append("-transp")
            subprocess.run(command + [str(pdf), str(qa / f"{stem}_{background}")], check=True)
            if background == "transparent":
                im = Image.open(qa / f"{stem}_{background}.png")
                assert im.mode == "RGBA" and im.getpixel((0, 0))[3] == 0
        records.append({"color": name, "hex": matplotlib.colors.to_hex(color), "stem": stem})
        print(f"Created and checked {name}", flush=True)

    # Contact sheets use PDF renders so they show the actual exported layout.
    for background in ["white", "transparent"]:
        fig, axes = plt.subplots(3, 3, figsize=(12, 11.7))
        for ax, record in zip(axes.flat, records):
            im = Image.open(qa / f"{record['stem']}_{background}.png").convert("RGBA")
            if background == "transparent":
                backing = Image.new("RGBA", im.size, "#f0f0f0")
                im = Image.alpha_composite(backing, im)
            ax.imshow(im)
            ax.set_axis_off()
            ax.set_title(record["color"].capitalize() + "  " + record["hex"],
                         fontsize=13, color="black", pad=8)
        fig.subplots_adjust(left=0.012, right=0.988, bottom=0.01, top=0.925,
                            wspace=0.10, hspace=0.28)
        fig.suptitle("Acquisition colors - posterior fixed to Tab10 blue", fontsize=16, y=0.985)
        target = OUTPUT / "comparison.png" if background == "white" else qa / "comparison_transparent.png"
        fig.savefig(target, dpi=170, facecolor="white")
        plt.close(fig)
    rows = ["# Acquisition color variants", "", "All variants preserve version 9: 219 x 200 TeX pt, 9 pt main text,",
            "2:1 panel heights, black axes, and a blue Gaussian process posterior.",
            "The acquisition curve, shading, star, and dashed guides share the selected Tab10 color.",
            "", "![Comparison](comparison.png)", "", "| Color | Hex | White PDF | Transparent PDF |",
            "|---|---|---|---|"]
    for r in records:
        rows.append(f"| {r['color'].capitalize()} | {r['hex']} | [White](non_transparent/{r['stem']}_white.pdf) | [Transparent](transparent/{r['stem']}_transparent.pdf) |")
    rows.extend(["", "Regenerate from the repository root:", "", "```sh", 
                 ".venv/bin/python plots/bo_acquisition_colors.py", "```", "",
                 "Uses the cached snapshot; earlier versions are untouched. All files remain local and Git-ignored."])
    (OUTPUT / "README.md").write_text("\n".join(rows) + "\n")
    print(f"Saved 18 PDFs and comparison.png in {OUTPUT}")


if __name__ == "__main__":
    main()
