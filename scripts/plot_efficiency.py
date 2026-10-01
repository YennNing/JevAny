"""Draw docs/efficiency-latency.png from results/efficiency-a100-v1.json."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "efficiency-a100-v1.json"
OUT = ROOT / "docs" / "efficiency-latency.png"

INK, MUTED, GRID, BG = "#111111", "#555555", "#E6E6E6", "#FCFCFB"
OURS, OURS_ARROW, OURS_HALO = "#2A73D9", "#8DB6EE", "#CFE2FB"
THEIRS, THEIRS_ARROW = "#8A8A8A", "#C9C9C9"
FRONTIER = "#EE6A30"

LABELS = {
    "JevAny-Qwen3.8-27B": "JevAny-27B",
    "JevAny-Muse-Glimmer-30B": "JevAny-Muse-30B",
    "JevAny-Qwen3.5-4B": "JevAny-4B",
    "JevAny-Qwen3.5-4B-Direct-Token": "JevAny-4B-DT",
    "JevAny-Gemma-4B": "JevAny-Gemma-E4B",
    "Jev-Omni": "Jev-Omni-12B",
    "Bespoke-Nimble-9B": "Nimble-9B",
    "Open-Jev-27B-v1.1": "Open-Jev-27B",
    "Open-Jev-9B": "Open-Jev-9B",
    "Intern-Decision-4B": "Intern-4B",
    "Intern-Decision-2B": "Intern-2B",
    "Intern-Decision-0.8B": "Intern-0.8B",
    "JevK5": "JevK5-4B",
    "decider-4b": "decider-4b",
    "decider-2b": "decider-2b",
    "Laya": "Laya-0.4B",
}

# Label offsets in points from the accelerated point: (dx, dy, ha, va).
PLACEMENT = {
    "transfer": {
        "JevAny-Qwen3.8-27B": (0, 10, "center", "bottom"),
        "JevAny-Muse-Glimmer-30B": (6, -9, "left", "top"),
        "JevAny-Qwen3.5-4B": (-9, 4, "right", "bottom"),
        "JevAny-Qwen3.5-4B-Direct-Token": (-9, -4, "right", "top"),
        "JevAny-Gemma-4B": (6, 7, "left", "bottom"),
        "Jev-Omni": (0, 7, "center", "bottom"),
        "Bespoke-Nimble-9B": (8, 0, "left", "center", "before"),
        "Open-Jev-27B-v1.1": (2, -7, "left", "top"),
        "Open-Jev-9B": (2, -7, "left", "top"),
        "Intern-Decision-4B": (2, -7, "left", "top"),
        "Intern-Decision-2B": (2, -7, "left", "top"),
        "Intern-Decision-0.8B": (2, -7, "left", "top"),
        "JevK5": (-7, 0, "right", "center"),
        "decider-4b": (8, 0, "left", "center", "before"),
        "decider-2b": (2, -7, "left", "top"),
        "Laya": (7, 0, "left", "center"),
    },
    "jevbench": {
        "JevAny-Qwen3.8-27B": (0, 10, "center", "bottom"),
        "JevAny-Muse-Glimmer-30B": (4, -9, "left", "top"),
        "JevAny-Qwen3.5-4B": (-11, 0, "right", "center"),
        "JevAny-Qwen3.5-4B-Direct-Token": (-11, 0, "right", "center"),
        "JevAny-Gemma-4B": (0, -9, "center", "top"),
        "Jev-Omni": (22, -7, "center", "top"),
        "Bespoke-Nimble-9B": (8, 1, "left", "bottom", "before"),
        "Open-Jev-27B-v1.1": (0, 7, "center", "bottom"),
        "Open-Jev-9B": (0, -7, "center", "top"),
        "Intern-Decision-4B": (-7, 0, "right", "center"),
        "Intern-Decision-2B": (8, -1, "left", "top", "before"),
        "Intern-Decision-0.8B": (0, -7, "center", "top"),
        "decider-4b": (-7, 0, "right", "center"),
        "decider-2b": (-7, 0, "right", "center"),
        "Laya": (7, 0, "left", "center"),
    },
}

HIDDEN = {"jevbench": {"JevK5"}}
YLIM = {"transfer": (49.2, 89.2), "jevbench": (56.0, 91.0)}
TITLES = {"transfer": "Transfer", "jevbench": "JevBench public"}


def is_ours(model: str) -> bool:
    return model.startswith("JevAny-")


def load_pairs(suite: str) -> list[dict]:
    rows = json.loads(SOURCE.read_text())["rows"]
    pairs: dict[str, dict] = {}
    for row in rows:
        point = (row[suite]["median_ms"], row[suite]["accuracy"])
        pair = pairs.setdefault(row["model"], {"model": row["model"]})
        if row["configuration"].startswith("Default"):
            pair["before"] = point
        elif "after" not in pair or point[0] < pair["after"][0]:
            pair["after"] = point
    for pair in pairs.values():
        pair.setdefault("after", pair["before"])
        if pair["after"] == pair["before"]:
            del pair["before"]
    hidden = HIDDEN.get(suite, set())
    return [p for name, p in pairs.items() if name not in hidden]


def frontier(pairs: list[dict]) -> list[tuple[float, float]]:
    best, steps = -1.0, []
    for x, y in sorted(p["after"] for p in pairs):
        if y > best:
            steps.append((x, y))
            best = y
    return steps


def draw_panel(ax, suite: str) -> None:
    pairs = load_pairs(suite)
    steps = frontier(pairs)
    on_frontier = set(steps)

    ax.set_xscale("log")
    ax.set_xlim(7, 3500)
    ax.set_ylim(*YLIM[suite])
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000, 2000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10 ms", "20 ms", "50 ms", "100 ms", "200 ms", "500 ms", "1 s", "2 s"])
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCCCCC")
    ax.tick_params(colors=MUTED, length=0, labelsize=10, pad=6)
    ax.set_xlabel("Median latency per request (log scale)", color=MUTED, fontsize=10.5, labelpad=8)
    ax.set_ylabel("Accuracy", color=MUTED, fontsize=10.5)
    ax.set_title(TITLES[suite], loc="left", fontsize=13, fontweight="bold", color=INK, pad=14)

    xs, ys = [steps[0][0]], [steps[0][1]]
    for x, y in steps[1:]:
        xs += [x, x]
        ys += [ys[-1], y]
    xs.append(ax.get_xlim()[1])
    ys.append(ys[-1])
    ax.plot(xs, ys, color=FRONTIER, lw=2, zorder=2, solid_joinstyle="miter")

    for pair in sorted(pairs, key=lambda p: is_ours(p["model"])):
        ours = is_ours(pair["model"])
        color = OURS if ours else THEIRS
        after = pair["after"]
        if "before" in pair:
            ax.annotate(
                "", xy=after, xytext=pair["before"], zorder=3 if ours else 2.5,
                arrowprops=dict(
                    arrowstyle="-|>", color=OURS_ARROW if ours else THEIRS_ARROW,
                    lw=1.6 if ours else 1.1, mutation_scale=9 if ours else 7,
                    shrinkA=4.5, shrinkB=5.5,
                ),
            )
            ax.scatter(*pair["before"], s=46 if ours else 30, facecolor="white",
                       edgecolor=color, lw=1.8 if ours else 1.4, zorder=4)
        if ours and after in on_frontier:
            ax.scatter(*after, s=420, color=OURS_HALO, lw=0, zorder=3.5)
        ax.scatter(*after, s=60 if ours else 26, color=color, lw=0, zorder=5)

        dx, dy, ha, va, *anchor = PLACEMENT[suite][pair["model"]]
        xy = pair["before"] if anchor == ["before"] else after
        ax.annotate(
            LABELS[pair["model"]], xy=xy, xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, zorder=6,
            fontsize=10 if ours else 9, fontweight="bold" if ours else "normal",
            color=INK if ours else MUTED,
        )


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "axes.facecolor": BG})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6), dpi=200)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.775, bottom=0.19, wspace=0.13)
    for ax, suite in zip(axes, ("transfer", "jevbench")):
        draw_panel(ax, suite)

    fig.text(0.055, 0.955, "Accuracy vs measured latency, before and after acceleration",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.055, 0.922,
             "Same A100-40GB nodes, batch size 1, one request = one question; every model re-run "
             "on both suites after the change (accuracy re-measured too).",
             fontsize=10.5, color=MUTED)

    handles = [
        Line2D([], [], ls="", marker="o", ms=7, color=OURS),
        Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=OURS, mew=1.8),
        Line2D([], [], ls="", marker="o", ms=5, color=THEIRS),
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=THEIRS, mew=1.4),
        Line2D([], [], ls="", marker="o", ms=15, color=OURS_HALO),
        Line2D([], [], color=FRONTIER, lw=2),
    ]
    labels = ["JevAny, accelerated", "JevAny, before", "Third-party, after",
              "Third-party, before", "JevAny on the frontier", "Pareto frontier (after)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.048, 0.905), ncol=6,
               frameon=False, fontsize=10, labelcolor=MUTED, handletextpad=0.4, columnspacing=1.8)

    fig.text(0.055, 0.025,
             "After: flash-linear-attention + causal-conv1d installed for every model with Gated "
             "DeltaNet layers (Qwen3.5/3.8 backbones), ours and third-party alike. JevAny also runs "
             "fused SDPA attention and, for\nsingle-GPU models, CUDA graphs (one graph per padded "
             "length). decider also enables its own CUDA-graph engine. Jev-Omni (Gemma 4) and Laya "
             "(ModernBERT) do not use these kernels and were not re-run.\nJevAny-27B/30B and "
             "Open-Jev-27B stay pipeline-sharded over 2-3 GPUs (no CUDA graphs). JevAny times the "
             "forward pass only; third-party timings include each runtime's tokenization.\n"
             "JevK5 is omitted from the JevBench panel and its frontier; its numbers are in "
             "EFFICIENCY.md.",
             fontsize=8.6, color=MUTED, linespacing=1.55)

    fig.savefig(OUT, dpi=200, metadata={"Software": None})
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
