"""Draws every figure in README.md / docs/figures from the saved result files -- no model calls.

Usage: python scripts/make_figures.py [--name meta-llama-3-8b-instruct-standalone]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

OUT = Path("docs/figures")
# Fixed colour per method, everywhere (validated categorical slots; raw = neutral baseline).
COLOR = {"raw": "#8a8a85", "tree": "#2a78d6", "graph": "#eb6834", "extra": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1f2328", "#59636e", "#e4e6e9", "#fcfcfb"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "legend.frameon": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="meta-llama-3-8b-instruct-standalone")
    name = parser.parse_args().name
    OUT.mkdir(parents=True, exist_ok=True)
    style()

    tree_run = load(f"results/felm_science_auroc_{name}.json")
    graph_run = load(f"results/felm_science_graph_{name}-no_contradiction.json")
    metrics = load(f"results/felm_science_metrics_{name}.json")
    surrogate = load(f"results/felm_science_surrogate_{name}.json")
    emission = load(f"results/emission_model_felm_science_{name}.json")
    rules = load(f"results/link_rule_choice_{name}.json")

    sentences = [s for a in graph_run["answers"] for s in a["sentences"]]
    labels = [bool(s["label"]) for s in sentences]
    scores = {
        "raw": [p["raw"] for p in tree_run["predictions"]],
        "tree": [p["curve_tree"] for p in tree_run["predictions"]],
        "graph": [s["curve_graph"] for s in sentences],
    }
    names = {"raw": "Llama raw confidence", "tree": "BTProp tree", "graph": "Answer graph (ours)"}

    # 1. Headline metrics, grouped bars, direct value labels.
    keys = {"raw": "raw", "tree": "curve_tree", "graph": "curve_graph_no_contradiction"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.26
    for i, m in enumerate(("raw", "tree", "graph")):
        vals = [metrics[keys[m]][k] for k in ("auroc", "auc_pr_false", "f1_false")]
        xs = [j + (i - 1) * (width + 0.02) for j in range(3)]
        ax.bar(xs, vals, width, color=COLOR[m], label=names[m])
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":", label="Chance: AUROC 0.5")
    ax.axhline(metrics["raw"]["false_rate"], color=MUTED, lw=1, ls="--",
               label=f"Chance: AUC-PR {metrics['raw']['false_rate']:.3f}")  # fmt: skip
    ax.set_xticks(range(3), ["AUROC", "AUC-PR (false = positive)", "F1 (false = positive)"])
    ax.set_ylim(0, 1.0)
    ax.set_title("FELM-Science test half: 304 sentences, 49 false (Llama-3-8B)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    save(fig, "results_metrics.png")

    # 2. ROC and precision-recall curves.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    is_false = [not y for y in labels]
    for m in ("raw", "tree", "graph"):
        hallucination = [1 - p for p in scores[m]]
        fpr, tpr, _ = roc_curve(is_false, hallucination)
        a1.plot(
            fpr, tpr, color=COLOR[m], lw=2, label=f"{names[m]} ({metrics[keys[m]]['auroc']:.3f})"
        )
        prec, rec, _ = precision_recall_curve(is_false, hallucination)
        a2.plot(
            rec, prec, color=COLOR[m], lw=2,
            label=f"{names[m]} ({metrics[keys[m]]['auc_pr_false']:.3f})",
        )  # fmt: skip
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC (AUROC)")
    a2.axhline(metrics["raw"]["false_rate"], color=MUTED, lw=1, ls=":")
    a2.set(xlabel="Recall of false sentences", ylabel="Precision", title="Precision-recall")
    a1.legend(fontsize=9, loc="lower right")
    a2.legend(fontsize=9, loc="upper right")
    save(fig, "roc_pr_curves.png")

    # 3. Calibration: binned truth rate on the fit half and the fitted logistic curve.
    edges = [0.0, *emission["bins_sep"], 1.0]
    ct, cf = emission["counts_true"], emission["counts_false"]
    log = emission["logistic"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for b in range(len(ct)):
        n = ct[b] + cf[b]
        if n:
            mid = (edges[b] + edges[b + 1]) / 2
            ax.scatter(mid, ct[b] / n, s=30 + 3 * n, color=COLOR["tree"], alpha=0.85, zorder=3)
            ax.text(mid, ct[b] / n - 0.07, f"n={n}", ha="center", fontsize=9, color=MUTED)
    xs = [i / 200 for i in range(1, 200)]
    ys = []
    for c in xs:
        z = log["slope"] * math.log(c / (1 - c)) + log["intercept"]
        ys.append(1 / (1 + math.exp(-z)))
    ax.plot(xs, ys, color=COLOR["graph"], lw=2, label="Fitted logistic curve (Platt)")
    ax.scatter([], [], color=COLOR["tree"], label="Share true in each of the paper's 5 bins")
    ax.set(
        xlabel="Llama's P(True) for the sentence", ylabel="Share actually true",
        ylim=(0, 1.05), title="Calibration on the fit half (291 sentences)",
    )  # fmt: skip
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "calibration.png")

    # 4. Surrogate delegation curve.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for scheme, color, label in (
        ("children_only", COLOR["extra"], "Replace child nodes only"),
        ("any_node", COLOR["graph"], "Replace any node (incl. the sentence)"),
    ):
        pts = [p for p in surrogate["curve"] if p["scheme"] == scheme]
        x = [100 * p["fraction"] for p in pts]
        y = [p["curve_tree_auroc_mean"] for p in pts]
        sd = [p["curve_tree_auroc_std"] for p in pts]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=6, label=label)
        ax.fill_between(x, [a - b for a, b in zip(y, sd, strict=True)],
                        [a + b for a, b in zip(y, sd, strict=True)], color=color, alpha=0.15)  # fmt: skip
        ax.text(x[-1] + 2, y[-1], f"{y[-1]:.3f}", va="center", fontsize=9, color=INK)
    ax.axhline(surrogate["curve"][0]["curve_tree_auroc_mean"], color=MUTED, lw=1, ls=":")
    ax.set(
        xlabel="Share of Llama confidence scores replaced by the surrogate (%)",
        ylabel="Tree AUROC", xlim=(-3, 112),
        title="Extension 2: accuracy vs. LLM calls saved (mean ± std, 5 repeats)",
    )  # fmt: skip
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "surrogate_delegation.png")

    # 5. Link rules: fit-half AUROC (used to choose) per rule.
    order = ["none", "any", "strict", "no_contradiction"]
    pretty = ["No links", "All links", "Strict\ncontradiction", "No contradiction\n(chosen)"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = [rules[r]["auroc"] for r in order]
    colors = [COLOR["graph"] if r == rules["chosen"] else COLOR["raw"] for r in order]
    ax.bar(range(4), vals, 0.6, color=colors)
    for i, (v, r) in enumerate(zip(vals, order, strict=True)):
        ax.text(i, v + 0.004, f"{v:.3f}\n{rules[r]['num_links']} links", ha="center", fontsize=9)
    ax.set_xticks(range(4), pretty)
    ax.set_ylim(0.65, 0.77)
    ax.set(ylabel="AUROC on the fit half", title="Choosing the cross-link rule (fit half only)")
    save(fig, "link_rules.png")

    # 6. Progress across runs.
    first = load("results/felm_science_auroc_meta-llama-3-8b-instruct.json")["auroc"]
    steps = [
        ("First run\n(our prompts)", first["raw"], first["curve_tree"]),
        ("Paper's method +\nstand-alone sentences", tree_run["auroc"]["raw"],
         tree_run["auroc"]["curve_tree"]),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (_lbl, raw, tree) in enumerate(steps):
        ax.bar(i - 0.17, raw, 0.32, color=COLOR["raw"], label="Llama raw" if i == 0 else None)
        ax.bar(i + 0.17, tree, 0.32, color=COLOR["tree"], label="BTProp tree" if i == 0 else None)
        ax.text(i - 0.17, raw + 0.008, f"{raw:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.17, tree + 0.008, f"{tree:.3f}", ha="center", fontsize=9)
    g = metrics[keys["graph"]]["auroc"]
    ax.bar(2, g, 0.32, color=COLOR["graph"], label="Answer graph (ours)")
    ax.text(2, g + 0.008, f"{g:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(3), [s[0] for s in steps] + ["+ our extension"])
    ax.set_ylim(0.5, 0.82)
    ax.set(ylabel="AUROC (test half)", title="How the numbers moved")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "progress.png")

    # 7. Loopy BP convergence.
    iters = [r["iterations"] for r in graph_run["runs"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(iters, bins=range(0, max(iters) + 5, 4), color=COLOR["tree"], rwidth=0.85)
    looped = sum(r["has_cycle"] for r in graph_run["runs"])
    conv = sum(r["converged"] for r in graph_run["runs"])
    ax.set(
        xlabel="Message-passing rounds until settled (tolerance 1e-6, damping 0.5)",
        ylabel="Answer graphs",
        title=f"Loopy BP: {conv}/{len(iters)} runs converged ({looped} with loops)",
    )
    save(fig, "convergence.png")


if __name__ == "__main__":
    main()
