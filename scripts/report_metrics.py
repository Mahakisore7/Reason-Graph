"""AUROC, AUC-PR and F1 (false statement = positive) for every variant, from the saved
predictions of scripts/evaluate_felm.py and scripts/evaluate_graph.py -- no model calls.

Usage: python scripts/report_metrics.py [--name meta-llama-3-8b-instruct-standalone]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.metrics import all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="meta-llama-3-8b-instruct-standalone")
    args = parser.parse_args()

    graph = json.loads(
        Path(f"results/felm_science_graph_{args.name}.json").read_text(encoding="utf-8")
    )
    sentences = [s for answer in graph["answers"] for s in answer["sentences"]]
    groups = [a for a, answer in enumerate(graph["answers"]) for _ in answer["sentences"]]

    tree = json.loads(
        Path(f"results/felm_science_auroc_{args.name}.json").read_text(encoding="utf-8")
    )
    predictions = tree["predictions"]
    # Same test sentences in the same order -- checked, not assumed.
    assert [p["statement"] for p in predictions] == [s["statement"] for s in sentences]

    labels = [bool(s["label"]) for s in sentences]
    columns = {v: [float(p[v]) for p in predictions] for v in tree["auroc"]}
    for v in ("table_graph", "curve_graph"):
        columns[v] = [float(s[v]) for s in sentences]
    # Graph runs under other cross-link rules (D35), if present.
    for path in sorted(Path("results").glob(f"felm_science_graph_{args.name}-*.json")):
        rule = path.stem.removeprefix(f"felm_science_graph_{args.name}-")
        other = json.loads(path.read_text(encoding="utf-8"))
        rows = [s for answer in other["answers"] for s in answer["sentences"]]
        assert [r["statement"] for r in rows] == [s["statement"] for s in sentences]
        for v in ("table_graph", "curve_graph"):
            columns[f"{v}_{rule}"] = [float(r[v]) for r in rows]

    report = {v: all_metrics(labels, scores, groups) for v, scores in columns.items()}
    print(f"{len(labels)} sentences, {labels.count(False)} false, {len(set(groups))} answers")
    print(f"{'variant':<30} {'AUROC':>6} {'AUC-PR':>7} {'F1':>6}")
    for v, m in report.items():
        print(f"{v:<30} {m['auroc']:6.3f} {m['auc_pr_false']:7.3f} {m['f1_false']:6.3f}")
    print(f"(random guessing: AUC-PR = {report['raw']['false_rate']:.3f})")

    out = Path(f"results/felm_science_metrics_{args.name}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
