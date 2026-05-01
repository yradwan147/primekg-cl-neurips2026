"""Generate LaTeX results tables and figures from experiment results.

Loads JSON result files from the results/ directory and produces:
- Main results table: All methods x all CL metrics
- Ablation table: CMKL variants x all metrics
- Figures: heatmaps, forgetting curves, sensitivity plots

Usage:
    python scripts/generate_tables.py --results-dir results
    python scripts/generate_tables.py --format both
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Metrics we care about (in display order)
CL_METRICS = [
    "Average Performance (AP)",
    "Average Forgetting (AF)",
    "Backward Transfer (BWT)",
    "Forward Transfer (FWT)",
    "Remembering (REM)",
]

METHOD_DISPLAY = {
    "naive_sequential": "Naive Sequential",
    "joint_training": "Joint Training",
    "ewc": "EWC",
    "experience_replay": "Experience Replay",
    "cmkl": "CMKL (Ours)",
}


def load_result_files(results_dir: Path) -> dict[str, dict]:
    """Load all JSON result files from a directory."""
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("ablation_"):
            key = f.stem  # e.g., "ablation_struct_only"
        elif f.name.startswith("cmkl_"):
            key = "cmkl"
        else:
            # Baseline files like naive_sequential_TransE.json
            key = f.stem.rsplit("_", 1)[0]  # e.g., "naive_sequential"

        try:
            with open(f) as fh:
                results[key] = json.load(fh)
            logger.info(f"Loaded {f.name} -> key={key}")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {f.name}")
    return results


def compute_mean_std(result_list: list[dict], metric: str) -> tuple[float, float]:
    """Compute mean and std for a metric across seeds."""
    vals = [r[metric] for r in result_list if metric in r]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def generate_main_table(results: dict, fmt: str = "latex") -> str:
    """Generate the main results table (methods x metrics).

    Args:
        results: Dict of method_name -> result data.
        fmt: 'latex' or 'markdown'.

    Returns:
        Formatted table string.
    """
    methods = ["naive_sequential", "joint_training", "ewc", "experience_replay", "cmkl"]
    short_metrics = ["AP", "AF", "BWT", "FWT", "REM"]

    rows = []
    for method in methods:
        if method not in results:
            continue

        data = results[method]
        result_list = data.get("results", [])
        if not result_list:
            continue

        row = [METHOD_DISPLAY.get(method, method)]
        for metric in CL_METRICS:
            mean, std = compute_mean_std(result_list, metric)
            if len(result_list) > 1:
                row.append(f"{mean:.4f} $\\pm$ {std:.4f}" if fmt == "latex"
                           else f"{mean:.4f} +/- {std:.4f}")
            else:
                row.append(f"{mean:.4f}")
        rows.append(row)

    if fmt == "latex":
        return _format_latex_table(
            rows, ["Method"] + short_metrics,
            caption="Continual learning results on the MCGL benchmark.",
            label="tab:main_results",
        )
    else:
        return _format_markdown_table(rows, ["Method"] + short_metrics)


def generate_ablation_table(results: dict, fmt: str = "latex") -> str:
    """Generate ablation results table."""
    ablation_names = [
        ("cmkl", "CMKL (Full)"),
        ("ablation_struct_only", "Struct Only"),
        ("ablation_text_only", "Text Only"),
        ("ablation_concat_fusion", "Concat Fusion"),
        ("ablation_global_ewc", "Global EWC"),
        ("ablation_random_replay", "Random Replay"),
    ]
    short_metrics = ["AP", "AF", "BWT", "FWT", "REM"]

    rows = []
    for key, display_name in ablation_names:
        if key not in results:
            continue

        data = results[key]
        result_list = data.get("results", [])
        if not result_list:
            continue

        row = [display_name]
        for metric in CL_METRICS:
            mean, std = compute_mean_std(result_list, metric)
            if len(result_list) > 1:
                row.append(f"{mean:.4f} $\\pm$ {std:.4f}" if fmt == "latex"
                           else f"{mean:.4f} +/- {std:.4f}")
            else:
                row.append(f"{mean:.4f}")
        rows.append(row)

    if fmt == "latex":
        return _format_latex_table(
            rows, ["Variant"] + short_metrics,
            caption="Ablation study results for CMKL components.",
            label="tab:ablation_results",
        )
    else:
        return _format_markdown_table(rows, ["Variant"] + short_metrics)


def _format_latex_table(rows: list, headers: list, caption: str, label: str) -> str:
    """Format rows as a LaTeX table."""
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _format_markdown_table(rows: list, headers: list) -> str:
    """Format rows as a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_figures(results: dict, figures_dir: Path) -> list[str]:
    """Generate result figures and return list of saved paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    try:
        from src.evaluation.visualization import (
            plot_results_heatmap,
            plot_method_comparison,
            plot_forgetting_curves,
        )
    except ImportError:
        logger.warning("Visualization module not available, skipping figures")
        return saved

    # 1. Results matrix heatmaps for each method
    for method_key, data in results.items():
        if method_key.startswith("ablation_"):
            continue
        result_list = data.get("results", [])
        if not result_list or "results_matrix" not in result_list[0]:
            continue

        # Use first seed's matrix for heatmap
        R = np.array(result_list[0]["results_matrix"])
        task_names = data.get("task_names", [f"T{i}" for i in range(R.shape[1])])
        display_name = METHOD_DISPLAY.get(method_key, method_key)

        path = figures_dir / f"heatmap_{method_key}.png"
        try:
            plot_results_heatmap(R, task_names, title=f"{display_name} Results Matrix")
            import matplotlib.pyplot as plt
            plt.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(str(path))
            logger.info(f"Saved {path}")
        except Exception as e:
            logger.warning(f"Failed to generate heatmap for {method_key}: {e}")

    # 2. Method comparison bar chart
    methods_data = {}
    for method_key in ["naive_sequential", "joint_training", "ewc", "experience_replay", "cmkl"]:
        if method_key in results:
            result_list = results[method_key].get("results", [])
            if result_list:
                ap_vals = [r.get("Average Performance (AP)", 0) for r in result_list]
                methods_data[METHOD_DISPLAY.get(method_key, method_key)] = {
                    "mean": float(np.mean(ap_vals)),
                    "std": float(np.std(ap_vals)) if len(ap_vals) > 1 else 0.0,
                }

    if methods_data:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(methods_data.keys())
            means = [methods_data[n]["mean"] for n in names]
            stds = [methods_data[n]["std"] for n in names]
            bars = ax.bar(names, means, yerr=stds, capsize=5,
                          color=["#4ECDC4", "#FF6B6B", "#45B7D1", "#FFA07A", "#98D8C8"])
            ax.set_ylabel("Average Performance (AP)")
            ax.set_title("Method Comparison")
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            path = figures_dir / "method_comparison.png"
            plt.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(str(path))
            logger.info(f"Saved {path}")
        except Exception as e:
            logger.warning(f"Failed to generate method comparison: {e}")

    # 3. Buffer size sensitivity (if available)
    buf_key = "ablation_buffer_size_sweep"
    if buf_key in results and "sweep_results" in results[buf_key]:
        try:
            import matplotlib.pyplot as plt
            sweep = results[buf_key]["sweep_results"]
            buf_sizes = sorted(int(k) for k in sweep.keys())
            ap_means = []
            ap_stds = []
            for bs in buf_sizes:
                vals = [r["Average Performance (AP)"] for r in sweep[str(bs)]]
                ap_means.append(np.mean(vals))
                ap_stds.append(np.std(vals) if len(vals) > 1 else 0.0)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.errorbar(buf_sizes, ap_means, yerr=ap_stds, marker="o", capsize=4)
            ax.set_xlabel("Buffer Size")
            ax.set_ylabel("Average Performance (AP)")
            ax.set_title("Replay Buffer Size Sensitivity")
            ax.set_xscale("log")
            plt.tight_layout()
            path = figures_dir / "buffer_size_sweep.png"
            plt.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(str(path))
            logger.info(f"Saved {path}")
        except Exception as e:
            logger.warning(f"Failed to generate buffer size plot: {e}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results tables and figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument(
        "--format", choices=["latex", "markdown", "both"], default="both",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_result_files(results_dir)
    if not results:
        logger.warning(f"No result files found in {results_dir}")
        return

    logger.info(f"Loaded {len(results)} result files")

    # Generate tables
    formats = ["latex", "markdown"] if args.format == "both" else [args.format]

    for fmt in formats:
        ext = "tex" if fmt == "latex" else "md"

        # Main results table
        main_table = generate_main_table(results, fmt)
        if main_table:
            path = output_dir / f"main_results.{ext}"
            path.write_text(main_table)
            logger.info(f"Saved main results table: {path}")
            print(f"\n--- Main Results ({fmt}) ---")
            print(main_table)

        # Ablation table
        ablation_table = generate_ablation_table(results, fmt)
        if ablation_table:
            path = output_dir / f"ablation_results.{ext}"
            path.write_text(ablation_table)
            logger.info(f"Saved ablation table: {path}")
            print(f"\n--- Ablation Results ({fmt}) ---")
            print(ablation_table)

    # Generate figures
    saved_figures = generate_figures(results, figures_dir)
    if saved_figures:
        logger.info(f"Generated {len(saved_figures)} figures")
    else:
        logger.info("No figures generated (need result data)")


if __name__ == "__main__":
    main()
