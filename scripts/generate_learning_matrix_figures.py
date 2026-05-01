"""Generate learning matrix heatmaps and per-task performance figures for Paper A.

Reads per-task results matrices from experiment JSON files and produces:
1. Learning matrix heatmaps (CGLB-style) for each method
2. Per-task peak MRR bar chart
3. AP evolution curves (AP after each task)

Usage:
    python scripts/generate_learning_matrix_figures.py
"""

import json
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




# === Configuration ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "papers" / "paper_a_benchmark" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Task short names for axis labels
TASK_SHORT_NAMES = [
    "Base\n(t0)",
    "Disease\n(r1)",
    "Drug\n(r1)",
    "Disease\n(r2)",
    "Gene/Prot\n(r2)",
    "Gene/Prot\n(r3)",
    "Phenotype\n(r3)",
    "BioProcess\n(r4)",
    "Phenotype\n(r4)",
    "Anat/Path\n(r5)",
]

# Methods and their result file patterns
METHODS = {
    "CMKL\n(MoE-DistMult)": {
        "dir": "results_run12",
        "pattern": "cmkl_DistMult_seed*.json",
        "exclude": ["sf_", "a0.", "a1.", "a2."],
    },
    "Naive Seq.\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "naive_sequential_DistMult_seed*.json",
    },
    "Joint Train.\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "joint_training_DistMult_seed*.json",
    },
    "EWC\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "ewc_DistMult_seed*.json",
    },
    "Exp. Replay\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "experience_replay_DistMult_seed*.json",
    },
    "SI\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "si_DistMult_seed*.json",
    },
    "Distillation\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "distillation_DistMult_seed*.json",
    },
    "MIR\n(DistMult)": {
        "dir": "results_run12",
        "pattern": "mir_replay_DistMult_seed*.json",
    },
    "LKGE\n(TransE)": {
        "dir": "results",
        "pattern": "lkge_TransE_seed*.json",
    },
}


def load_matrices(method_config: dict) -> list[np.ndarray]:
    """Load results_matrix arrays from all seed files for a method."""
    results_dir = PROJECT_ROOT / method_config["dir"]
    pattern = str(results_dir / method_config["pattern"])
    files = sorted(glob.glob(pattern))

    exclude = method_config.get("exclude", [])

    matrices = []
    for f in files:
        bn = os.path.basename(f)
        if any(ex in bn for ex in exclude):
            continue
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data.get("results"), list) and data["results"]:
            mat = data["results"][0].get("results_matrix")
        elif isinstance(data.get("results"), dict):
            mat = data["results"].get("results_matrix")
        else:
            mat = data.get("results_matrix")

        if mat is not None:
            matrices.append(np.array(mat))
    return matrices


def generate_learning_matrix_figure() -> None:
    """Generate multi-panel learning matrix heatmap (Figure 2 in paper)."""
    # Select 4 representative methods for the main figure
    selected = [
        "CMKL\n(MoE-DistMult)",
        "Naive Seq.\n(DistMult)",
        "EWC\n(DistMult)",
        "SI\n(DistMult)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    for idx, method_name in enumerate(selected):
        ax = axes[idx]
        matrices = load_matrices(METHODS[method_name])
        if not matrices:
            ax.set_title(f"{method_name}\n(no data)")
            continue

        # Pad to uniform size
        max_n = max(m.shape[0] for m in matrices)
        padded = []
        for m in matrices:
            if m.shape[0] < max_n:
                p = np.zeros((max_n, max_n))
                p[: m.shape[0], : m.shape[1]] = m
                padded.append(p)
            else:
                padded.append(m)
        matrices = padded

        avg_matrix = np.mean(matrices, axis=0)
        n_tasks = avg_matrix.shape[0]

        im = ax.imshow(
            avg_matrix,
            cmap="YlOrRd",
            aspect="equal",
            vmin=0,
            vmax=max(0.3, avg_matrix.max()),
            interpolation="nearest",
        )

        # Add text annotations — show ALL cells including zeros
        for i in range(n_tasks):
            for j in range(n_tasks):
                val = avg_matrix[i, j]
                if j <= i:  # only lower-triangular + diagonal are meaningful
                    color = "white" if val > 0.15 else "black"
                    txt = f"{val:.3f}" if val >= 0.0005 else "0"
                    ax.text(
                        j, i, txt,
                        ha="center", va="center",
                        fontsize=6.5, color=color,
                    )

        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Evaluated on task", fontsize=10)
        if idx == 0:
            ax.set_ylabel("After training through task", fontsize=10)

        # Sparse tick labels (1, 5, 10) but keep all minor ticks
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels(
            [str(k+1) if k+1 in [1, 5, 10] else "" for k in range(n_tasks)],
            fontsize=9,
        )
        ax.set_yticklabels(
            [str(k+1) if k+1 in [1, 5, 10] else "" for k in range(n_tasks)],
            fontsize=9,
        )

    plt.tight_layout()
    fig.colorbar(im, ax=axes, label="Filtered MRR", shrink=0.8, pad=0.02)

    out_path = FIGURES_DIR / "learning_matrix.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def generate_per_task_bar_chart() -> None:
    """Generate per-task peak MRR bar chart (Figure 3 in paper)."""
    selected = [
        "CMKL\n(MoE-DistMult)",
        "Naive Seq.\n(DistMult)",
        "EWC\n(DistMult)",
        "SI\n(DistMult)",
        "Distillation\n(DistMult)",
        "LKGE\n(TransE)",
    ]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Determine max n_tasks across all methods
    n_tasks = 0
    for method_name in selected:
        matrices = load_matrices(METHODS[method_name])
        if matrices:
            n_tasks = max(n_tasks, matrices[0].shape[0])
    if n_tasks == 0:
        print("No data for per-task bar chart")
        return

    n_methods = len(selected)
    bar_width = 0.18
    x = np.arange(n_tasks)

    for idx, method_name in enumerate(selected):
        matrices = load_matrices(METHODS[method_name])
        if not matrices:
            continue

        # Pad smaller matrices to n_tasks x n_tasks
        padded = []
        for m in matrices:
            if m.shape[0] < n_tasks:
                p = np.zeros((n_tasks, n_tasks))
                p[: m.shape[0], : m.shape[1]] = m
                padded.append(p)
            else:
                padded.append(m[:n_tasks, :n_tasks])
        matrices = padded

        avg_matrix = np.mean(matrices, axis=0)
        std_matrix = np.std(matrices, axis=0) if len(matrices) > 1 else np.zeros_like(avg_matrix)

        # Diagonal = peak per-task MRR
        diag_mean = np.diag(avg_matrix)
        diag_std = np.diag(std_matrix)

        offset = (idx - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, diag_mean, bar_width,
            yerr=diag_std, capsize=2,
            label=method_name.replace("\n", " "),
            color=colors[idx], alpha=0.85,
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Peak MRR (diagonal of learning matrix)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT_NAMES, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = FIGURES_DIR / "per_task_performance.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def generate_ap_evolution() -> None:
    """Generate AP evolution curve (AP after each task) for all methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # 9 distinct colors — one per method, no duplicates
    colors = [
        "#e41a1c",  # CMKL - red
        "#377eb8",  # Naive Seq - blue
        "#4daf4a",  # Joint Train - green
        "#984ea3",  # EWC - purple
        "#ff7f00",  # Exp. Replay - orange
        "#a65628",  # SI - brown
        "#f781bf",  # Distillation - pink
        "#17becf",  # MIR - cyan
        "#666666",  # LKGE - gray
    ]
    markers = ["o", "s", "D", "^", "v", "P", "X", "p", "h"]

    idx = 0
    for method_name, config in METHODS.items():
        if "RAG" in method_name:
            continue
        matrices = load_matrices(config)
        if not matrices:
            idx += 1
            continue
        avg_matrix = np.mean(matrices, axis=0)
        n_tasks = avg_matrix.shape[0]

        # AP after training through task i = mean of row i (columns 0..i)
        ap_evolution = []
        for i in range(n_tasks):
            ap_i = np.mean(avg_matrix[i, : i + 1])
            ap_evolution.append(ap_i)

        ax.plot(
            range(1, n_tasks + 1), ap_evolution,
            marker=markers[idx % len(markers)], markersize=6, linewidth=2,
            label=method_name.replace("\n", " "),
            color=colors[idx % len(colors)],
        )
        idx += 1

    ax.set_xlabel("After training through task $k$", fontsize=12)
    ax.set_ylabel("Average Performance (AP)", fontsize=12)
    ax.set_xticks([1, 5, 10])
    ax.set_xticklabels(["1", "5", "10"], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xticks(range(1, 11), minor=True)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = FIGURES_DIR / "ap_evolution.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating learning matrix figures for Paper A...")
    print(f"Output directory: {FIGURES_DIR}")

    generate_learning_matrix_figure()
    generate_per_task_bar_chart()
    generate_ap_evolution()

    print("\nDone! Generated figures:")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  {f.name}")
