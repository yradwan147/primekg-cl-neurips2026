"""Generate dataset temporal evolution figure for Paper A.

Shows the distribution of added/removed/persistent edges across entity type pairs
and the task sequence structure.

Usage:
    python scripts/generate_dataset_figure.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "papers" / "paper_a_benchmark" / "figures"

# Task entity type pairs and approximate triple counts
TASKS = [
    ("Base\n(t0)", 8_100_495, 0, 0),
    ("Disease\n(r1)", 0, 17_006, 0),
    ("Drug\n(r1)", 0, 125_340, 0),
    ("Disease\n(r2)", 0, 115_379, 0),
    ("Gene/Prot\n(r2)", 0, 2_850_590, 0),
    ("Gene/Prot\n(r3)", 0, 99_758, 0),
    ("Phenotype\n(r3)", 0, 47_994, 0),
    ("BioProcess\n(r4)", 0, 116_115, 0),
    ("Phenotype\n(r4)", 0, 57_387, 0),
    ("Anat/Path\n(r5)", 0, 2_752_672, 0),
]

TASK_NAMES = [t[0] for t in TASKS]
PERSISTENT = [t[1] for t in TASKS]
ADDED = [t[2] for t in TASKS]
REMOVED = [t[3] for t in TASKS]


def generate_temporal_evolution_figure() -> None:
    """Generate stacked bar chart showing temporal evolution per task."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(TASK_NAMES))
    width = 0.6

    # Stacked bars
    p1 = ax.bar(x, PERSISTENT, width, label="Persistent", color="#2196F3", alpha=0.8)
    p2 = ax.bar(x, ADDED, width, bottom=PERSISTENT, label="Added ($t_1$)", color="#4CAF50", alpha=0.8)
    p3 = ax.bar(x, [-r for r in REMOVED], width, label="Removed ($t_1$)", color="#F44336", alpha=0.8)

    ax.set_xlabel("Continual Learning Task (entity-type pair)", fontsize=11)
    ax.set_ylabel("Number of edges", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_NAMES, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Format y-axis with millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x/1e3:.0f}K"))

    plt.tight_layout()
    out_path = FIGURES_DIR / "temporal_evolution.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def generate_task_sequence_figure() -> None:
    """Generate task sequence diagram showing the CL evaluation structure."""
    fig, ax = plt.subplots(figsize=(10, 3.5))

    n_tasks = 10
    colors = plt.cm.Set3(np.linspace(0, 1, n_tasks))

    # Draw task blocks
    for i in range(n_tasks):
        # t0 block
        rect1 = plt.Rectangle((i * 2.2, 0.5), 0.9, 0.8,
                               facecolor=colors[i], edgecolor="black", linewidth=0.8)
        ax.add_patch(rect1)
        ax.text(i * 2.2 + 0.45, 0.9, f"$t_0$", ha="center", va="center", fontsize=7)

        # t1 block
        rect2 = plt.Rectangle((i * 2.2 + 1.0, 0.5), 0.9, 0.8,
                               facecolor=colors[i], edgecolor="black", linewidth=0.8, alpha=0.6)
        ax.add_patch(rect2)
        ax.text(i * 2.2 + 1.45, 0.9, f"$t_1$", ha="center", va="center", fontsize=7)

        # Arrow between t0 and t1
        ax.annotate("", xy=(i * 2.2 + 1.0, 0.9), xytext=(i * 2.2 + 0.9, 0.9),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))

        # Task label
        short_names = ["G-G", "G-BP", "G-MF", "G-CC", "G-D", "G-P", "D-Ph", "Dr-D", "Dr-SE", "A-G"]
        ax.text(i * 2.2 + 0.95, 0.2, short_names[i], ha="center", va="center", fontsize=7, fontweight="bold")

        # Arrow to next task
        if i < n_tasks - 1:
            ax.annotate("", xy=((i + 1) * 2.2, 0.9), xytext=(i * 2.2 + 1.9, 0.9),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Labels
    ax.text(-0.5, 0.9, "Tasks:", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text(n_tasks * 1.1, 1.5, "Sequential\ntraining", ha="center", va="center", fontsize=8, style="italic")

    ax.set_xlim(-1, n_tasks * 2.2 + 0.5)
    ax.set_ylim(-0.1, 1.8)
    ax.axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "task_sequence.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating dataset figures...")
    generate_temporal_evolution_figure()
    generate_task_sequence_figure()
    print("Done!")
