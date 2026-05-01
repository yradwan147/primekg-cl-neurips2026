"""Aggregate results_stratified/*_seed*.json into the LaTeX table + prose
that will be inserted into Paper A §4.5.

Usage:
    python scripts/build_stratified_table.py

Writes:
    papers/.../tables/stratified_results.tex
    Also prints a prose paragraph suitable for §4.5.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path
from collections import defaultdict

RESULTS = Path("./results_stratified")
OUT_TEX = Path(
    "<HOME>/Anonymous_PrimeKG-CL_NeurIPS2026/tables/stratified_results.tex"
)

PRETTY = {
    "naive_sequential": "Naive Sequential",
    "ewc": "EWC",
    "joint_training": "Joint Training",
}


def load_all() -> dict:
    """Return {(method, decoder): [per-seed stratified dicts]}."""
    out = defaultdict(list)
    for fp in sorted(RESULTS.glob("*_seed*.json")):
        try:
            data = json.load(open(fp))
            method = data["baseline"]
            decoder = data["model"]
            for r in data["results"]:
                if "stratified" in r:
                    out[(method, decoder)].append(r["stratified"])
        except Exception:
            continue
    return out


def agg(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (st.mean(vals), st.stdev(vals))


def fmt(mu: float, sd: float) -> str:
    import math
    if math.isnan(mu):
        return "---"
    return f"${mu:.3f} \\pm {sd:.3f}$"


def build_table(all_data: dict) -> str:
    methods = ["naive_sequential", "ewc", "joint_training"]
    decoders = ["DistMult", "RotatE"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stratified filtered MRR on the final-task test set. "
        r"\textbf{Persistent} triples are in both $t_0$ and $t_1$ (a correctly "
        r"updating model should retain them); \textbf{removed} triples are in "
        r"$t_0$ but deprecated in $t_1$ (an ideal model should forget them: "
        r"lower is better); \textbf{added} triples are new in $t_1$ (mean MRR "
        r"across tasks 1--9). Values are mean $\pm$ std over 5 seeds.}",
        r"\label{tab:stratified}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Decoder} & "
        r"\textbf{Persistent $\uparrow$} & "
        r"\textbf{Removed $\downarrow$} & "
        r"\textbf{Added $\uparrow$} \\",
        r"\midrule",
    ]

    for method in methods:
        for decoder in decoders:
            runs = all_data.get((method, decoder), [])
            if not runs:
                lines.append(
                    f"{PRETTY[method]} & {decoder} & --- & --- & --- \\\\"
                )
                continue
            p_vals = [r.get("persistent", {}).get("MRR") for r in runs if r.get("persistent", {}).get("MRR") is not None]
            r_vals = [r.get("removed", {}).get("MRR") for r in runs if r.get("removed", {}).get("MRR") is not None]
            a_vals = [r.get("added", {}).get("MRR") for r in runs if r.get("added", {}).get("MRR") is not None]
            p_mu, p_sd = agg(p_vals); r_mu, r_sd = agg(r_vals); a_mu, a_sd = agg(a_vals)
            lines.append(
                f"{PRETTY[method]} & {decoder} & {fmt(p_mu, p_sd)} & "
                f"{fmt(r_mu, r_sd)} & {fmt(a_mu, a_sd)} \\\\ "
                f"% n={len(runs)} seeds"
            )
        if method != methods[-1]:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load_all()
    if not data:
        print("No stratified results yet under results_stratified/; re-run after jobs finish.")
        return

    tex = build_table(data)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex + "\n")
    print(f"Wrote {OUT_TEX}")
    print()
    print("=== cells populated ===")
    for k, runs in sorted(data.items()):
        ps = [r.get("persistent", {}).get("MRR") for r in runs if r.get("persistent", {}).get("MRR") is not None]
        rs = [r.get("removed", {}).get("MRR") for r in runs if r.get("removed", {}).get("MRR") is not None]
        ads = [r.get("added", {}).get("MRR") for r in runs if r.get("added", {}).get("MRR") is not None]
        print(f"  {k[0]:<18s} {k[1]:<8s}  n={len(runs)}  persist={st.mean(ps) if ps else '—':.4f}  removed={st.mean(rs) if rs else '—':.4f}  added={st.mean(ads) if ads else '—':.4f}" if ps and rs and ads else f"  {k[0]:<18s} {k[1]:<8s}  n={len(runs)}  (incomplete)")


if __name__ == "__main__":
    main()
