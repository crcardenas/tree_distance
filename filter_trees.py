#!/usr/bin/env python3
import argparse
import re
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from Bio import Phylo
import subprocess

# Precompile once
SUPPORT_RE = re.compile(r"\)(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)(?=[:),])")

def collapse_by_length(tree, tol):
    collapsed = total = 0
    for clade in tree.get_nonterminals():
        total += 1
        if (clade.branch_length or 0.0) <= tol:
            clade.branch_length = 0.0
            collapsed += 1
    pct = collapsed / total if total else 0.0
    return pct, collapsed

def mean_support(tree):
    vals = [cl.confidence or 0.0 for cl in tree.get_nonterminals()]
    return float(np.mean(vals)) if vals else 0.0

def mean_dual_support_from_string(nw: str):
    m = SUPPORT_RE.findall(nw)
    if not m:
        return 0.0, 0.0
    a = np.fromiter((float(x) for x, _ in m), dtype=float)
    b = np.fromiter((float(y) for _, y in m), dtype=float)
    return float(a.mean()), float(b.mean())

def run_pxcolt(treefile: Path, threshold: float, out_file: Path):
    cmd = ["pxcolt", "-t", str(treefile), "-l", str(threshold), "-o", str(out_file)]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_pxrr(treefile: Path, outgroups: str, out_file: Path):
    cmd = ["pxrr", "-t", str(treefile), "-r", "-g", outgroups, "-o", str(out_file)]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def load_trees(path: Path):
    return list(Phylo.parse(str(path), "newick"))

def load_newick_lines(path: Path):
    # One tree per line, as in your pipeline
    with open(path, "r") as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def write_trees(trees: List, outpath: Path):
    with outpath.open("w") as f:
        Phylo.write(trees, f, "newick")

def main():
    parser = argparse.ArgumentParser(
        description="Filter and reroot trees with optional collapse by length (--tol) or support (--supp)."
    )
    parser.add_argument("-t", "--treefile", type=Path, required=True, help="Input Newick file (one tree per line)")
    parser.add_argument("--keep", type=float, default=0.25, help="Fraction of top trees to retain")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for output files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tol", nargs="?", const=1e-6, type=float, default=None,
                       help="Collapse branches with length ≤ tol (default 1e-6 if flag provided without value)")
    group.add_argument("--supp", type=int, help="Collapse nodes with support < supp%% via pxcolt")
    parser.add_argument("--ranked_og", required=True, help="Comma-separated ranked outgroups for rerooting")
    parser.add_argument("--shalrt", action="store_true",
                        help="Parse dual supports (SH-aLRT/Bootstrap) and output both")
    args = parser.parse_args()

    prefix = (args.prefix + "_") if args.prefix else ""
    temp_dir = Path(tempfile.mkdtemp(prefix="filter_tmp_"))
    collapse_input = args.treefile

    # 1) Collapse
    if args.tol is not None:
        print(f"[INFO] Collapsing branches ≤ {args.tol}")
        trees = load_trees(collapse_input)
        stats = []
        for idx, tr in enumerate(trees):
            pct, col = collapse_by_length(tr, args.tol)
            stats.append((idx, pct, col))
        df = pd.DataFrame(stats, columns=["tree_index", "pct_null", "collapsed"])
        df.sort_values("pct_null", inplace=True)
        df.to_csv(f"{prefix}tol_polytomy_stats.csv", index=False)
        print(f"[SAVE] {prefix}tol_polytomy_stats.csv")
        collapse_input = temp_dir / "tol_collapsed.treefile"
        write_trees(trees, collapse_input)

    elif args.supp is not None:
        print(f"[INFO] Collapsing nodes with support < {args.supp}%% via pxcolt")
        thr = args.supp / 100.0
        collapse_output = temp_dir / "supp_collapsed.treefile"
        run_pxcolt(args.treefile, thr, collapse_output)
        trees = load_trees(collapse_output)

        if args.shalrt:
            lines = load_newick_lines(collapse_output)
            stats = [(idx,)+mean_dual_support_from_string(nw) for idx, nw in enumerate(lines)]
            df = pd.DataFrame(stats, columns=["tree_index", "mean_shalrt", "mean_bootstrap"])
        else:
            stats = [(idx, mean_support(tr)) for idx, tr in enumerate(trees)]
            df = pd.DataFrame(stats, columns=["tree_index", "mean_support"])

        df.sort_values(df.columns[-1], ascending=False, inplace=True)
        df.to_csv(f"{prefix}supp_polytomy_stats.csv", index=False)
        print(f"[SAVE] {prefix}supp_polytomy_stats.csv")
        collapse_input = collapse_output

    else:
        print("[INFO] No collapse—using mean support for filtering")
        trees = load_trees(collapse_input)
        if args.shalrt:
            lines = load_newick_lines(collapse_input)
            stats = [(idx,)+mean_dual_support_from_string(nw) for idx, nw in enumerate(lines)]
            df = pd.DataFrame(stats, columns=["tree_index", "mean_shalrt", "mean_bootstrap"])
        else:
            stats = [(idx, mean_support(tr)) for idx, tr in enumerate(trees)]
            df = pd.DataFrame(stats, columns=["tree_index", "mean_support"])
        # keep prior behavior of writing the summary
        out_csv = f"{prefix}mean_support_stats.csv"
        df.sort_values(df.columns[-1], ascending=False, inplace=True)
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

    # 2) Filtering with keep==1 preserving order
    if args.keep >= 1.0:
        print("[INFO] --keep 1.00 detected: preserving input order")
        filtered_file = collapse_input
    else:
        trees = load_trees(collapse_input)
        n_keep = max(1, int(len(trees) * args.keep))
        top_idxs = df["tree_index"].iloc[:n_keep].tolist()
        filtered = [trees[i] for i in top_idxs]
        filtered_file = temp_dir / "filtered.treefile"
        write_trees(filtered, filtered_file)
        print(f"[INFO] Filtered top {args.keep*100:.1f}% trees")

    # 3) Reroot
    print(f"[INFO] Rerooting with outgroups: {args.ranked_og}")
    final_out = Path(f"{prefix}filtered_trees.rr.treefile")
    run_pxrr(filtered_file, args.ranked_og, final_out)
    print(f"[SAVE] {final_out}")

if __name__ == "__main__":
    main()
