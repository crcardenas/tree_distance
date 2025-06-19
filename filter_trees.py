#!/usr/bin/env python3
# filter_trees.py

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import Phylo
from typing import List, Tuple

def collapse_by_length(tree, tol):
    """Collapse (zero‐length) branches ≤ tol. Return (#collapsed, total_internal)."""
    collapsed = 0
    total = 0
    for cl in tree.get_nonterminals():
        bl = cl.branch_length or 0.0
        total += 1
        if bl <= tol:
            cl.branch_length = 0.0
            collapsed += 1
    pct = collapsed / total if total else 0.0
    return pct, collapsed

def collapse_by_support(tree, supp):
    """Collapse nodes with support < supp. Return (#collapsed, total_internal)."""
    collapsed = 0
    total = 0
    supports = []
    for cl in tree.get_nonterminals():
        total += 1
        support = cl.confidence if cl.confidence is not None else 0.0
        supports.append(support)
        if support < supp:
            cl.branch_length = 0.0
            collapsed += 1
    pct = collapsed / total if total else 0.0
    mean_sup = np.mean(supports) if supports else 0.0
    return pct, collapsed, mean_sup

def mean_support(tree):
    """Compute mean bootstrap/support over internal nodes."""
    supports = []
    for cl in tree.get_nonterminals():
        supports.append(cl.confidence or 0.0)
    return np.mean(supports) if supports else 0.0

def load_trees(path: Path):
    return list(Phylo.parse(str(path), "newick"))

def write_trees(trees: List, outpath: Path):
    with outpath.open("w") as f:
        Phylo.write(trees, f, "newick")

def main():
    p = argparse.ArgumentParser(
        description="Filter trees by branch‐length tol or node support supp"
    )
    p.add_argument("treefile", type=Path, help="Input Newick file (one per line)")
    p.add_argument("--keep", type=float, default=0.25,
                   help="Fraction of top trees to keep")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tol", type=float,
                       help="Collapse branches with length ≤ tol")
    group.add_argument("--supp", type=float,
                       help="Collapse nodes with support < supp")
    args = p.parse_args()

    trees = load_trees(args.treefile)
    stats = []

    if args.tol is not None:
        csv_name = "tol_polytomy_stats.csv"
        for idx, tr in enumerate(trees):
            pct, col = collapse_by_length(tr, args.tol)
            stats.append((idx, pct, col))
        # sort ascending pct_null (fewest collapses = most resolved)
        stats_df = pd.DataFrame(stats, columns=["tree_index", "pct_null", "collapsed"])
        stats_df.sort_values("pct_null", inplace=True)
    elif args.supp is not None:
        csv_name = "supp_polytomy_stats.csv"
        for idx, tr in enumerate(trees):
            pct, col, mean_sup = collapse_by_support(tr, args.supp)
            stats.append((idx, pct, col, mean_sup))
        # sort descending mean_sup (highest support first)
        stats_df = pd.DataFrame(stats, columns=["tree_index", "pct_low_support", "collapsed", "mean_support"])
        stats_df.sort_values("mean_support", ascending=False, inplace=True)
    else:
        # neither tol nor supp: sort by mean support
        csv_name = "mean_support_stats.csv"
        for idx, tr in enumerate(trees):
            m = mean_support(tr)
            stats.append((idx, m))
        stats_df = pd.DataFrame(stats, columns=["tree_index", "mean_support"])
        stats_df.sort_values("mean_support", ascending=False, inplace=True)

    # write stats
    stats_df.to_csv(csv_name, index=False)
    print(f"[SAVE] {csv_name}")

    # select top keep fraction
    n_keep = max(1, int(len(trees) * args.keep))
    top_idxs = stats_df["tree_index"].iloc[:n_keep].tolist()
    filtered = [trees[i] for i in top_idxs]

    write_trees(filtered, Path("filtered_trees.treefile"))
    print(f"[SAVE] filtered_trees.treefile ({n_keep} of {len(trees)})")

if __name__ == "__main__":
    main()
