#!/usr/bin/env python3
# filter_trees.py
# Cody Raul Cardenas - 20250618

from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
from Bio import Phylo
from typing import List, Tuple

def load_trees(treefile: Path) -> List[Phylo.BaseTree.Tree]:
    """Read all trees from a Newick file into a list."""
    return list(Phylo.parse(str(treefile), "newick"))

def calculate_null_branch_percent(
    tree: Phylo.BaseTree.Tree,
    tol: float = 1e-6
) -> float:

#    % of internal branches with length <= tol.
#    Uses Biopython's Clade.branch_length.

    # get all internal nodes (nonterminals)
    internals = tree.get_nonterminals()
    # collect their branch lengths
    blens = [cl.branch_length for cl in internals if cl.branch_length is not None]
    if not blens:
        # no internal branches = treat as fully unresolved
        return 1.0
    arr = np.array(blens)
    return np.sum(arr <= tol) / len(arr)

def select_most_resolved_trees(
    trees: List[Phylo.BaseTree.Tree],
    keep_fraction: float = 0.25,
    tol: float = 1e-6
) -> Tuple[List[Phylo.BaseTree.Tree], pd.DataFrame]:

#    Compute percent‐null for each tree, keep the top X% most resolved.
#    Returns (selected_trees, stats_df).
    
    stats = []
    for idx, tr in enumerate(trees):
        pct_null = calculate_null_branch_percent(tr, tol)
        stats.append((idx, pct_null))
    df = pd.DataFrame(stats, columns=["tree_index", "pct_null"])
    df_sorted = df.sort_values("pct_null")
    n_keep = max(1, int(len(df_sorted) * keep_fraction))
    keep_idxs = df_sorted.iloc[:n_keep]["tree_index"].tolist()
    selected = [trees[i] for i in keep_idxs]
    return selected, df_sorted

def write_trees(
    trees: List[Phylo.BaseTree.Tree],
    outpath: Path,
    schema: str = "newick"
):
#    Write a list of Biopython trees back out to file."""
#    Biopython wants a file or handle
    with outpath.open("w") as f:
        Phylo.write(trees, f, schema)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Filter out the X% least-resolved trees by % zero-length internal branches"
    )
    p.add_argument("treefile", type=Path,
                   help="input Newick file (multiple trees)")
    p.add_argument("--keep", type=float, default=0.25,
                   help="fraction to keep (default 0.25)")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="branch-length cutoff for 'zero' (default 1e-6)")
    args = p.parse_args()

# Load
    all_trees = load_trees(args.treefile)
    sel, stats_df = select_most_resolved_trees(
        all_trees, keep_fraction=args.keep, tol=args.tol
    )

# Write outputs
    write_trees(sel, Path("filtered_trees.treefile"))
    stats_df.to_csv("polytomy_stats.csv", index=False)

    print(f"Read       : {len(all_trees)} trees")
    print(f"Kept       : {len(sel)} trees ({len(sel)/len(all_trees):.1%})")
    print("Wrote      : filtered_trees.treefile")
    print("Statistics : polytomy_stats.csv")
