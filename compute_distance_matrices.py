#!/usr/bin/env python3
# compute_distance_matrices.py
# Cody Raul Cardenas - 20250623

import argparse
import threading
from pathlib import Path
from typing import List, Tuple, FrozenSet
import numpy as np
import pandas as pd
from Bio import Phylo
from concurrent.futures import ProcessPoolExecutor

# Module‐level verbose flag; default True
VERBOSE = True
print_lock = threading.Lock()


def load_trees(treefile: Path) -> List[Phylo.BaseTree.Tree]:
    """Read all trees (one Newick per line)."""
    return list(Phylo.parse(str(treefile), "newick"))


def get_splits_and_lengths(
    tree: Phylo.BaseTree.Tree,
    taxa: set[str]
) -> Tuple[dict[FrozenSet[str], float], int]:
    """
    From internal nodes, build a {split_set: branch_length} dict,
    and return max possible splits = len(taxa)-3.
    """
    max_splits = max(1, len(taxa) - 3)
    splits: dict[FrozenSet[str], float] = {}

    for cl in tree.get_nonterminals():
        leaves = {t.name for t in cl.get_terminals()} & taxa
        if len(leaves) < 2 or len(leaves) > len(taxa) - 2:
            continue
        comp = taxa - leaves
        side = frozenset(leaves if len(leaves) <= len(comp) else comp)
        splits[side] = cl.branch_length or 0.0

    return splits, max_splits


def compute_pairwise(
    args: Tuple[int, int, List[Phylo.BaseTree.Tree], List[set[str]], bool, bool, bool]
) -> Tuple[int, int, float, float, float]:
    """
    Compute RF and KF distances for a single tree pair.
    args is a 7-tuple:
      (i, j, trees, taxa_sets, do_rf, do_normrf, do_kf)
    """
    i, j, trees, taxa_sets, do_rf, do_normrf, do_kf = args
    L = taxa_sets[i] & taxa_sets[j]

    if len(L) < 4:
        return i, j, 0, 1.0, 0.0

    if VERBOSE:
        with print_lock:
            if do_rf:
                print(f"Calculating Robinson-Foulds distance between trees {i} and {j}")
            if do_normrf:
                print(f"Calculating Normalized Robinson-Foulds distance between trees {i} and {j}")
            if do_kf:
                print(f"Calculating Kuhner–Felsenstein branch-score distance between trees {i} and {j}")

    s1, max_sp = get_splits_and_lengths(trees[i], L)
    s2, _      = get_splits_and_lengths(trees[j], L)

    set1, set2 = set(s1.keys()), set(s2.keys())
    diff = set1 ^ set2
    rf = len(diff)
    normrf = rf / (2 * max_sp) if max_sp > 0 else 0.0

    all_splits = sorted(set1 | set2)
    vec1 = np.array([s1.get(sp, 0.0) for sp in all_splits])
    vec2 = np.array([s2.get(sp, 0.0) for sp in all_splits])
    kf = float(np.linalg.norm(vec1 - vec2))

    return (
        i, j,
        rf     if do_rf     else 0,
        normrf if do_normrf else 0.0,
        kf     if do_kf     else 0.0
    )


def compute_matrices_parallel(
    trees: List[Phylo.BaseTree.Tree],
    threads: int,
    do_rf: bool,
    do_normrf: bool,
    do_kf: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute RF and KF matrices in parallel."""
    n = len(trees)
    rf_mat      = np.zeros((n, n))
    norm_rf_mat = np.zeros((n, n))
    kf_mat      = np.zeros((n, n))
    taxa_sets   = [{t.name for t in tr.get_terminals()} for tr in trees]

    if VERBOSE:
        print(f"> Loaded {n} trees; computing distances using {threads} threads...")

    args_list = [
        (i, j, trees, taxa_sets, do_rf, do_normrf, do_kf)
        for i in range(n) for j in range(i, n)
    ]

    with ProcessPoolExecutor(max_workers=threads) as executor:
        for i, j, rf, normrf, kf in executor.map(compute_pairwise, args_list):
            if do_rf:
                rf_mat[i, j] = rf_mat[j, i] = rf
            if do_normrf:
                norm_rf_mat[i, j] = norm_rf_mat[j, i] = normrf
            if do_kf:
                kf_mat[i, j] = kf_mat[j, i] = kf

    return rf_mat, norm_rf_mat, kf_mat


def save_matrix(mat: np.ndarray, fname: str):
    """Save a square matrix with row/col labels tree0…treeN."""
    n = mat.shape[0]
    names = [f"tree{idx}" for idx in range(n)]
    df = pd.DataFrame(mat, index=names, columns=names)
    df.to_csv(fname, sep="\t", float_format="%.6f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute RF and KF distance matrices"
    )
    parser.add_argument("treefile", type=Path,
                        help="Newick file with one tree per line")
    parser.add_argument("--rf",      action="store_true", help="Compute Robinson-Foulds distance ")
    parser.add_argument("--normrf",  action="store_true", help="Compute normalized Robinson-Foulds distance ")
    parser.add_argument("--kf",      action="store_true", help="Compute Kuhner–Felsenstein branch-score")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of parallel processes")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false",
                        help="Suppress per-pair computation logs")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    # Set verbose from flag
    VERBOSE = args.verbose

    trees = load_trees(args.treefile)
    rf_mat, norm_rf_mat, kf_mat = compute_matrices_parallel(
        trees, args.threads, args.rf, args.normrf, args.kf
    )

    if args.rf:
        save_matrix(rf_mat, "RFmatrix_trees.txt")
        print("Saved: RFmatrix_trees.txt")
    if args.normrf:
        save_matrix(norm_rf_mat, "norm_RFmatrix_trees.txt")
        print("Saved: norm_RFmatrix_trees.txt")
    if args.kf:
        save_matrix(kf_mat, "KFmatrix_trees.txt")
        print("Saved: KFmatrix_trees.txt")
