#!/usr/bin/env python3
#Cody Raul Cardenas 20250725
"""
filter_by_pca.py

Split a tree list according to thresholds on PCA axes.

Example:
  python filter_by_pca.py \
    --eigenvec RFmatrix_trees_eigenvectors.csv \
    --treefile filtered_trees.treefile \
    --filter PC1>=0.033 --filter PC1<=0.47 --filter PC2<0 \
    --prefix mycluster \
    --log IQtree_locustrees.log
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd
from Bio import Phylo

# Allowed comparison ops
OPS = {
    '>':  lambda s, v: s > v,
    '>=': lambda s, v: s >= v,
    '<':  lambda s, v: s < v,
    '<=': lambda s, v: s <= v,
    '==': lambda s, v: s == v,
}

FILTER_RE = re.compile(r'^(PC\d+)\s*(<=|>=|<|>|==)\s*(-?\d+(\.\d+)?)$')

LOG_LINE_RE = re.compile(r'^\s*(\d+)\s+[A-Za-z]*\s+\d+\s+\d+\s+\d+\s+\d+\s+[\w+]+/?.*?\s+(p\d+_.+)$')

def parse_filter(expr: str):
    m = FILTER_RE.match(expr.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid filter: '{expr}'")
    pc, op, val, _ = m.groups()
    return pc, OPS[op], float(val)

def load_eigenvec(path: Path):
    df = pd.read_csv(path, index_col=0)
    return df.T  # Transpose: rows=trees, cols=PCs

def load_trees(path: Path):
    return list(Phylo.parse(str(path), "newick"))

def write_trees(trees, path: Path):
    with path.open('w') as f:
        Phylo.write(trees, f, 'newick')

def parse_log_file(log_path: Path):
    mapping = {}
    in_table = False
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            # Detect start of table
            if line.startswith("Subset") and "Name" in line:
                in_table = True
                continue
            # Detect end of table
            if line.startswith("Linked") and "total sequences" in line:
                break
            if not in_table:
                continue  # Skip lines outside the target block
            if "WARNING" in line:
                continue  # Skip warning lines
            # Split line into fields — should be tab-separated
            fields = line.split('\t')
            if len(fields) < 8:
                continue  # Skip malformed lines
            try:
                subset_idx = int(fields[0])
                partition = fields[-1]
                tree_name = f"tree{subset_idx - 1}"  # tree0 == subset 1
                clean_partition = "_".join(partition.split("_")[1:])  # Remove p#_
                mapping[tree_name] = clean_partition
            except Exception:
                continue  # Skip lines with bad formatting

    return mapping

def main():
    p = argparse.ArgumentParser(description="Filter trees by PCA axis thresholds")
    p.add_argument('--eigenvec', type=Path, required=True,
                   help="CSV of eigenvectors (rows=PCs, cols=tree names)")
    p.add_argument('--treefile', type=Path, required=True,
                   help="Newick file (one tree per line) matching eigenvec columns")
    p.add_argument('--filter', metavar='EXPR', action='append', required=True,
                   help="Filter expression, e.g. 'PC1>=0.033' (can repeat)")
    p.add_argument('--prefix', default='cluster',
                   help="Prefix for output files (default 'cluster')")
    p.add_argument('--log', type=Path,
                   help="Log file with partition names (optional)")
    args = p.parse_args()

    filters = [parse_filter(f) for f in args.filter]
    ev = load_eigenvec(args.eigenvec)

    for pc, _, _ in filters:
        if pc not in ev.columns:
            sys.exit(f"[ERROR] {pc} not found in eigenvector columns: {list(ev.columns)}")

    mask = pd.Series(True, index=ev.index)
    for pc, fn, val in filters:
        mask &= fn(ev[pc], val)

    selected = ev.index[mask].tolist()
    rejected = ev.index[~mask].tolist()

    if args.log:
        tree_to_partition = parse_log_file(args.log)
        sel_data = [(t, tree_to_partition.get(t, 'NA')) for t in selected]
        rej_data = [(t, tree_to_partition.get(t, 'NA')) for t in rejected]
        pd.DataFrame(sel_data, columns=['tree', 'partition']).to_csv(f"{args.prefix}_selected.csv", index=False)
        pd.DataFrame(rej_data, columns=['tree', 'partition']).to_csv(f"{args.prefix}_rejected.csv", index=False)
    else:
        pd.DataFrame(selected, columns=['tree']).to_csv(f"{args.prefix}_selected.csv", index=False)
        pd.DataFrame(rejected, columns=['tree']).to_csv(f"{args.prefix}_rejected.csv", index=False)

    print(f"[SAVE] {args.prefix}_selected.csv ({len(selected)} trees)")
    print(f"[SAVE] {args.prefix}_rejected.csv ({len(rejected)} trees)")

    trees = load_trees(args.treefile)
    name_to_tree = dict(zip(ev.index, trees))

    sel_trees = [name_to_tree[n] for n in selected if n in name_to_tree]
    rej_trees = [name_to_tree[n] for n in rejected if n in name_to_tree]

    write_trees(sel_trees, Path(f"{args.prefix}_selected.treefile"))
    write_trees(rej_trees, Path(f"{args.prefix}_rejected.treefile"))
    print(f"[SAVE] {args.prefix}_selected.treefile")
    print(f"[SAVE] {args.prefix}_rejected.treefile")

if __name__ == "__main__":
    main()
