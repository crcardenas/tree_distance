#!/usr/bin/env python3
# compute_distance_matrices.py
# Cody Raul Cardenas - 20250725

import argparse
from pathlib import Path
import tempfile
import subprocess
import pandas as pd
import sys
from pathlib import Path
import re

# run IQTree2
def run_IQTree2_RF(tree: Path, prefix: str):
    cmd = ["iqtree", "-rf_all", str(tree), "--prefix", str(prefix)]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[SAVE] {prefix}.rfdist")

def convert_matrix(path: Path, prefix: str = ""):
    output_path = Path(f"{prefix}.tsv") if prefix else Path("RF_matrix.tsv")
    with open(path, "r") as f:
        lines = f.readlines()

    lines = lines[1:]
    cleaned_data = [line.strip().split() for line in lines]
    row_names = [row[0] for row in cleaned_data]
    data = [row[1:] for row in cleaned_data]

    df = pd.DataFrame(data, columns=row_names, index=row_names)
    df.index = [name.lower() for name in df.index]
    df.columns = [name.lower() for name in df.columns]
    
    df.to_csv(output_path, sep='\t')
    print(f"[SAVE] Converted matrix saved to {output_path}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Compute Robinson-Foulds distance using IQtree v 2.3.6"
    )
    parser.add_argument(
        "-t", "--tree", 
        type=Path, 
        required=True,
        help="Input multi-tree file (one Newick per line)"
    )
    parser.add_argument(
        "-p", "--prefix", 
        type=str,
        default="",
        help="Prefix for output, otherwise default output 'RF_matrix.tsv'"
    )
    args = parser.parse_args()

    prefix = f"{args.prefix}" if args.prefix else ""

    run_IQTree2_RF(tree=args.tree, prefix=args.prefix)
    convert_matrix(path=Path(f"{prefix}.rfdist"), prefix=prefix)


if __name__ == "__main__":
    main()
