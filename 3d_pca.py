#!/usr/bin/env python3
# 3d_pca.py
# Cody Raul Cardenas - 20250618

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

def parse_pcs(s: str):
    """
    Parse a comma-separated list of three integers for PC axes.
    """
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("`--pcs` requires exactly three comma-separated values, e.g. 1,2,3")
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("All values in `--pcs` must be integers.")
    if any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError("PC indices must be 1 or greater.")
    return vals

def main():
    p = argparse.ArgumentParser(
        description="3D PCA plot from eigenvectors CSV with selectable axes"
    )
    p.add_argument("eigenvec_csv", type=Path,
                   help="CSV file of eigenvectors (rows=PCs, cols=samples)")
    p.add_argument("--pcs", type=parse_pcs, default=[1,2,3],
                   help="Three comma-separated PC indices to plot, e.g. 1,2,3 (default)")
    p.add_argument("--html", type=Path, default=None,
                   help="Output HTML file (default: <stem>_3d_pca.html)")
    args = p.parse_args()

    # Load eigenvectors: rows are PC1, PC2, ..., columns are tree names
    df = pd.read_csv(args.eigenvec_csv, index_col=0)
    coords = df.T  # now rows are trees, columns are PCs

    pcs = args.pcs
    cols = [f"PC{v}" for v in pcs]
    missing = [c for c in cols if c not in coords.columns]
    if missing:
        raise ValueError(f"Missing components in CSV: {missing}")

    # build the DataFrame for plotting
    plot_df = coords[cols].reset_index().rename(columns={"index": "tree"})

    # 3D scatter
    fig = px.scatter_3d(
        plot_df,
        x=cols[0], y=cols[1], z=cols[2],
        text="tree",
        title=f"3D PCA ({cols[0]}, {cols[1]}, {cols[2]}) of {args.eigenvec_csv.stem}",
        labels={"tree": "Tree"}
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))

    out_html = args.html or args.eigenvec_csv.with_name(f"{args.eigenvec_csv.stem}_3d_pca.html")
    print(f"[SAVE] {out_html}")
    fig.write_html(str(out_html))
    print("Done.")

if __name__ == "__main__":
    main()
