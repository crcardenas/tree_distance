#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_pcs(s: str):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--pcs requires exactly three comma-separated PC indices, e.g. 1,2,3")
    vals = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid PC index: {p}")
        if v < 1:
            raise argparse.ArgumentTypeError("PC indices must be ≥ 1")
        vals.append(v)
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="3D PCA plot (interactive HTML) with optional support/tol coloring."
    )
    parser.add_argument(
        "eigenvec", type=str,
        help="Transposed CSV with eigenvectors (PC1-PCn as rows, tree columns)"
    )
    parser.add_argument(
        "--supp", type=str,
        help="CSV with tree_index and mean_support columns"
    )
    parser.add_argument(
        "--tol", type=str,
        help="CSV with tree_index and pct_null columns"
    )
    parser.add_argument(
        "--pcs", type=parse_pcs, default=[1, 2, 3],
        help="Three comma-separated PC indices to plot, e.g. 1,2,3"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Output HTML filename (default: <eigenvec_stem>_3d_pca.html)"
    )
    args = parser.parse_args()

    if args.tol and args.supp:
        parser.error("--tol and --supp are mutually exclusive")

    # Load and transpose the eigenvector matrix
    df = pd.read_csv(args.eigenvec, index_col=0).transpose().reset_index()
    df = df.rename(columns={"index": "tree"})

    # Convert PC columns to numeric
    pcs = args.pcs
    for i, v in enumerate(pcs, start=1):
        col = f"PC{v}"
        if col not in df.columns:
            parser.error(f"Missing column '{col}' in {args.eigenvec}")
    # ensure numeric
    for col in df.columns:
        if col.startswith("PC"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    color_col = None
    # Add support coloring if provided
    if args.supp:
        supp_df = pd.read_csv(args.supp, usecols=["tree_index", "mean_support"])
        supp_df["tree"] = supp_df["tree_index"].apply(lambda x: f"tree{x}")
        df = df.merge(supp_df[["tree", "mean_support"]], on="tree", how="left")
        color_col = "mean_support"
    # Add tol coloring if provided
    elif args.tol:
        tol_df = pd.read_csv(args.tol, usecols=["tree_index", "pct_null"])
        tol_df["tree"] = tol_df["tree_index"].apply(lambda x: f"tree{x}")
        df = df.merge(tol_df[["tree", "pct_null"]], on="tree", how="left")
        color_col = "pct_null"

    # Create the interactive 3D scatter
    fig = px.scatter_3d(
        df,
        x=f"PC{pcs[0]}",
        y=f"PC{pcs[1]}",
        z=f"PC{pcs[2]}",
        color=color_col if color_col else None,
        hover_name="tree",
        title=f"3D PCA ({pcs[0]},{pcs[1]},{pcs[2]}) of {Path(args.eigenvec).stem}",
        color_continuous_scale="Viridis",
        opacity=0.8,
        height=800
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))

    # Determine output filename
    if args.save:
        out_html = Path(args.save)
    else:
        stem = Path(args.eigenvec).stem
        out_html = Path(f"{stem}_3d_pca.html")

    fig.write_html(str(out_html))
    print(f"[SAVE] 3D PCA interactive plot saved as {out_html}")


if __name__ == "__main__":
    main()
