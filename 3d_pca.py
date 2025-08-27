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


def make_plot(df, pcs, color_col, label, out_html):
    fig = px.scatter_3d(
        df,
        x=f"PC{pcs[0]}",
        y=f"PC{pcs[1]}",
        z=f"PC{pcs[2]}",
        color=color_col if color_col else None,
        hover_name="tree",
        title=f"3D PCA ({pcs[0]},{pcs[1]},{pcs[2]}) {label}",
        color_continuous_scale="Viridis",
        opacity=0.8,
        height=800
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
    fig.update_layout(scene_aspectmode='cube')
    fig.write_html(str(out_html))
    print(f"[SAVE] 3D PCA interactive plot saved as {out_html}")


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
        help="CSV with support values: mean_support, or mean_shalrt + mean_bootstrap"
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
        "-p", "--prefix", type=str, default=None,
        help="Prefix for output HTML files (default: eigenvec stem)"
    )
    args = parser.parse_args()

    if args.tol and args.supp:
        parser.error("--tol and --supp are mutually exclusive")

    # Load and transpose the eigenvector matrix
    df = pd.read_csv(args.eigenvec, index_col=0).transpose().reset_index()
    df = df.rename(columns={"index": "tree"})

    # Validate PC columns
    pcs = args.pcs
    for v in pcs:
        col = f"PC{v}"
        if col not in df.columns:
            parser.error(f"Missing column '{col}' in {args.eigenvec}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Determine prefix
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = Path(args.eigenvec).stem

    # Handle support case
    if args.supp:
        supp_df = pd.read_csv(args.supp)
        supp_df["tree"] = supp_df["tree_index"].apply(lambda x: f"tree{x}")
        df = df.merge(supp_df, on="tree", how="left")

        has_shalrt = "mean_shalrt" in supp_df.columns
        has_boot = "mean_bootstrap" in supp_df.columns

        if has_shalrt and has_boot:
            out1 = Path(f"{prefix}_3d_pca_shalrt.html")
            out2 = Path(f"{prefix}_3d_pca_bootstrap.html")
            make_plot(df, pcs, "mean_shalrt", "(SH-aLRT)", out1)
            make_plot(df, pcs, "mean_bootstrap", "(Bootstrap)", out2)
            return
        elif "mean_support" in supp_df.columns:
            color_col = "mean_support"
        elif has_shalrt:
            color_col = "mean_shalrt"
        elif has_boot:
            color_col = "mean_bootstrap"
        else:
            raise RuntimeError(f"No usable support columns found in {args.supp}")

        out_html = Path(f"{prefix}_3d_pca.html")
        make_plot(df, pcs, color_col, "", out_html)

    # Handle tol case
    elif args.tol:
        tol_df = pd.read_csv(args.tol, usecols=["tree_index", "pct_null"])
        tol_df["tree"] = tol_df["tree_index"].apply(lambda x: f"tree{x}")
        df = df.merge(tol_df[["tree", "pct_null"]], on="tree", how="left")
        out_html = Path(f"{prefix}_3d_pca.html")
        make_plot(df, pcs, "pct_null", "(tol)", out_html)

    # No coloring
    else:
        out_html = Path(f"{prefix}_3d_pca.html")
        make_plot(df, pcs, None, "", out_html)


if __name__ == "__main__":
    main()
