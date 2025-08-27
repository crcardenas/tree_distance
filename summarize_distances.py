#!/usr/bin/env python3
# summarize_distances.py
# Cody Raul Cardenas - corrected 20250721 (eigenvectors plotted directly)
# updated 20250827 to handle dual supports (shalrt + bootstrap) and robust column detection

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def load_matrix(path: Path):
    print(f"[LOAD] Reading matrix from {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.shape[0] != df.shape[1]:
        raise RuntimeError(f"Matrix {path.name} is not square: {df.shape}")
    labels = df.index.tolist()
    return df.values, labels


def summarize(mat: np.ndarray, name: str):
    print(f"\n[SUMMARY] {name}")
    print(f"  shape     {mat.shape}")
    print(f"  min       {mat.min():.6f}")
    print(f"  max       {mat.max():.6f}")
    print(f"  mean      {mat.mean():.6f}")
    print(f"  median    {np.median(mat):.6f}")
    print(f"  std.dev   {mat.std():.6f}")


def plot_heatmap(mat: np.ndarray, name: str):
    print(f"[PLOT] Heatmap for {name}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"shrink": 0.8})
    plt.title(f"{name} Distance Matrix")
    plt.xlabel("Tree index")
    plt.ylabel("Tree index")
    plt.tight_layout()
    out = f"{name}_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVE] {out}")


def plot_histogram(mat: np.ndarray, name: str):
    print(f"[PLOT] Histogram for {name}")
    triu = mat[np.triu_indices_from(mat, k=1)]
    plt.figure(figsize=(6, 4))
    plt.hist(triu, bins=50, edgecolor="black")
    plt.title(f"{name} Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    out = f"{name}_histogram.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVE] {out}")


def run_pca(mat: np.ndarray, labels: list[str], name: str, comps: int):
    n_samples = mat.shape[0]
    n_comp = min(comps, n_samples)
    print(f"[PCA] Performing PCA with {n_comp} components")
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(mat)  # still computed to produce eigenvalues/vecs

    # Scree plot
    print(f"[PLOT] Scree plot for {name}")
    plt.figure(figsize=(6, 4))
    xs = np.arange(1, n_comp + 1)
    plt.bar(xs, pca.explained_variance_ratio_, edgecolor="black")
    plt.xticks(xs)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"{name} PCA Scree")
    plt.tight_layout()
    scree_out = f"{name}_scree.png"
    plt.savefig(scree_out, dpi=300)
    plt.close()
    print(f"[SAVE] {scree_out}")

    # Eigenvalues
    eig_df = pd.DataFrame({
        "PC": xs,
        "eigenvalue": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    eig_out = f"{name}_eigenvalues.csv"
    eig_df.to_csv(eig_out, index=False)
    print(f"[SAVE] {eig_out}")

    # Eigenvectors (rows PC1..PCn, columns = labels)
    comp_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i}" for i in xs],
        columns=labels
    )
    comp_out = f"{name}_eigenvectors.csv"
    comp_df.to_csv(comp_out)
    print(f"[SAVE] {comp_out}")


def _build_color_vector_from_df(df_supp: pd.DataFrame, length: int, col: str):
    # build fast mapping from tree_index -> value; return list for indices 0..length-1
    if 'tree_index' not in df_supp.columns:
        raise RuntimeError("Support file is missing 'tree_index' column")
    mapping = df_supp.set_index('tree_index')[col].to_dict()
    return [mapping.get(i, np.nan) for i in range(length)]


def plot_dual_support_pca(eigvec_df: pd.DataFrame, name: str, shalrt_vals: list, boot_vals: list):
    pc1 = eigvec_df.loc["PC1"].values
    pc2 = eigvec_df.loc["PC2"].values

    print(f"[PLOT] Dual PCA scatter (SH-aLRT & Bootstrap) for {name}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    sc1 = axes[0].scatter(pc1, pc2, c=shalrt_vals, cmap="viridis", s=30, edgecolor='k')
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("SH-aLRT support")
    fig.colorbar(sc1, ax=axes[0], shrink=0.7, label="mean_shalrt")

    sc2 = axes[1].scatter(pc1, pc2, c=boot_vals, cmap="viridis", s=30, edgecolor='k')
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Bootstrap support")
    fig.colorbar(sc2, ax=axes[1], shrink=0.7, label="mean_bootstrap")

    out = f"{name}_pca.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVE] {out}")


def plot_pca_scatter(eigvec_df: pd.DataFrame, name: str, color=None, color_label=None):
    print(f"[PLOT] PCA scatter PC1 vs PC2 for {name}")
    pc1 = eigvec_df.loc["PC1"].values
    pc2 = eigvec_df.loc["PC2"].values

    plt.figure(figsize=(6, 6))
    if color is not None:
        sc = plt.scatter(pc1, pc2, c=color, cmap='viridis', s=30, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label(color_label)
    else:
        plt.scatter(pc1, pc2, s=30, alpha=0.7, edgecolor='k')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{name} – PC1 vs PC2")
    plt.tight_layout()
    out = f"{name}_pca.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVE] {out}")


def plot_pca_panels(eigvec_df: pd.DataFrame, name: str, pcas: int, color=None, color_label=None):
    pcs = list(eigvec_df.index)
    pcas = min(pcas, len(pcs) - 1)
    if pcas < 1:
        print("[WARN] --pcas <1; skipping panels.")
        return

    print(f"[PLOT] PCA panels for {name}: PC1 vs PC2..PC{pcas+1}")
    fig, axes = plt.subplots(1, pcas, figsize=(5 * pcas, 5), constrained_layout=True)
    if pcas == 1:
        axes = [axes]
    pc1 = eigvec_df.loc["PC1"].values
    for idx, ax in enumerate(axes, start=2):
        pci = eigvec_df.loc[f"PC{idx}"].values
        if color is not None:
            sc = ax.scatter(pc1, pci, c=color, cmap='viridis', s=30, edgecolor='k')
        else:
            ax.scatter(pc1, pci, s=30, alpha=0.7, edgecolor='k')
        ax.set_xlabel("PC1")
        ax.set_ylabel(f"PC{idx}")
        ax.set_title(f"PC1 vs PC{idx}")
    if color is not None:
        fig.colorbar(sc, ax=axes, shrink=0.6, label=color_label)
    out = f"{name}_pca_panels.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVE] {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize & visualize distance matrices with PCA and optional coloring by tol or supp stats."
    )
    parser.add_argument("matrices", nargs="+", help="Tab-delimited matrix files (RF, normRF, KF)")
    parser.add_argument("--hist", action="store_true", help="Generate histograms")
    parser.add_argument("--comps", type=int, default=10, help="PCA components to compute")
    parser.add_argument("--pcas", type=int, default=1, help="Additional PC panels")
    parser.add_argument("--supp", type=str, help="CSV of mean_support_stats OR mean_shalrt/mean_bootstrap stats")
    parser.add_argument("--tol", type=str, help="CSV of tol_polytomy_stats (tree_index, pct_null)")
    args = parser.parse_args()

    if args.supp and args.tol:
        parser.error("--supp and --tol are mutually exclusive")

    for path_str in args.matrices:
        path = Path(path_str)
        if not path.exists():
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue
        name = path.stem

        mat, labels = load_matrix(path)
        summarize(mat, name)
        plot_heatmap(mat, name)
        if args.hist:
            plot_histogram(mat, name)

        run_pca(mat, labels, name, comps=args.comps)

        eig_file = f"{name}_eigenvectors.csv"
        eigvec_df = pd.read_csv(eig_file, index_col=0)

        color = None
        label_col = None

        if args.supp:
            df_supp = pd.read_csv(args.supp)
            # dual support columns present -> two-panel PCA
            if "mean_shalrt" in df_supp.columns and "mean_bootstrap" in df_supp.columns:
                shalrt_vec = _build_color_vector_from_df(df_supp, len(labels), "mean_shalrt")
                boot_vec = _build_color_vector_from_df(df_supp, len(labels), "mean_bootstrap")
                plot_dual_support_pca(eigvec_df, name, shalrt_vec, boot_vec)
                continue
            # single-column possibilities
            if "mean_support" in df_supp.columns:
                color = _build_color_vector_from_df(df_supp, len(labels), "mean_support")
                label_col = "mean_support"
            elif "mean_shalrt" in df_supp.columns:
                color = _build_color_vector_from_df(df_supp, len(labels), "mean_shalrt")
                label_col = "mean_shalrt"
            elif "mean_bootstrap" in df_supp.columns:
                color = _build_color_vector_from_df(df_supp, len(labels), "mean_bootstrap")
                label_col = "mean_bootstrap"
            else:
                raise RuntimeError(f"Unrecognized support columns in {args.supp}. Expected 'mean_support' or 'mean_shalrt'/'mean_bootstrap'.")

        elif args.tol:
            df_tol = pd.read_csv(args.tol)
            color = _build_color_vector_from_df(df_tol, len(labels), "pct_null")
            label_col = "pct_null"

        plot_pca_scatter(eigvec_df, name, color=color, color_label=label_col)
        plot_pca_panels(eigvec_df, name, pcas=args.pcas, color=color, color_label=label_col)


if __name__ == "__main__":
    main()
