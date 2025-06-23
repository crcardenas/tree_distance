#!/usr/bin/env python3
# summarize_distances.py
# Cody Raul Cardenas - 20250618 (updated with coloring flags and panel fix)

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
    coords = pca.fit_transform(mat)

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

    # Eigenvectors
    comp_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i}" for i in xs],
        columns=labels
    )
    comp_out = f"{name}_eigenvectors.csv"
    comp_df.to_csv(comp_out)
    print(f"[SAVE] {comp_out}")

    return coords


def plot_pca_scatter(coords: np.ndarray, name: str, color=None, color_label=None):
    print(f"[PLOT] PCA scatter PC1 vs PC2 for {name}")
    plt.figure(figsize=(6, 6))
    if color is not None:
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=color, cmap='viridis', s=30, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label(color_label)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.7, edgecolor='k')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{name} – PC1 vs PC2")
    plt.tight_layout()
    out = f"{name}_pca.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVE] {out}")


def plot_pca_panels(coords: np.ndarray, name: str, pcas: int, color=None, color_label=None):
    """
    Create subplots: PC1 vs PC2..PC{pcas+1}, with optional colorbar.
    """
    n_comp = coords.shape[1]
    pcas = min(pcas, n_comp - 1)
    if pcas < 1:
        print("[WARN] --pcas <1; skipping panels.")
        return

    print(f"[PLOT] PCA panels for {name}: PC1 vs PC2..PC{pcas+1}")
    fig, axes = plt.subplots(1, pcas, figsize=(5 * pcas, 5), constrained_layout=True)
    if pcas == 1:
        axes = [axes]
    for idx, ax in enumerate(axes, start=2):
        if color is not None:
            sc = ax.scatter(coords[:, 0], coords[:, idx], c=color, cmap='viridis', s=30, edgecolor='k')
        else:
            ax.scatter(coords[:, 0], coords[:, idx], s=30, alpha=0.7, edgecolor='k')
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
    parser.add_argument("--supp", type=str, help="CSV of mean_support_stats (tree_index, mean_support)")
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

        coords = run_pca(mat, labels, name, comps=args.comps)

        # prepare coloring vector
        if args.supp:
            df_supp = pd.read_csv(args.supp)
            color = [df_supp.loc[df_supp.tree_index == i, 'mean_support'].values[0] for i in range(len(labels))]
            label_col = 'mean_support'
        elif args.tol:
            df_tol = pd.read_csv(args.tol)
            color = [df_tol.loc[df_tol.tree_index == i, 'pct_null'].values[0] for i in range(len(labels))]
            label_col = 'pct_null'
        else:
            color = None
            label_col = None

        plot_pca_scatter(coords, name, color=color, color_label=label_col)
        plot_pca_panels(coords, name, pcas=args.pcas, color=color, color_label=label_col)


if __name__ == "__main__":
    main()
