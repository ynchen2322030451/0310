import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "./experiments_phys_levels"

RUN_TAG = "reduced"
CASE_ID = 3

PREFIX = os.path.join(OUT_DIR, f"paper_2d_case{CASE_ID:03d}_{RUN_TAG}")

PRIOR_FILE = f"{PREFIX}_prior_points.csv"
POST_FILE = f"{PREFIX}_posterior_points.csv"
GRID_FILE = f"{PREFIX}_grid_map.csv"
META_FILE = f"{PREFIX}_meta.json"
FEAS131_FILE = f"{PREFIX}_feasible_points_thr131.csv"

FIG_DIR = os.path.join(OUT_DIR, "paper_inverse_2d_figures")
os.makedirs(FIG_DIR, exist_ok=True)


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def hist2d_density(ax, df, xcol, ycol, bins=60, alpha=0.7, label=None):
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    h = ax.hist2d(x, y, bins=bins, density=True, alpha=alpha)
    if label is not None:
        ax.text(
            0.02, 0.95, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )
    return h


def contour_from_grid(ax, df_grid, xcol, ycol, zcol, levels=15, filled=True, alpha=0.85):
    xs = np.sort(df_grid[xcol].unique())
    ys = np.sort(df_grid[ycol].unique())

    nx = len(xs)
    ny = len(ys)

    z = df_grid.pivot(index=ycol, columns=xcol, values=zcol).values

    XX, YY = np.meshgrid(xs, ys)

    if filled:
        cs = ax.contourf(XX, YY, z, levels=levels, alpha=alpha)
    else:
        cs = ax.contour(XX, YY, z, levels=levels, alpha=alpha)
    return cs


def main():
    for p in [PRIOR_FILE, POST_FILE, GRID_FILE, META_FILE, FEAS131_FILE]:
        require_file(p)

    df_prior = pd.read_csv(PRIOR_FILE)
    df_post = pd.read_csv(POST_FILE)
    df_grid = pd.read_csv(GRID_FILE)
    df_feas131 = pd.read_csv(FEAS131_FILE)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    xcol = meta["param_x"]
    ycol = meta["param_y"]

    # --------------------------------------------------
    # Figure 1: prior vs posterior density
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    hist2d_density(axes[0], df_prior, xcol, ycol, bins=55, alpha=0.85, label="Prior")
    axes[0].set_xlabel(xcol)
    axes[0].set_ylabel(ycol)
    axes[0].set_title("Prior samples in dominant-parameter plane")

    hist2d_density(axes[1], df_post, xcol, ycol, bins=55, alpha=0.85, label="Posterior")
    axes[1].set_xlabel(xcol)
    axes[1].set_ylabel(ycol)
    axes[1].set_title("Posterior samples in dominant-parameter plane")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"case{CASE_ID:03d}_prior_vs_posterior_density_{RUN_TAG}.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------
    # Figure 2: feasible-region color map (131 MPa)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    cs = contour_from_grid(
        ax=ax,
        df_grid=df_grid,
        xcol=xcol,
        ycol=ycol,
        zcol="stress_mean",
        levels=18,
        filled=True,
        alpha=0.85
    )
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Predicted stress mean (MPa)")

    ax.contour(
        np.sort(df_grid[xcol].unique()),
        np.sort(df_grid[ycol].unique()),
        df_grid.pivot(index=ycol, columns=xcol, values="stress_mean").values,
        levels=[131.0],
        linewidths=1.8
    )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Stress mean map with 131 MPa boundary")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"case{CASE_ID:03d}_stress_mean_map_thr131_{RUN_TAG}.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------
    # Figure 3: feasible-region map + posterior overlay
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    cs = contour_from_grid(
        ax=ax,
        df_grid=df_grid,
        xcol=xcol,
        ycol=ycol,
        zcol="stress_mean",
        levels=18,
        filled=True,
        alpha=0.78
    )
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Predicted stress mean (MPa)")

    # threshold contour
    ax.contour(
        np.sort(df_grid[xcol].unique()),
        np.sort(df_grid[ycol].unique()),
        df_grid.pivot(index=ycol, columns=xcol, values="stress_mean").values,
        levels=[131.0],
        linewidths=1.8
    )

    # posterior overlay
    n_plot = min(1200, len(df_post))
    df_plot = df_post.sample(n=n_plot, random_state=2026) if len(df_post) > n_plot else df_post
    ax.scatter(
        df_plot[xcol], df_plot[ycol],
        s=7, alpha=0.22, linewidths=0
    )

    # feasible posterior overlay
    if len(df_feas131) > 0:
        n_plot_feas = min(800, len(df_feas131))
        df_feas_plot = df_feas131.sample(n=n_plot_feas, random_state=2026) if len(df_feas131) > n_plot_feas else df_feas131
        ax.scatter(
            df_feas_plot[xcol], df_feas_plot[ycol],
            s=10, alpha=0.45, linewidths=0, marker="x"
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Posterior samples relative to the 131 MPa feasible region")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"case{CASE_ID:03d}_posterior_overlay_thr131_{RUN_TAG}.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------
    # Save plotting meta
    # --------------------------------------------------
    out_meta = {
        "case_id": CASE_ID,
        "run_tag": RUN_TAG,
        "param_x": xcol,
        "param_y": ycol,
        "figure_dir": FIG_DIR,
        "input_files": {
            "prior": PRIOR_FILE,
            "posterior": POST_FILE,
            "grid": GRID_FILE,
            "meta": META_FILE,
            "feasible_131": FEAS131_FILE,
        },
        "generated_figures": [
            f"case{CASE_ID:03d}_prior_vs_posterior_density_{RUN_TAG}.png",
            f"case{CASE_ID:03d}_stress_mean_map_thr131_{RUN_TAG}.png",
            f"case{CASE_ID:03d}_posterior_overlay_thr131_{RUN_TAG}.png",
        ],
    }

    with open(os.path.join(FIG_DIR, f"case{CASE_ID:03d}_plotting_meta_{RUN_TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()