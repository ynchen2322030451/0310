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

FIG_DIR = os.path.join(OUT_DIR, "paper_inverse_2d_figures_final")
os.makedirs(FIG_DIR, exist_ok=True)

PRIOR_BINS = 80
POST_BINS = 80
POST_SAMPLE_MAX = 1200
FEAS_SAMPLE_MAX = 800


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def build_hist_density(df, xcol, ycol, bins=80):
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    xedges = np.linspace(np.min(x), np.max(x), bins + 1)
    yedges = np.linspace(np.min(y), np.max(y), bins + 1)

    H, xe, ye = np.histogram2d(x, y, bins=[xedges, yedges], density=True)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])

    XX, YY = np.meshgrid(xc, yc, indexing="xy")
    Z = H.T
    return XX, YY, Z


def contour_levels_from_density(Z, percentiles=(60, 80, 92, 97)):
    vals = Z[Z > 0].ravel()
    if len(vals) == 0:
        return None
    levels = sorted(np.percentile(vals, percentiles))
    levels = np.unique(levels)
    if len(levels) < 2:
        return None
    return levels


def plot_prior_posterior_contours(df_prior, df_post, xcol, ycol, save_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    XXp, YYp, Zp = build_hist_density(df_prior, xcol, ycol, bins=PRIOR_BINS)
    XXq, YYq, Zq = build_hist_density(df_post, xcol, ycol, bins=POST_BINS)

    levels_p = contour_levels_from_density(Zp)
    levels_q = contour_levels_from_density(Zq)

    if levels_p is not None:
        ax.contour(
            XXp, YYp, Zp,
            levels=levels_p,
            linewidths=1.2,
            linestyles="--",
            alpha=0.8
        )

    if levels_q is not None:
        ax.contour(
            XXq, YYq, Zq,
            levels=levels_q,
            linewidths=1.6,
            alpha=0.95
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Prior–posterior contraction in dominant-parameter plane")

    ax.text(
        0.03, 0.96,
        "Dashed: prior\nSolid: posterior",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def make_grid_arrays(df_grid, xcol, ycol, zcol):
    xs = np.sort(df_grid[xcol].unique())
    ys = np.sort(df_grid[ycol].unique())

    Z = df_grid.pivot(index=ycol, columns=xcol, values=zcol).values
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY, Z


def plot_feasible_map_with_overlay(df_grid, df_post, df_feas131, xcol, ycol, save_path):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))

    XX, YY, Z = make_grid_arrays(df_grid, xcol, ycol, "stress_mean")

    cs = ax.contourf(XX, YY, Z, levels=18, alpha=0.88)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Predicted stress mean (MPa)")

    boundary = ax.contour(
        XX, YY, Z,
        levels=[131.0],
        linewidths=2.1
    )

    # posterior samples
    if len(df_post) > POST_SAMPLE_MAX:
        df_post_plot = df_post.sample(POST_SAMPLE_MAX, random_state=2026)
    else:
        df_post_plot = df_post

    ax.scatter(
        df_post_plot[xcol],
        df_post_plot[ycol],
        s=8,
        alpha=0.20,
        linewidths=0
    )

    # feasible posterior samples
    if len(df_feas131) > 0:
        if len(df_feas131) > FEAS_SAMPLE_MAX:
            df_feas_plot = df_feas131.sample(FEAS_SAMPLE_MAX, random_state=2026)
        else:
            df_feas_plot = df_feas131

        ax.scatter(
            df_feas_plot[xcol],
            df_feas_plot[ycol],
            s=12,
            alpha=0.55,
            marker="x"
        )

    xmin, xmax = np.min(XX), np.max(XX)
    ymin, ymax = np.min(YY), np.max(YY)

    ax.text(
        xmin + 0.03 * (xmax - xmin),
        ymin + 0.08 * (ymax - ymin),
        "feasible",
        fontsize=11,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

    ax.text(
        xmax - 0.18 * (xmax - xmin),
        ymax - 0.06 * (ymax - ymin),
        "infeasible",
        fontsize=11,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

    ax.text(
        0.03, 0.96,
        "Solid boundary: 131 MPa\nDots: posterior\nCrosses: feasible posterior",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Posterior samples relative to the 131 MPa feasible region")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_stress_mean_map_clean(df_grid, xcol, ycol, save_path):
    fig, ax = plt.subplots(figsize=(6.3, 5.2))

    XX, YY, Z = make_grid_arrays(df_grid, xcol, ycol, "stress_mean")

    cs = ax.contourf(XX, YY, Z, levels=18, alpha=0.9)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Predicted stress mean (MPa)")

    ax.contour(
        XX, YY, Z,
        levels=[131.0],
        linewidths=2.0
    )

    xmin, xmax = np.min(XX), np.max(XX)
    ymin, ymax = np.min(YY), np.max(YY)

    ax.text(
        xmin + 0.03 * (xmax - xmin),
        ymin + 0.08 * (ymax - ymin),
        "feasible",
        fontsize=11,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

    ax.text(
        xmax - 0.18 * (xmax - xmin),
        ymax - 0.06 * (ymax - ymin),
        "infeasible",
        fontsize=11,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
    )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Stress mean map with 131 MPa boundary")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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

    out1 = os.path.join(FIG_DIR, f"case{CASE_ID:03d}_prior_posterior_contours_{RUN_TAG}.png")
    out2 = os.path.join(FIG_DIR, f"case{CASE_ID:03d}_stress_mean_map_clean_{RUN_TAG}.png")
    out3 = os.path.join(FIG_DIR, f"case{CASE_ID:03d}_posterior_overlay_final_{RUN_TAG}.png")

    plot_prior_posterior_contours(df_prior, df_post, xcol, ycol, out1)
    plot_stress_mean_map_clean(df_grid, xcol, ycol, out2)
    plot_feasible_map_with_overlay(df_grid, df_post, df_feas131, xcol, ycol, out3)

    meta_out = {
        "case_id": CASE_ID,
        "run_tag": RUN_TAG,
        "param_x": xcol,
        "param_y": ycol,
        "generated": [out1, out2, out3],
        "note": "Other reduced parameters are fixed at posterior medians in the 2D grid map.",
    }

    with open(os.path.join(FIG_DIR, f"case{CASE_ID:03d}_plotting_meta_final_{RUN_TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved final-style 2D figures to:", FIG_DIR)


if __name__ == "__main__":
    main()