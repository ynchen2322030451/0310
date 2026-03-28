# run_speedup_benchmark.py
# ============================================================
# Speed benchmark for paper:
#   - neural inference latency
#   - batch throughput
#   - estimated speedup against one high-fidelity simulation
# ============================================================

import os
import json
import time
import pickle
import numpy as np
import torch

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    THRESHOLD_SWEEP,
    SEED,
    FIXED_CKPT_PATH,
    FIXED_SCALER_PATH,
)
from run_phys_levels_main import HeteroMLP, get_device

# ------------------------------------------------------------
# User setting: fill this with your average high-fidelity time
# per sample (seconds). Replace X later with measured value.
# Example:
# HIGH_FIDELITY_SECONDS_PER_CASE = 900.0
# ------------------------------------------------------------
HIGH_FIDELITY_SECONDS_PER_CASE = 3600
FORWARD_UQ_MONTE_CARLO_SAMPLES = 20000
INVERSE_BENCHMARK_CASES = 20
INVERSE_POSTERIOR_SAMPLES_PER_CASE = 1200

LEVEL = 2
N_WARMUP = 50
N_REPEAT_SINGLE = 500
N_REPEAT_BATCH = 200
BATCH_SIZE = 1024


def get_artifact_dir(level: int) -> str:
    mapping = {
        0: os.path.join(OUT_DIR, "fixed_surrogate_fixed_base"),
        2: os.path.join(OUT_DIR, "fixed_surrogate_fixed_level2"),
    }
    if level not in mapping:
        raise ValueError(f"Unsupported level for fixed surrogate: {level}")
    return mapping[level]


def load_checkpoint_and_scalers(level: int):
    art_dir = get_artifact_dir(level)
    ckpt_path = os.path.join(art_dir, f"checkpoint_level{level}.pt")
    scaler_path = os.path.join(art_dir, f"scalers_level{level}.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return ckpt, scalers["sx"], scalers["sy"], ckpt_path, scaler_path


def build_model_from_ckpt(ckpt, device):
    best_params = ckpt["best_params"]
    model = HeteroMLP(
        in_dim=len(INPUT_COLS),
        out_dim=len(OUTPUT_COLS),
        width=int(best_params["width"]),
        depth=int(best_params["depth"]),
        dropout=float(best_params["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    ckpt, sx, sy = load_checkpoint_and_scalers(LEVEL)
    model = build_model_from_ckpt(ckpt, device)

    # random normalized inputs
    x_single = torch.randn(1, len(INPUT_COLS), dtype=torch.float32, device=device)
    x_batch = torch.randn(BATCH_SIZE, len(INPUT_COLS), dtype=torch.float32, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x_single)
            _ = model(x_batch)
    sync_if_needed(device)

    # single-sample latency
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_REPEAT_SINGLE):
            _ = model(x_single)
    sync_if_needed(device)
    t1 = time.perf_counter()
    single_latency = (t1 - t0) / N_REPEAT_SINGLE

    # batch throughput
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_REPEAT_BATCH):
            _ = model(x_batch)
    sync_if_needed(device)
    t1 = time.perf_counter()
    batch_time = (t1 - t0) / N_REPEAT_BATCH
    per_sample_batch_latency = batch_time / BATCH_SIZE
    throughput = BATCH_SIZE / batch_time

    result = {
        "level": LEVEL,
        "device": str(device),
        "single_sample_latency_sec": float(single_latency),
        "batch_size": int(BATCH_SIZE),
        "batch_time_sec": float(batch_time),
        "per_sample_batch_latency_sec": float(per_sample_batch_latency),
        "batch_throughput_samples_per_sec": float(throughput),
        "high_fidelity_seconds_per_case": HIGH_FIDELITY_SECONDS_PER_CASE,
    }

    if HIGH_FIDELITY_SECONDS_PER_CASE is not None:
        result["speedup_single_vs_hf"] = float(HIGH_FIDELITY_SECONDS_PER_CASE / single_latency)
        result["speedup_batch_per_sample_vs_hf"] = float(HIGH_FIDELITY_SECONDS_PER_CASE / per_sample_batch_latency)

    out_json = os.path.join(OUT_DIR, "paper_speedup_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[DONE] Speed benchmark saved to:", out_json)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()