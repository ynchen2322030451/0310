# run_practical_speed_benchmark.py
# ============================================================
# Practical wall-clock speed benchmark for:
#   1) high-fidelity workflow (typically CPU)
#   2) neural-network / surrogate workflow (typically GPU)
#
# Output:
#   - paper_speed_benchmark_detailed.json
# ============================================================

import os
import json
import time
import shutil
import socket
import platform
import subprocess
from datetime import datetime

import numpy as np

try:
    import psutil
except ImportError:
    raise ImportError("Please install psutil: pip install psutil")

# ============================================================
# User settings
# ============================================================

OUT_DIR = "./experiments_phys_levels"
OUTPUT_JSON = os.path.join(OUT_DIR, "paper_speed_benchmark_detailed.json")

# Replace these
HF_COMMAND = ["/home/tjzs/.conda/envs/HP-env/bin/python", "/home/tjzs/Documents/fenics_data/fenics_data/0318-UQ-test-fenics.py"]
NN_COMMAND = ["/home/tjzs/.conda/envs/pytorch-env/bin/python", "/home/tjzs/Documents/0310/run_speedup_benchmark.py"]

HF_REPEATS = 3
NN_REPEATS = 5

HF_TIMEOUT = None
NN_TIMEOUT = None

SAVE_STDOUT_STDERR = True
TOP_N_PROCESSES = 12


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_str():
    return datetime.now().isoformat()


def command_to_str(cmd):
    return " ".join(cmd)


def safe_check_output(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def run_subprocess(cmd, timeout=None):
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    t1 = time.perf_counter()

    return {
        "returncode": proc.returncode,
        "wall_clock_sec": float(t1 - t0),
        "stdout": proc.stdout if SAVE_STDOUT_STDERR else "",
        "stderr": proc.stderr if SAVE_STDOUT_STDERR else "",
    }


def get_cpu_info():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_current_mhz": None,
        "cpu_freq_min_mhz": None,
        "cpu_freq_max_mhz": None,
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 3),
    }

    try:
        f = psutil.cpu_freq()
        if f is not None:
            info["cpu_freq_current_mhz"] = f.current
            info["cpu_freq_min_mhz"] = f.min
            info["cpu_freq_max_mhz"] = f.max
    except Exception:
        pass

    cpu_model = None
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    info["cpu_model_name"] = cpu_model
    return info


def get_top_process_snapshot(top_n=12):
    rows = []
    for p in psutil.process_iter(["pid", "name", "username", "cpu_percent", "memory_percent", "cmdline"]):
        try:
            rows.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "username": p.info["username"],
                "cpu_percent": p.info["cpu_percent"],
                "memory_percent": p.info["memory_percent"],
                "cmdline": " ".join(p.info["cmdline"]) if p.info["cmdline"] else "",
            })
        except Exception:
            continue

    rows = sorted(rows, key=lambda x: (x["cpu_percent"], x["memory_percent"]), reverse=True)
    return rows[:top_n]


def get_system_load_snapshot():
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    snapshot = {
        "timestamp": now_str(),
        "loadavg": None,
        "cpu_percent_total": psutil.cpu_percent(interval=0.5),
        "memory_percent": vm.percent,
        "memory_used_gb": round(vm.used / (1024**3), 3),
        "memory_available_gb": round(vm.available / (1024**3), 3),
        "swap_percent": sm.percent,
        "top_processes": get_top_process_snapshot(TOP_N_PROCESSES),
    }

    try:
        snapshot["loadavg"] = os.getloadavg()
    except Exception:
        snapshot["loadavg"] = None

    return snapshot


def has_nvidia_smi():
    return shutil.which("nvidia-smi") is not None


def get_gpu_info():
    if not has_nvidia_smi():
        return {
            "gpu_available": False,
            "reason": "nvidia-smi not found"
        }

    query = safe_check_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits"
    ])

    gpus = []
    if query:
        lines = [x.strip() for x in query.splitlines() if x.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "name": parts[0],
                    "memory_total_mb": parts[1],
                    "memory_used_mb": parts[2],
                    "gpu_util_percent": parts[3],
                    "mem_util_percent": parts[4],
                    "temperature_c": parts[5],
                })

    proc_query = safe_check_output([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits"
    ])

    gpu_processes = []
    if proc_query:
        lines = [x.strip() for x in proc_query.splitlines() if x.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_processes.append({
                    "pid": parts[0],
                    "process_name": parts[1],
                    "used_memory_mb": parts[2],
                    "gpu_uuid": parts[3],
                })

    return {
        "gpu_available": True,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "gpu_processes": gpu_processes,
    }


def summarize_runs(run_list):
    times = [r["wall_clock_sec"] for r in run_list if r["returncode"] == 0]
    if len(times) == 0:
        return {
            "n_success": 0,
            "n_total": len(run_list),
            "mean_sec": None,
            "std_sec": None,
            "min_sec": None,
            "max_sec": None,
        }

    arr = np.asarray(times, dtype=float)
    return {
        "n_success": int(len(times)),
        "n_total": int(len(run_list)),
        "mean_sec": float(np.mean(arr)),
        "std_sec": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min_sec": float(np.min(arr)),
        "max_sec": float(np.max(arr)),
    }


def benchmark_command(label, cmd, repeats, timeout):
    print(f"[INFO] Benchmarking {label}: {command_to_str(cmd)}")

    result = {
        "label": label,
        "command": cmd,
        "command_str": command_to_str(cmd),
        "repeats": repeats,
        "timeout": timeout,
        "pre_snapshot": get_system_load_snapshot(),
        "pre_gpu_info": get_gpu_info(),
        "runs": [],
    }

    for i in range(repeats):
        print(f"[INFO] {label} run {i+1}/{repeats}")
        run_res = run_subprocess(cmd, timeout=timeout)
        run_res["run_id"] = i + 1
        result["runs"].append(run_res)

    result["post_snapshot"] = get_system_load_snapshot()
    result["post_gpu_info"] = get_gpu_info()
    result["summary"] = summarize_runs(result["runs"])
    return result


def main():
    ensure_dir(OUT_DIR)

    meta = {
        "timestamp": now_str(),
        "benchmark_type": "practical_wall_clock_comparison",
        "note": (
            "This benchmark compares practical wall-clock cost under the "
            "standard execution environments of the two workflows, rather than "
            "a hardware-normalized algorithmic benchmark on identical devices."
        ),
        "system": {
            "cpu_info": get_cpu_info(),
            "gpu_info_initial": get_gpu_info(),
        },
    }

    hf_result = benchmark_command(
        label="high_fidelity",
        cmd=HF_COMMAND,
        repeats=HF_REPEATS,
        timeout=HF_TIMEOUT,
    )

    nn_result = benchmark_command(
        label="neural_surrogate",
        cmd=NN_COMMAND,
        repeats=NN_REPEATS,
        timeout=NN_TIMEOUT,
    )

    comparison = {
        "hf_mean_sec": hf_result["summary"]["mean_sec"],
        "nn_mean_sec": nn_result["summary"]["mean_sec"],
        "speedup_hf_over_nn": None,
    }

    if hf_result["summary"]["mean_sec"] is not None and nn_result["summary"]["mean_sec"] is not None:
        if nn_result["summary"]["mean_sec"] > 0:
            comparison["speedup_hf_over_nn"] = float(
                hf_result["summary"]["mean_sec"] / nn_result["summary"]["mean_sec"]
            )

    output = {
        "meta": meta,
        "high_fidelity": hf_result,
        "neural_surrogate": nn_result,
        "comparison": comparison,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved benchmark to:", OUTPUT_JSON)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()