"""
Shared utilities for the anonymization time-analysis benchmark.

Used by both stages (diarize.py runs in the DiariZen env, anonymize.py in the
Seed-VC env), so this module only depends on the Python stdlib plus torch /
soundfile, which both envs have.
"""
import os
import sys
import json
import time
import platform
import socket
import subprocess
import datetime as _dt
from pathlib import Path


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class Timer:
    """Context manager: accumulates wall seconds into store[key] (GPU-synchronised)."""

    def __init__(self, store, key, sync=True):
        self.store, self.key, self.sync = store, key, sync

    def __enter__(self):
        if self.sync:
            cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync:
            cuda_sync()
        self.store[self.key] = self.store.get(self.key, 0.0) + (time.perf_counter() - self.t0)
        return False


def now_iso():
    return _dt.datetime.now().isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------- #
# Environment / hardware description
# --------------------------------------------------------------------------- #
def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True, timeout=20).strip()
    except Exception:
        return ""


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor()


def env_info():
    info = {
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": cpu_model(),
        "cpu_count_visible": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _sh("nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader"),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        })
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": p.name,
                "gpu_total_mem_gb": round(p.total_memory / 2**30, 2),
                "gpu_sm_count": p.multi_processor_count,
                "gpu_capability": f"{p.major}.{p.minor}",
            })
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    return info


def gpu_peak_mem_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return None


def gpu_reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def log(msg):
    print(f"[{now_iso()}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# File selection
# --------------------------------------------------------------------------- #
def list_wavs(wav_dir, limit=None, shard=None, files=None):
    """Deterministic (sorted) list of wav paths, optional --limit and --shard i/N."""
    wav_dir = Path(wav_dir)
    if files:
        paths = []
        for line in open(files):
            name = line.strip()
            if not name:
                continue
            paths.append(wav_dir / (name if name.endswith(".wav") else name + ".wav"))
    else:
        paths = sorted(wav_dir.glob("*.wav"))
    if shard:
        i, n = [int(x) for x in shard.split("/")]
        paths = [p for k, p in enumerate(paths) if k % n == i]
    if limit:
        paths = paths[:limit]
    return paths


def wav_duration_s(path):
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate, info.channels


# --------------------------------------------------------------------------- #
# RTTM helpers (no pyannote dependency, so the Seed-VC env can use them)
# --------------------------------------------------------------------------- #
def read_rttm(path):
    """Return list of dicts {start, end, spk} sorted by start."""
    segs = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        if dur <= 0:
            continue
        segs.append({"start": start, "end": start + dur, "spk": spk})
    segs.sort(key=lambda s: s["start"])
    return segs


def find_rttm(rttm_dir, file_id):
    for cand in (f"{file_id}.rttm", f"{file_id}_SPEAKER.rttm"):
        p = Path(rttm_dir) / cand
        if p.exists():
            return p
    return None


def merge_segments(segs, gap=0.3, pad=0.1, min_len=0.5, total_dur=None):
    """
    Per speaker: merge segments separated by <= gap seconds, then pad each
    merged segment by `pad` and grow tiny ones to `min_len`. Returns dict
    spk -> list of (start, end) clipped to [0, total_dur].
    """
    by_spk = {}
    for s in segs:
        by_spk.setdefault(s["spk"], []).append((s["start"], s["end"]))
    out = {}
    for spk, lst in by_spk.items():
        lst.sort()
        merged = []
        for a, b in lst:
            if merged and a - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        fixed = []
        for a, b in merged:
            a, b = a - pad, b + pad
            if b - a < min_len:
                c = (a + b) / 2
                a, b = c - min_len / 2, c + min_len / 2
            a = max(0.0, a)
            if total_dur is not None:
                b = min(total_dur, b)
            if b > a:
                fixed.append((a, b))
        out[spk] = fixed
    return out


def union_seconds(intervals):
    """Total covered seconds of a list of (start, end)."""
    total, cur = 0.0, None
    for a, b in sorted(intervals):
        if cur is None or a > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        total += cur[1] - cur[0]
    return total
