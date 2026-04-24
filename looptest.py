import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np


KERNEL_SOURCE = r"""
__global__ void burn_gpu(float *out, int inner_iters) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)(idx % 1024) * 0.0001f;

    for (int i = 0; i < inner_iters; ++i) {
        x = __sinf(x + 0.000001f * (float)i) * 1.000001f + x;
    }

    out[idx] = x;
}
"""


def find_nvidia_smi():
    path = shutil.which("nvidia-smi")
    if path:
        return path

    windows_path = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    return windows_path if os.path.exists(windows_path) else None


def query_nvidia_smi(nvidia_smi):
    if not nvidia_smi:
        return "nvidia-smi unavailable"

    base_cmd = [
        nvidia_smi,
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    fallback_cmd = [
        nvidia_smi,
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    for cmd in (base_cmd, fallback_cmd):
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return " | ".join(format_nvidia_smi_line(line) for line in result.stdout.splitlines())
        except (subprocess.SubprocessError, OSError):
            continue

    return "nvidia-smi query failed"


def format_nvidia_smi_line(line):
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 7:
        return line.strip()

    index, name, gpu_util, mem_util, mem_used, mem_total, temp, *rest = fields
    power = f" power={rest[0]}W" if rest else ""
    return (
        f"gpu{index} {name} util={gpu_util}% mem_util={mem_util}% "
        f"mem={mem_used}/{mem_total}MiB temp={temp}C{power}"
    )


def format_cuda_memory(drv):
    free_bytes, total_bytes = drv.mem_get_info()
    used_mb = (total_bytes - free_bytes) / 1024 / 1024
    total_mb = total_bytes / 1024 / 1024
    return f"cuda_mem={used_mb:.0f}/{total_mb:.0f} MiB"


def progress_bar(current, total, width=28):
    if total <= 0:
        return "[" + "#" * width + "]"

    filled = int(width * current / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple PyCUDA GPU utilization test")
    parser.add_argument("--launches", type=int, default=200, help="kernel launches to run")
    parser.add_argument("--duration-sec", type=float, default=0, help="run until this duration is reached")
    parser.add_argument("--blocks", type=int, default=4096, help="CUDA blocks per launch")
    parser.add_argument("--threads", type=int, default=256, help="CUDA threads per block")
    parser.add_argument("--inner-iters", type=int, default=20000, help="work per GPU thread")
    parser.add_argument("--stats-every", type=int, default=1, help="print every N launches")
    return parser.parse_args()


def import_pycuda():
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as drv
        from pycuda.compiler import SourceModule
    except ModuleNotFoundError as exc:
        missing = exc.name or "pycuda"
        print(f"Missing Python dependency: {missing}", file=sys.stderr)
        print("Install PyCUDA in this Python environment before running the CUDA test.", file=sys.stderr)
        raise SystemExit(1) from exc

    return drv, SourceModule


def main():
    args = parse_args()
    drv, SourceModule = import_pycuda()
    total_threads = args.blocks * args.threads
    nvidia_smi = find_nvidia_smi()

    module = SourceModule(KERNEL_SOURCE)
    burn_gpu = module.get_function("burn_gpu")
    out = drv.mem_alloc(np.empty(total_threads, dtype=np.float32).nbytes)

    print("GPU utilization test started", flush=True)
    print(
        f"launches={args.launches} duration_sec={args.duration_sec or 'off'} "
        f"blocks={args.blocks} threads={args.threads} inner_iters={args.inner_iters}",
        flush=True,
    )
    print(f"stats_source={nvidia_smi or 'PyCUDA only'}", flush=True)

    start_time = time.perf_counter()
    launch = 0
    last_kernel_ms = 0.0

    try:
        while True:
            launch += 1
            start_event = drv.Event()
            end_event = drv.Event()

            start_event.record()
            burn_gpu(
                out,
                np.int32(args.inner_iters),
                block=(args.threads, 1, 1),
                grid=(args.blocks, 1),
            )
            end_event.record()
            end_event.synchronize()
            last_kernel_ms = start_event.time_till(end_event)

            elapsed = time.perf_counter() - start_time
            done_by_launches = args.duration_sec <= 0 and launch >= args.launches
            done_by_time = args.duration_sec > 0 and elapsed >= args.duration_sec

            if launch % args.stats_every == 0 or done_by_launches or done_by_time:
                total = args.launches if args.duration_sec <= 0 else max(int(args.duration_sec), 1)
                current = launch if args.duration_sec <= 0 else min(int(elapsed), total)
                pct = (launch / args.launches * 100) if args.duration_sec <= 0 else min(elapsed / args.duration_sec * 100, 100)
                print(
                    f"{progress_bar(current, total)} {pct:6.2f}% "
                    f"launch={launch} elapsed={elapsed:7.2f}s kernel={last_kernel_ms:8.2f}ms "
                    f"{format_cuda_memory(drv)} gpu=({query_nvidia_smi(nvidia_smi)})",
                    flush=True,
                )

            if done_by_launches or done_by_time:
                break
    except KeyboardInterrupt:
        print("\nStopped by user", flush=True)

    total_elapsed = time.perf_counter() - start_time
    print(f"Finished: launches={launch} elapsed={total_elapsed:.2f}s last_kernel={last_kernel_ms:.2f}ms", flush=True)


if __name__ == "__main__":
    main()
