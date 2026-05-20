import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def query_gpu_memory() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    gpus = []
    for line in output.strip().splitlines():
        index, name, used, total = [part.strip() for part in line.split(",", 3)]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": int(used),
                "memory_total_mb": int(total),
            }
        )
    return gpus


def parse_args():
    parser = argparse.ArgumentParser(description="Run a command and record peak GPU memory usage with nvidia-smi.")
    parser.add_argument("--output_path", required=True, help="Where to save the monitoring json.")
    parser.add_argument("--cwd", default=None, help="Optional working directory for the child process.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds. Smaller values are more accurate but noisier.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Put it after --, for example: -- python train.py",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("You must provide a command after --")
    return args


def main():
    args = parse_args()
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found in PATH, cannot monitor GPU memory.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now().isoformat(timespec="seconds")
    process = subprocess.Popen(args.command, cwd=args.cwd)

    peak_by_gpu = {}
    samples = []

    try:
        while True:
            gpu_stats = query_gpu_memory()
            timestamp = datetime.now().isoformat(timespec="seconds")
            samples.append({"time": timestamp, "gpus": gpu_stats})

            for gpu in gpu_stats:
                gpu_index = gpu["index"]
                current_peak = peak_by_gpu.get(gpu_index)
                if current_peak is None or gpu["memory_used_mb"] > current_peak["memory_used_mb"]:
                    peak_by_gpu[gpu_index] = {
                        "index": gpu_index,
                        "name": gpu["name"],
                        "memory_used_mb": gpu["memory_used_mb"],
                        "memory_total_mb": gpu["memory_total_mb"],
                        "time": timestamp,
                    }

            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(args.interval)
    finally:
        if process.poll() is None:
            process.terminate()

    summary = {
        "command": args.command,
        "cwd": args.cwd,
        "start_time": start_time,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "interval_seconds": args.interval,
        "return_code": process.returncode,
        "peak_memory_by_gpu_mb": [peak_by_gpu[idx] for idx in sorted(peak_by_gpu)],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": samples}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")

    if process.returncode != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
