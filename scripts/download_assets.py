"""Download pinned public assets to shared storage with resume, retries and status."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/download_files_20260923.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--fallback", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (root / ".asset-download.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    # Configure before importing the Hub. HTTP enables observable, resumable partial files.
    os.environ["HF_HOME"] = str(root / ".hf_cache")
    os.environ["HF_HUB_CACHE"] = str(root / ".hf_cache/hub")
    os.environ["HF_XET_CACHE"] = str(root / ".hf_cache/xet")
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    from huggingface_hub import hf_hub_download

    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    tasks = []
    repo_states = {}
    priorities = {"nyu-visionx/VSI-Bench": 0, "allenai/Molmo2-ER": 1}
    for source in sorted(manifest["sources"],
                         key=lambda s: (priorities.get(s["repo"], 2), s["total_bytes"])):
        destination = root / ("models" if source["kind"] == "model" else "data/raw")
        destination = destination / source["repo"]
        destination.mkdir(parents=True, exist_ok=True)
        revision_file = destination / ".download_revision.json"
        expected_revision = {"repo": source["repo"], "revision": source["revision"]}
        if revision_file.exists():
            if json.loads(revision_file.read_text()) != expected_revision:
                raise ValueError(f"Different revision already exists: {destination}")
        else:
            if any(destination.iterdir()):
                raise ValueError(f"Nonempty untracked destination: {destination}")
            atomic_json(revision_file, expected_revision)
        repo_states[source["repo"]] = {
            "destination": str(destination), "revision": source["revision"],
            "expected_bytes": source["total_bytes"], "expected_files": len(source["files"]),
            "verified_bytes": 0, "verified_files": 0, "failed_files": 0,
        }
        for file in sorted(source["files"], key=lambda f: f["bytes"]):
            relative = Path(file["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe manifest path")
            tasks.append((source, file, destination))

    atomic_json(run_dir / "config.json", {
        "started_at": utc_now(), "manifest": str(manifest_path),
        "endpoint": args.endpoint, "fallback": args.fallback,
        "workers": args.workers, "attempts": args.attempts,
        "total_bytes": manifest["total_bytes"], "total_files": len(tasks),
        "verification": "pinned revision, Hub metadata and exact file size",
    })
    state_lock = threading.Lock()
    failures = []
    events = (run_dir / "events.jsonl").open("a", buffering=1)

    def event(value):
        with state_lock:
            events.write(json.dumps({"time": utc_now(), **value}) + "\n")

    def download(task):
        source, file, destination = task
        repo = source["repo"]
        force_download = False
        for attempt in range(args.attempts):
            endpoint = args.endpoint if attempt % 2 == 0 else args.fallback
            try:
                result = hf_hub_download(
                    repo_id=repo, filename=file["path"], revision=source["revision"],
                    repo_type=source["kind"], local_dir=destination,
                    endpoint=endpoint, token=False, etag_timeout=30,
                    force_download=force_download,
                )
                actual = Path(result).stat().st_size
                if actual != file["bytes"]:
                    force_download = True
                    raise ValueError("Downloaded size does not match pinned manifest")
                with state_lock:
                    repo_states[repo]["verified_files"] += 1
                    repo_states[repo]["verified_bytes"] += actual
                event({"event": "file_complete", "repo": repo, "path": file["path"],
                       "bytes": actual, "endpoint": endpoint})
                return
            except Exception as exc:
                # Do not log credentials or signed download URLs from exception messages.
                event({"event": "retry", "repo": repo, "path": file["path"],
                       "attempt": attempt + 1, "endpoint": endpoint,
                       "error_type": type(exc).__name__})
                if attempt + 1 < args.attempts:
                    time.sleep(min(5 * 2 ** (attempt // 2), 120))
        with state_lock:
            repo_states[repo]["failed_files"] += 1
            failures.append({"repo": repo, "path": file["path"]})
        event({"event": "file_failed", "repo": repo, "path": file["path"]})

    samples = deque(maxlen=7)

    def report(final=False):
        present = 0
        for _, file, destination in tasks:
            path = destination / file["path"]
            try:
                present += min(path.stat().st_size, file["bytes"])
            except FileNotFoundError:
                pass
        partial = 0
        for state in repo_states.values():
            for path in (Path(state["destination"]) / ".cache/huggingface").rglob("*.incomplete"):
                try:
                    partial += path.stat().st_size
                except FileNotFoundError:
                    pass
        now = time.monotonic()
        samples.append((now, present + partial))
        elapsed = now - samples[0][0]
        speed = max(0, (present + partial - samples[0][1]) / elapsed) if elapsed > 0 else 0
        with state_lock:
            status = {
                "updated_at": utc_now(), "pid": os.getpid(),
                "state": ("failed" if failures else "complete") if final else "running",
                "total_bytes": manifest["total_bytes"], "bytes_present": present + partial,
                "partial_bytes": partial, "rolling_bytes_per_second": speed,
                "eta_seconds": max(0, manifest["total_bytes"] - present - partial) / speed
                if speed > 0 else None,
                "sources": repo_states, "failures": failures,
            }
            atomic_json(run_dir / "status.json", status)
        print(f'{status["updated_at"]} {status["state"]} '
              f'{(present + partial) / 1e9:.3f}/{manifest["total_bytes"] / 1e9:.3f} GB '
              f'{speed / 1e6:.2f} MB/s', flush=True)
        if final and not failures:
            atomic_json(run_dir / "COMPLETE.json", status)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, task): task for task in tasks}
        pending = set(futures)
        report()
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=10)
            for future in done:
                try:
                    future.result()
                except Exception as exc:
                    source, file, _ = futures[future]
                    with state_lock:
                        repo_states[source["repo"]]["failed_files"] += 1
                        failures.append({"repo": source["repo"], "path": file["path"]})
                    event({"event": "worker_failed", "repo": source["repo"],
                           "path": file["path"], "error_type": type(exc).__name__})
            report()
    if sum(s["verified_files"] for s in repo_states.values()) != len(tasks) and not failures:
        failures.append({"error": "Verified file count does not match manifest"})
    report(final=True)
    events.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
