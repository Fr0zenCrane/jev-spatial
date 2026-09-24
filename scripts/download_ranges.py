"""Resume public, pinned large files in bounded HTTP ranges and verify SHA-256."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--range-workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (root / ".asset-download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(args.manifest.read_text())
    tasks = []
    for source in manifest["sources"]:
        folder = root / ("models" if source["kind"] == "model" else "data/raw") / source["repo"]
        for file in source["files"]:
            path = folder / file["path"]
            if path.exists() and path.stat().st_size == file["bytes"]:
                continue
            tasks.append((source, file, folder))

    def download(task):
        source, file, folder = task
        path = folder / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        if not partial.exists() and file.get("sha256"):
            cached = list((folder / ".cache/huggingface").rglob(f"*.{file['sha256']}.incomplete"))
            if cached:
                max(cached, key=lambda p: p.stat().st_size).replace(partial)
        prefix = "datasets/" if source["kind"] == "dataset" else ""
        url = (f"https://huggingface.co/{prefix}{source['repo']}/resolve/"
               f"{source['revision']}/{quote(file['path'])}")
        size = file["bytes"]
        parts = path.parent / (".parts-" + path.name)
        parts.mkdir(exist_ok=True)
        partial.touch(exist_ok=True)
        first = partial.stat().st_size
        span = 16 * 1024 * 1024

        def fetch_range(start):
            end = min(size - 1, start + span - 1)
            target = parts / f"{start:016d}.chunk"
            failures = 0
            with requests.Session() as session, target.open("ab", buffering=0) as stream:
                while start + stream.tell() <= end:
                    offset = start + stream.tell()
                    try:
                        with session.get(url, headers={"Range": f"bytes={offset}-{end}"},
                                         timeout=(15, 40), stream=True) as response:
                            response.raise_for_status()
                            expected = f"bytes {offset}-{end}/{size}"
                            if (response.status_code != 206
                                    or response.headers.get("Content-Range") != expected):
                                raise ValueError("Server did not honor requested byte range")
                            for chunk in response.iter_content(1024 * 1024):
                                if start + stream.tell() + len(chunk) > end + 1:
                                    raise ValueError("Range response overflow")
                                stream.write(chunk)
                        failures = 0
                    except (requests.RequestException, ValueError) as exc:
                        failures = failures + 1 if start + stream.tell() == offset else 0
                        print("retry", source["repo"], file["path"], type(exc).__name__,
                              "offset", start + stream.tell(), flush=True)
                        if failures >= 12:
                            raise RuntimeError("Repeated range failure without progress") from exc
                        time.sleep(min(2 ** failures, 20))
            if target.stat().st_size != end - start + 1:
                raise ValueError("Incorrect segment size")
            return start, target

        completed = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.range_workers) as ranges:
            pending_ranges = [ranges.submit(fetch_range, start) for start in range(first, size, span)]
            with partial.open("ab", buffering=0) as stream:
                for future in concurrent.futures.as_completed(pending_ranges):
                    start, chunk_path = future.result()
                    completed[start] = chunk_path
                    while stream.tell() in completed:
                        chunk_path = completed.pop(stream.tell())
                        stream.write(chunk_path.read_bytes())
                        chunk_path.unlink()
        if partial.stat().st_size != size:
            raise ValueError(f"Wrong size: {path}")
        if file.get("sha256") and sha256(partial) != file["sha256"]:
            raise ValueError(f"SHA-256 mismatch, partial file retained: {path}")
        partial.replace(path)
        print("verified", source["repo"], file["path"], size, flush=True)
        return {"repo": source["repo"], "path": file["path"], "bytes": size,
                "sha256": file.get("sha256")}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(download, task) for task in tasks]
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=10)
            for future in done:
                results.append(future.result())
            present = 0
            for source in manifest["sources"]:
                folder = root / ("models" if source["kind"] == "model" else "data/raw") / source["repo"]
                for file in source["files"]:
                    path = folder / file["path"]
                    partial = path.with_name(path.name + ".partial")
                    if path.exists():
                        present += min(path.stat().st_size, file["bytes"])
                    elif partial.exists():
                        current = partial.stat().st_size
                        present += current
                        for chunk in (path.parent / (".parts-" + path.name)).glob("*.chunk"):
                            if int(chunk.stem) >= current:
                                try:
                                    present += chunk.stat().st_size
                                except FileNotFoundError:
                                    pass
            status = {"state": "running" if pending else "complete", "bytes_present": present,
                      "total_bytes": manifest["total_bytes"], "files_remaining": len(pending)}
            temporary = args.run_dir / "status.json.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(args.run_dir / "status.json")
            print(f"{present / 1e9:.3f}/{manifest['total_bytes'] / 1e9:.3f} GB", flush=True)
    (args.run_dir / "COMPLETE.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
