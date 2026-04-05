from __future__ import annotations

import argparse
import csv
from typing import List

import numpy as np



def _load_dataset_vectors(path: str) -> np.ndarray:
    vectors: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            vectors.append([float(v) for v in row[1:]])
    return np.asarray(vectors, dtype=np.float32)


def _stats(arr: np.ndarray) -> str:
    return (
        f"min={float(np.min(arr)):.4f} max={float(np.max(arr)):.4f} "
        f"mean={float(np.mean(arr)):.4f} std={float(np.std(arr)):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training vs runtime feature distributions")
    parser.add_argument("--dataset", default="data/landmarks_from_public_kaggle.csv")
    parser.add_argument(
        "--runtime-npy",
        default="",
        help="Path to a .npy array captured from runtime feature vectors (shape [N,42] or [N,63])",
    )
    args = parser.parse_args()

    train = _load_dataset_vectors(args.dataset)
    print(f"[compare] train shape={train.shape} {_stats(train)}")

    if not args.runtime_npy:
        print("[compare] runtime file not provided. Run with --runtime-npy to compare live features.")
        return

    runtime = np.load(args.runtime_npy)
    runtime = np.asarray(runtime, dtype=np.float32)
    print(f"[compare] runtime shape={runtime.shape} {_stats(runtime)}")

    train_q = np.quantile(train, [0.01, 0.05, 0.5, 0.95, 0.99])
    run_q = np.quantile(runtime, [0.01, 0.05, 0.5, 0.95, 0.99])
    print(f"[compare] train quantiles={np.round(train_q, 4).tolist()}")
    print(f"[compare] runtime quantiles={np.round(run_q, 4).tolist()}")


if __name__ == "__main__":
    main()
