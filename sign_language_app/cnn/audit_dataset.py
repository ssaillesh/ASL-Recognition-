from __future__ import annotations

import argparse
import csv
from collections import Counter
from typing import List, Tuple

import numpy as np


def load_csv_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    labels: List[str] = []
    vectors: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(str(row[0]))
            vectors.append([float(v) for v in row[1:]])
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit landmark CSV quality")
    parser.add_argument("--dataset", default="data/landmarks_from_public_kaggle.csv")
    args = parser.parse_args()

    X, y = load_csv_dataset(args.dataset)
    counts = Counter(y.tolist())

    print(f"[audit] samples={len(y)} features={X.shape[1]}")
    print(f"[audit] unique_labels={len(counts)}")
    print(f"[audit] NaN_count={int(np.isnan(X).sum())} Inf_count={int(np.isinf(X).sum())}")
    print(
        f"[audit] value_range min={float(np.min(X)):.4f} max={float(np.max(X)):.4f} "
        f"mean={float(np.mean(X)):.4f} std={float(np.std(X)):.4f}"
    )

    max_count = max(counts.values()) if counts else 0
    min_count = min(counts.values()) if counts else 0
    imbalance_ratio = (max_count / min_count) if min_count else 0.0
    print(f"[audit] class imbalance ratio (max/min)={imbalance_ratio:.3f}")

    print("[audit] per-class counts:")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
