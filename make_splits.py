from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


SPLIT_NAMES = ("train", "valid", "test")


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            _clear_directory(item)
            item.rmdir()


def _validate_ratios(
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    ratios = (train_ratio, valid_ratio, test_ratio)
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0 (got train+valid+test={total:.12g})."
        )
    return ratios


def _split_counts(
    total_files: int, ratios: tuple[float, float, float]
) -> dict[str, int]:
    raw_counts = [total_files * ratio for ratio in ratios]
    counts = [math.floor(raw_count) for raw_count in raw_counts]
    remaining = total_files - sum(counts)
    remainders = [
        (raw_counts[idx] - counts[idx], idx)
        for idx, ratio in enumerate(ratios)
        if ratio > 0.0
    ]
    remainders.sort(key=lambda item: (-item[0], item[1]))

    for idx in range(remaining):
        _remainder, split_idx = remainders[idx % len(remainders)]
        counts[split_idx] += 1

    return dict(zip(SPLIT_NAMES, counts))


def split_train_valid_test(
    source_dir: Path,
    out_root: Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    files = sorted(source_dir.glob("*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in {source_dir}")

    ratios = _validate_ratios(train_ratio, valid_ratio, test_ratio)
    counts = _split_counts(len(files), ratios)

    rng = random.Random(seed)
    rng.shuffle(files)

    out_root.mkdir(parents=True, exist_ok=True)
    split_dirs = {name: out_root / name for name in SPLIT_NAMES}
    for split_dir in split_dirs.values():
        _clear_directory(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    start = 0
    groups: dict[str, list[Path]] = {}
    for name in SPLIT_NAMES:
        end = start + counts[name]
        groups[name] = files[start:end]
        start = end

    for name, group in groups.items():
        for src in group:
            dest = split_dirs[name] / src.name
            dest.symlink_to(src.resolve())

    print(
        "Split: "
        f"train={len(groups['train'])} "
        f"valid={len(groups['valid'])} "
        f"test={len(groups['test'])} -> {out_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/valid/test symlinks from .npz files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("2D_data_variable"),
        help="Directory containing .npz files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("split"),
        help="Output directory for train/valid/test splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of files assigned to train.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of files assigned to valid.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of files assigned to test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = parser.parse_args()
    split_train_valid_test(
        source_dir=args.source,
        out_root=args.out,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
