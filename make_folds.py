from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List


def split_folds(
    source_dir: Path, out_root: Path, n_folds: int = 5, seed: int = 42
) -> None:
    files: List[Path] = sorted(source_dir.glob("*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in {source_dir}")
    if len(files) % n_folds != 0:
        raise ValueError(
            f"File count ({len(files)}) not divisible by folds ({n_folds}); expected 150 for 5-fold."
        )

    random.seed(seed)
    random.shuffle(files)

    fold_size = len(files) // n_folds
    out_root.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(n_folds):
        fold_dir = out_root / f"fold{fold_idx}"
        train_dir = fold_dir / "train"
        valid_dir = fold_dir / "valid"
        for d in (train_dir, valid_dir):
            if d.exists():
                # remove old symlinks/dirs
                for item in d.iterdir():
                    if item.is_symlink() or item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        for sub in item.rglob("*"):
                            if sub.is_symlink() or sub.is_file():
                                sub.unlink()
                        sub_dirs = sorted(
                            (p for p in item.rglob("*") if p.is_dir()), reverse=True
                        )
                        for sd in sub_dirs:
                            sd.rmdir()
                        item.rmdir()
            d.mkdir(parents=True, exist_ok=True)

        val_start = fold_idx * fold_size
        val_files = files[val_start : val_start + fold_size]
        train_files = [f for f in files if f not in val_files]

        for target_dir, group in ((train_dir, train_files), (valid_dir, val_files)):
            for src in group:
                rel = src.relative_to(source_dir)
                dest = target_dir / rel.name
                dest.symlink_to(src.resolve())

        print(
            f"Fold {fold_idx}: train={len(train_files)} valid={len(val_files)} -> {fold_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 5-fold CV symlinks.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("2D_data_variable"),
        help="Directory containing .npz files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("5_fold"),
        help="Output directory for folds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = parser.parse_args()
    split_folds(args.source, args.out, n_folds=5, seed=args.seed)


if __name__ == "__main__":
    main()
