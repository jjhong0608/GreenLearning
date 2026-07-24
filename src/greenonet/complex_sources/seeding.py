from __future__ import annotations

import numpy as np


SPLIT_IDS: dict[str, int] = {"train": 0, "valid": 1, "test": 2}


def derive_indexed_seed(base_seed: int, split: str, index: int) -> int:
    if split not in SPLIT_IDS:
        raise ValueError(f"Unknown split: {split}")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative.")
    if index < 0:
        raise ValueError("sample index must be non-negative.")
    sequence = np.random.SeedSequence([base_seed, SPLIT_IDS[split], index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
