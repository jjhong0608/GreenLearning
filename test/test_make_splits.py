from pathlib import Path

import pytest

from make_splits import split_train_valid_test


def _write_npz_files(source_dir: Path, count: int) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for idx in range(count):
        path = source_dir / f"sample_{idx:03d}.npz"
        path.write_bytes(b"npz-placeholder")
        files.append(path)
    return files


def _split_names(out_root: Path, split_name: str) -> list[str]:
    return sorted(path.name for path in (out_root / split_name).iterdir())


def _all_split_names(out_root: Path) -> set[str]:
    names: set[str] = set()
    for split_name in ("train", "valid", "test"):
        names.update(_split_names(out_root, split_name))
    return names


def test_split_train_valid_test_creates_expected_symlink_counts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    files = _write_npz_files(source_dir, count=10)
    out_root = tmp_path / "split"

    split_train_valid_test(
        source_dir=source_dir,
        out_root=out_root,
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )

    assert len(_split_names(out_root, "train")) == 6
    assert len(_split_names(out_root, "valid")) == 2
    assert len(_split_names(out_root, "test")) == 2
    assert _all_split_names(out_root) == {path.name for path in files}
    for split_name in ("train", "valid", "test"):
        for item in (out_root / split_name).iterdir():
            assert item.is_symlink()
            assert item.resolve() in files


def test_split_train_valid_test_allows_zero_ratio(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_npz_files(source_dir, count=5)
    out_root = tmp_path / "split"

    split_train_valid_test(
        source_dir=source_dir,
        out_root=out_root,
        train_ratio=0.8,
        valid_ratio=0.2,
        test_ratio=0.0,
    )

    assert len(_split_names(out_root, "train")) == 4
    assert len(_split_names(out_root, "valid")) == 1
    assert (out_root / "test").is_dir()
    assert _split_names(out_root, "test") == []


def test_split_train_valid_test_uses_largest_remainder_counts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    files = _write_npz_files(source_dir, count=7)
    out_root = tmp_path / "split"

    split_train_valid_test(
        source_dir=source_dir,
        out_root=out_root,
        train_ratio=0.5,
        valid_ratio=0.25,
        test_ratio=0.25,
    )

    assert len(_split_names(out_root, "train")) == 3
    assert len(_split_names(out_root, "valid")) == 2
    assert len(_split_names(out_root, "test")) == 2
    assert _all_split_names(out_root) == {path.name for path in files}


def test_split_train_valid_test_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_npz_files(source_dir, count=12)
    out_a = tmp_path / "split_a"
    out_b = tmp_path / "split_b"

    split_train_valid_test(source_dir=source_dir, out_root=out_a, seed=123)
    split_train_valid_test(source_dir=source_dir, out_root=out_b, seed=123)

    assert {
        split_name: _split_names(out_a, split_name)
        for split_name in ("train", "valid", "test")
    } == {
        split_name: _split_names(out_b, split_name)
        for split_name in ("train", "valid", "test")
    }


def test_split_train_valid_test_replaces_existing_split_contents(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    _write_npz_files(source_dir, count=5)
    out_root = tmp_path / "split"
    stale_file = out_root / "train" / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale")

    split_train_valid_test(
        source_dir=source_dir,
        out_root=out_root,
        train_ratio=1.0,
        valid_ratio=0.0,
        test_ratio=0.0,
    )

    assert not stale_file.exists()
    assert len(_split_names(out_root, "train")) == 5
    assert _split_names(out_root, "valid") == []
    assert _split_names(out_root, "test") == []


def test_split_train_valid_test_rejects_empty_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No .npz files"):
        split_train_valid_test(source_dir=tmp_path, out_root=tmp_path / "split")


def test_split_train_valid_test_rejects_ratios_that_do_not_sum_to_one(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    _write_npz_files(source_dir, count=3)

    with pytest.raises(ValueError, match="sum to 1.0"):
        split_train_valid_test(
            source_dir=source_dir,
            out_root=tmp_path / "split",
            train_ratio=0.5,
            valid_ratio=0.3,
            test_ratio=0.3,
        )


def test_split_train_valid_test_rejects_negative_ratios(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_npz_files(source_dir, count=3)

    with pytest.raises(ValueError, match="non-negative"):
        split_train_valid_test(
            source_dir=source_dir,
            out_root=tmp_path / "split",
            train_ratio=0.8,
            valid_ratio=-0.1,
            test_ratio=0.3,
        )
