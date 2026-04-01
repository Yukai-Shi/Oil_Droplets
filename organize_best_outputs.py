import re
import shutil
from pathlib import Path


ROOT = Path("models") / "best"


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    k = 1
    while True:
        cand = parent / f"{stem}_dup{k}{suffix}"
        if not cand.exists():
            return cand
        k += 1


def infer_tag_for_best(name_stem: str) -> tuple[str, str] | None:
    # name_stem examples:
    # best_free_layout_inflow_square_t100000_trajectory
    # best_t4000_final
    if not name_stem.startswith("best_"):
        return None
    core = name_stem[len("best_") :]
    kind = None
    if core.endswith("_trajectory"):
        core = core[: -len("_trajectory")]
        kind = "gifs"
    elif core.endswith("_final"):
        core = core[: -len("_final")]
        kind = "frames"
    else:
        return None

    m = re.search(r"_t\d+$", core)
    if m:
        tag = core[: m.start()]
    else:
        tag = core

    if not tag:
        tag = "legacy_unknown"
    return tag, kind


def infer_target(file: Path) -> Path | None:
    stem = file.stem
    ext = file.suffix.lower()
    if ext not in {".gif", ".png"}:
        return None

    if stem.startswith("long_rollout_"):
        tag = stem[len("long_rollout_") :]
        if not tag:
            tag = "legacy_unknown"
        return ROOT / tag / "rollout" / file.name

    parsed = infer_tag_for_best(stem)
    if parsed is not None:
        tag, kind = parsed
        return ROOT / tag / kind / file.name

    return ROOT / "legacy_misc" / ("gifs" if ext == ".gif" else "frames") / file.name


def main():
    if not ROOT.exists():
        print(f"[Skip] {ROOT} not found.")
        return

    moved = 0
    for f in ROOT.iterdir():
        if not f.is_file():
            continue
        target = infer_target(f)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target = ensure_unique(target)
        shutil.move(str(f), str(target))
        moved += 1
        print(f"[Move] {f.name} -> {target.relative_to(ROOT)}")

    print(f"[Done] moved_files={moved}")


if __name__ == "__main__":
    main()
