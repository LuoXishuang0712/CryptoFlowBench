from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
SELECTED_CASES_PATH = PROJECT_ROOT / "selected_cases.txt"


def read_case_list(path: Path = SELECTED_CASES_PATH) -> list[str]:
    """Read the canonical mainline case directories in declared order."""
    values = [
        value
        for raw in path.read_text(encoding="utf-8").splitlines()
        if (value := raw.split("#", 1)[0].strip())
    ]
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate case directories in {path}: {', '.join(duplicates)}")
    return values


def selected_case_set(path: Path = SELECTED_CASES_PATH) -> set[str]:
    return set(read_case_list(path))


def filter_selected_cases(
    case_dirs: Iterable[str],
    *,
    path: Path = SELECTED_CASES_PATH,
) -> list[str]:
    """Return selected cases in selected_cases.txt order, not discovery order."""
    available = set(case_dirs)
    return [case_dir for case_dir in read_case_list(path) if case_dir in available]


def discover_selected_cases(
    root: Path,
    *,
    required_files: Iterable[str] = (),
    path: Path = SELECTED_CASES_PATH,
) -> list[str]:
    required = tuple(required_files)
    return [
        case_dir
        for case_dir in read_case_list(path)
        if (root / case_dir).is_dir()
        and all((root / case_dir / filename).is_file() for filename in required)
    ]


def ensure_selected_case(case_dir: str, path: Path = SELECTED_CASES_PATH) -> str:
    if case_dir not in selected_case_set(path):
        raise ValueError(
            f"{case_dir!r} is not a mainline case in {path.name}; "
            "move it into the selected list before using it in aggregate workflows"
        )
    return case_dir
