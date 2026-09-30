from __future__ import annotations

import re
from typing import Any


CHAIN_QA_VERSION_RE = re.compile(r"^chain_qa\.v(?P<major>\d+)(?:_(?P<minor>\d+))?$")


def parse_chain_qa_version(value: str | None) -> tuple[int, int] | None:
    match = CHAIN_QA_VERSION_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    return int(match.group("major")), int(match.group("minor") or 0)


def dataset_version_sort_key(value: str | None) -> tuple[Any, ...]:
    parsed = parse_chain_qa_version(value)
    if parsed is not None:
        return (1, parsed[0], parsed[1], "")
    return (0, 0, 0, str(value or "").lower())


def chain_qa_version_at_least(value: str | None, major: int, minor: int = 0) -> bool:
    parsed = parse_chain_qa_version(value)
    return parsed is not None and parsed >= (major, minor)


def chain_qa_versions_compatible(left: str | None, right: str | None) -> bool:
    left_version = parse_chain_qa_version(left)
    right_version = parse_chain_qa_version(right)
    return (
        left_version is not None
        and right_version is not None
        and left_version[0] == right_version[0]
    )
