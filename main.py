from __future__ import annotations

import argparse

from case_selection import read_case_list


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the active onchain-case workflow and canonical evaluation cases."
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print only the selected case directories, one per line.",
    )
    args = parser.parse_args()
    cases = read_case_list()
    if args.list_cases:
        print("\n".join(cases))
        return 0
    print("onchain-case: evidence -> KGQA/RAG + chain_qa.v3_1 -> agent/baseline evaluation")
    print(f"mainline cases: {len(cases)} (selected_cases.txt)")
    print("use --list-cases to print the canonical evaluation set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
