#!/usr/bin/env python3
"""Validate that all file paths referenced in CLAUDE.md actually exist.

Usage:
    python scripts/validate_claude_md_paths.py [--fix]

Checks backtick-quoted paths like `doc/csl/language/dsds.md` and
`tut/gemv-01-complete-program/layout.csl` against the actual filesystem.

Exit code:
    0 — all paths valid
    1 — some paths invalid (printed to stderr)
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path prefix → actual directory mapping (matching CLAUDE.md shorthand)
PREFIX_MAP = {
    "doc/": "Cerebras_Docs/",
    "tut/": "examples/tutorials/",
    "bench/": "examples/benchmarks/",
    "lib/": "source-code/csl_libraries/",
    "kb/": "AI-Knowledge-Base/",
}


def expand_path(short_path: str) -> str:
    """Expand CLAUDE.md shorthand path to actual filesystem path."""
    for prefix, expansion in PREFIX_MAP.items():
        if short_path.startswith(prefix):
            return os.path.join(PROJECT_ROOT, expansion + short_path[len(prefix):])
    # Try as-is (relative to project root)
    return os.path.join(PROJECT_ROOT, short_path)


def extract_paths(content: str) -> list[tuple[int, str]]:
    """Extract backtick-quoted paths from markdown content.
    
    Returns list of (line_number, path_string) tuples.
    """
    results = []
    for i, line in enumerate(content.split("\n"), 1):
        # Match backtick-quoted paths that look like file references
        for m in re.finditer(r'`([^`]+\.[a-z]{1,4}(?:/[^`]*)?)`', line):
            path = m.group(1)
            # Filter out things that aren't paths
            if path.startswith("@") or path.startswith("<") or "(" in path:
                continue
            if "/" not in path and not path.endswith(".md"):
                continue
            results.append((i, path))
        # Also match directory paths like `tut/gemv-01*/`
        for m in re.finditer(r'`((?:doc|tut|bench|lib|kb)/[^`]+/)`', line):
            path = m.group(1).rstrip("*")
            results.append((i, path))
    return results


def main():
    claude_md = os.path.join(PROJECT_ROOT, "CLAUDE.md")
    if not os.path.exists(claude_md):
        print("ERROR: CLAUDE.md not found", file=sys.stderr)
        return 1

    with open(claude_md, "r") as f:
        content = f.read()

    paths = extract_paths(content)
    invalid = []
    valid_count = 0

    for line_num, short_path in paths:
        expanded = expand_path(short_path)
        # Allow glob-style paths (e.g., `tut/gemv-01*/`)
        if "*" in expanded:
            import glob
            if not glob.glob(expanded):
                invalid.append((line_num, short_path, expanded))
            else:
                valid_count += 1
        elif os.path.exists(expanded):
            valid_count += 1
        else:
            invalid.append((line_num, short_path, expanded))

    print(f"Checked {len(paths)} paths: {valid_count} valid, {len(invalid)} invalid")

    if invalid:
        print("\nInvalid paths:", file=sys.stderr)
        for line_num, short_path, expanded in invalid:
            print(f"  Line {line_num}: `{short_path}` → {expanded}", file=sys.stderr)
        return 1

    print("All paths valid ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
