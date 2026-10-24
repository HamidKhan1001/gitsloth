"""Turn groups of units into a chain of commits, entirely inside object storage and a scratch index.

The invariant: the tree after the last group must be byte-identical to the working-tree snapshot. If it is not,
nothing is written to any ref.
"""
from __future__ import annotations

from collections import defaultdict

from .gitio import git, temp_index, text
from .units import ZERO, Unit, lines_of


class InvariantError(Exception):
    pass


def render(base_lines: list[bytes], hunks: list[Unit], applied: set[int]) -> bytes:
    """File content with exactly the `applied` hunks of this file applied to the base text."""
    out: list[bytes] = []
    pos = 0
    for h in sorted(hunks, key=lambda h: h.start0):
        out.extend(base_lines[pos : h.start0])
        out.extend(h.new_lines if h.idx in applied else h.old_lines)
        pos = h.start0 + h.oc
    out.extend(base_lines[pos:])
    return b"".join(out)


def build_trees(repo: str, base_tree: str, work_tree: str, units: list[Unit], groups: list[list[int]]) -> list[str]:
    """Cumulative tree after each group. Raises InvariantError unless the last tree equals work_tree."""
    flat = sorted(i for g in groups for i in g)
    if flat != list(range(len(units))):
        raise InvariantError("groups must cover every unit exactly once")
    by_file: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        if u.kind == "hunk":
            by_file[u.path].append(u)
    base_lines = {p: lines_of(git(repo, "cat-file", "blob", f"{base_tree}:{p}")) for p in by_file}
    modes = {p: text(repo, "ls-tree", base_tree, "--", p).split()[0] for p in by_file}
    applied: set[int] = set()
    trees: list[str] = []
    with temp_index(repo) as idx:
        git(repo, "read-tree", base_tree, index=idx)
        for g in groups:
            entries: list[bytes] = []
            touched: set[str] = set()
            for i in g:
                u = units[i]
                applied.add(i)
                if u.kind == "hunk":
                    touched.add(u.path)
                elif u.new is None:
                    entries.append(f"0 {ZERO}\t{u.path}".encode("utf-8", "surrogateescape") + b"\0")
                else:
                    entries.append(f"{u.new[0]} {u.new[1]}\t{u.path}".encode("utf-8", "surrogateescape") + b"\0")
            for p in touched:
                blob = git(repo, "hash-object", "-w", "--stdin", input=render(base_lines[p], by_file[p], applied)).decode().strip()
                entries.append(f"{modes[p]} {blob}\t{p}".encode("utf-8", "surrogateescape") + b"\0")
            # removals first, so a file->directory change cannot collide with the new paths
            entries.sort(key=lambda e: not e.startswith(b"0 "))
            git(repo, "update-index", "--add", "--replace", "-z", "--index-info", input=b"".join(entries), index=idx)
            trees.append(text(repo, "write-tree", index=idx))
    if trees[-1] != work_tree:
        raise InvariantError(f"final tree {trees[-1][:10]} != working snapshot {work_tree[:10]}; refusing to write anything")
    return trees


def commit_chain(repo: str, parent: str, trees: list[str], messages: list[str]) -> list[str]:
    commits = []
    for t, m in zip(trees, messages):
        parent = text(repo, "commit-tree", t, "-p", parent, "-m", m)
        commits.append(parent)
    return commits
