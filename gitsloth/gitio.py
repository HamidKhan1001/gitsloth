"""Thin git subprocess wrapper. Every call is explicit about the repo; nothing here writes to the user's index or working tree."""
from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager


class GitError(Exception):
    pass


def git(repo: str, *args: str, input: bytes | None = None, index: str | None = None, ok: tuple[int, ...] = (0,)) -> bytes:
    env = dict(os.environ)
    if index:
        env["GIT_INDEX_FILE"] = index
    env["GIT_OPTIONAL_LOCKS"] = "0"
    p = subprocess.run(["git", "-C", repo, *args], input=input, capture_output=True, env=env)
    if p.returncode not in ok:
        raise GitError(f"git {' '.join(args)} failed: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout


def text(repo: str, *args: str, **kw) -> str:
    return git(repo, *args, **kw).decode(errors="replace").strip()


@contextmanager
def temp_index(repo: str):
    """A scratch index file, so the user's real index is never read for writing."""
    d = tempfile.mkdtemp(prefix="gitsloth-")
    path = os.path.join(d, "index")
    try:
        yield path
    finally:
        for f in os.listdir(d):
            os.unlink(os.path.join(d, f))
        os.rmdir(d)


def toplevel(path: str) -> str:
    return text(path, "rev-parse", "--show-toplevel")


def snapshot_tree(repo: str) -> str:
    """Tree object of the current working directory (tracked + untracked, .gitignore respected), via a scratch index."""
    with temp_index(repo) as idx:
        git(repo, "read-tree", "HEAD", index=idx)
        git(repo, "add", "-A", index=idx)
        return text(repo, "write-tree", index=idx)
