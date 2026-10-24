import os
import random

import pytest

from gitsloth.build import InvariantError, build_trees
from gitsloth.gitio import git, snapshot_tree, text
from gitsloth.units import extract_units

from .helpers import make_repo, random_edit, sh, write


def lines(n, tag="l"):
    return b"\n".join(f"{tag}{i}".encode() for i in range(n)) + b"\n"


def prepare(tmp_path, seed):
    rng = random.Random(seed)
    base = {f"src/f{i}.txt": lines(rng.randint(5, 40), f"f{i}_") for i in range(5)}
    base["crlf.txt"] = b"a\r\nb\r\nc\r\nd\r\ne\r\n"
    base["noeol.txt"] = b"one\ntwo\nthree"  # no trailing newline
    base["gone.txt"] = lines(6, "gone")
    base["empty.txt"] = b""
    base["bin.dat"] = bytes(range(256)) * 4
    r = make_repo(tmp_path, base)
    ch = {}
    for p in list(base):
        if p in ("gone.txt", "bin.dat", "empty.txt"):
            continue
        if rng.random() < 0.8:
            ch[p] = random_edit(rng, base[p])
    ch["gone.txt"] = None
    ch["bin.dat"] = bytes(range(255, -1, -1)) * 4
    ch["empty.txt"] = b"was empty\n"
    ch["brand_new.py"] = lines(7, "n")
    ch["newdir/deep/x.txt"] = lines(3, "x")
    write(r, ch)
    return r


@pytest.mark.parametrize("seed", range(25))
def test_any_grouping_reproduces_working_tree(tmp_path, seed):
    r = prepare(tmp_path, seed)
    base_tree = text(r, "rev-parse", "HEAD^{tree}")
    work = snapshot_tree(r)
    units = extract_units(r, base_tree, work)
    assert len(units) > 5
    rng = random.Random(seed + 1000)
    k = rng.randint(1, min(8, len(units)))
    idxs = list(range(len(units)))
    rng.shuffle(idxs)
    groups = [[] for _ in range(k)]
    for n, i in enumerate(idxs):
        groups[n % k if n < k else rng.randrange(k)].append(i)
    trees = build_trees(r, base_tree, work, units, groups)
    assert trees[-1] == work and len(trees) == k


def test_single_hunks_are_actually_split(tmp_path):
    r = make_repo(tmp_path, {"a.txt": lines(30)})
    base = text(r, "rev-parse", "HEAD^{tree}")
    ls = lines(30).split(b"\n")
    ls[2] = b"CHANGED-EARLY"
    ls[25] = b"CHANGED-LATE"
    write(r, {"a.txt": b"\n".join(ls)})
    work = snapshot_tree(r)
    units = extract_units(r, base, work)
    assert [u.kind for u in units] == ["hunk", "hunk"]
    t1, t2 = build_trees(r, base, work, units, [[1], [0]])
    first = git(r, "cat-file", "blob", f"{t1}:a.txt").split(b"\n")
    assert first[25] == b"CHANGED-LATE" and first[2] == b"l2"


def test_mismatched_target_is_refused(tmp_path):
    r = make_repo(tmp_path, {"a.txt": lines(10)})
    base = text(r, "rev-parse", "HEAD^{tree}")
    write(r, {"a.txt": lines(10).replace(b"l3", b"X")})
    work = snapshot_tree(r)
    units = extract_units(r, base, work)
    with pytest.raises(InvariantError):
        build_trees(r, base, base, units, [[0]])  # claims the target is the base tree


def test_missing_unit_is_refused(tmp_path):
    r = prepare(tmp_path, 3)
    base = text(r, "rev-parse", "HEAD^{tree}")
    work = snapshot_tree(r)
    units = extract_units(r, base, work)
    with pytest.raises(InvariantError):
        build_trees(r, base, work, units, [[0]])


def test_user_index_and_worktree_untouched(tmp_path):
    r = prepare(tmp_path, 7)
    sh(r, "add", "brand_new.py")  # partially staged state must survive
    before_status = text(r, "status", "--porcelain=v1")
    before_files = {p: open(os.path.join(r, p), "rb").read() for p in ["brand_new.py", "src/f0.txt"]}
    base = text(r, "rev-parse", "HEAD^{tree}")
    work = snapshot_tree(r)
    units = extract_units(r, base, work)
    build_trees(r, base, work, units, [[i] for i in range(len(units))])
    assert text(r, "status", "--porcelain=v1") == before_status
    assert before_files == {p: open(os.path.join(r, p), "rb").read() for p in before_files}
