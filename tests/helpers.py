import os
import random
import subprocess

from gitsloth.gitio import git, snapshot_tree, text


def sh(repo, *a):
    subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def make_repo(tmp_path, files: dict[str, bytes]) -> str:
    r = str(tmp_path / "repo")
    os.makedirs(r)
    sh(r, "init", "-q")
    write(r, files)
    sh(r, "add", "-A")
    sh(r, "commit", "-qm", "base")
    return r


def write(r, files):
    for p, data in files.items():
        fp = os.path.join(r, p)
        if data is None:
            os.unlink(fp)
            continue
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "wb") as f:
            f.write(data)


def random_edit(rng: random.Random, data: bytes) -> bytes:
    ls = data.split(b"\n")
    for _ in range(rng.randint(1, 6)):
        op = rng.choice("ird")
        k = rng.randrange(len(ls) + 1)
        if op == "i":
            ls.insert(k, f"new{rng.randint(0, 999)}".encode())
        elif op == "r" and ls:
            ls[min(k, len(ls) - 1)] = f"changed{rng.randint(0, 999)}".encode()
        elif op == "d" and len(ls) > 2:
            del ls[min(k, len(ls) - 1)]
    return b"\n".join(ls)
