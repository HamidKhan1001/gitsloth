"""Break the difference between two trees into indivisible units: single -U0 hunks of text files, or whole-file changes."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from .gitio import git

MAX_HUNKS_PER_FILE = 40  # beyond this a file is treated as one unit (generated/reformatted files)
MAX_TEXT_BYTES = 1_000_000
ZERO = "0" * 40
HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


@dataclass
class Unit:
    idx: int
    path: str
    kind: str  # "hunk" | "file"
    desc: str = ""
    # hunk fields (1-based line numbers as printed by `diff -U0`)
    os: int = 0
    oc: int = 0
    ns: int = 0
    nc: int = 0
    # file fields: (mode, sha) or None when absent on that side
    old: tuple[str, str] | None = None
    new: tuple[str, str] | None = None
    new_lines: list[bytes] = field(default_factory=list, repr=False)
    old_lines: list[bytes] = field(default_factory=list, repr=False)

    @property
    def start0(self) -> int:
        """0-based index in the base file where this hunk's old range begins (insertion point when oc == 0)."""
        return self.os if self.oc == 0 else self.os - 1


def lines_of(data: bytes) -> list[bytes]:
    return io.BytesIO(data).readlines()  # splits on \n only; keeps a final unterminated line intact


def _clip(b: bytes, n: int = 110) -> str:
    s = b.decode("utf-8", errors="replace").rstrip("\r\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def _hunk_desc(path: str, func: str, old: list[bytes], new: list[bytes]) -> str:
    out = [f"{path}" + (f"  (in {func})" if func else "")]
    for tag, ls in (("-", old), ("+", new)):
        for ln in ls[:7]:
            out.append(f"{tag} {_clip(ln)}")
        if len(ls) > 7:
            out.append(f"{tag} … {len(ls) - 7} more lines")
    return "\n".join(out)


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8000]


def _blob(repo: str, sha: str) -> bytes:
    return git(repo, "cat-file", "blob", sha)


def extract_units(repo: str, base_tree: str, work_tree: str) -> list[Unit]:
    raw = git(repo, "diff-tree", "-r", "-z", "--no-renames", "--no-abbrev", base_tree, work_tree).split(b"\0")
    units: list[Unit] = []
    i = 0
    while i + 1 < len(raw) and raw[i]:
        meta = raw[i].decode().lstrip(":").split()
        path = raw[i + 1].decode("utf-8", errors="surrogateescape")
        i += 2
        om, nm, osha, nsha, status = meta[0], meta[1], meta[2], meta[3], meta[4][0]
        old = None if status == "A" else (om, osha)
        new = None if status == "D" else (nm, nsha)
        splittable = status == "M" and om == nm and om in ("100644", "100755")
        hunks: list[Unit] = []
        if splittable:
            ob, nb = _blob(repo, osha), _blob(repo, nsha)
            if not (_is_binary(ob) or _is_binary(nb) or len(ob) > MAX_TEXT_BYTES or len(nb) > MAX_TEXT_BYTES):
                ol, nl = lines_of(ob), lines_of(nb)
                d = git(repo, "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", base_tree, work_tree, "--", path)
                for line in d.split(b"\n"):
                    m = HUNK_RE.match(line)
                    if not m:
                        continue
                    a, b, c, dd = (int(m.group(k)) if m.group(k) is not None else 1 for k in (1, 2, 3, 4))
                    u = Unit(0, path, "hunk", os=a, oc=b, ns=c, nc=dd)
                    u.old_lines = ol[u.start0 : u.start0 + u.oc]
                    u.new_lines = nl[(c if dd == 0 else c - 1) : (c if dd == 0 else c - 1) + dd]
                    u.desc = _hunk_desc(path, m.group(5).decode("utf-8", errors="replace").strip(), u.old_lines, u.new_lines)
                    hunks.append(u)
                if 0 < len(hunks) <= MAX_HUNKS_PER_FILE:
                    units.extend(hunks)
                    continue
        units.append(_file_unit(repo, path, old, new, status))
    for n, u in enumerate(units):
        u.idx = n
    return units


def _file_unit(repo: str, path: str, old, new, status: str) -> Unit:
    u = Unit(0, path, "file", old=old, new=new)
    if new is None:
        u.desc = f"{path}\nDELETED file"
    elif new[0] in ("120000", "160000"):
        u.desc = f"{path}\n{'symlink' if new[0] == '120000' else 'submodule'} {'added' if old is None else 'changed'}"
    else:
        data = _blob(repo, new[1])
        if _is_binary(data):
            u.desc = f"{path}\nBINARY file {'added' if old is None else 'changed'} ({len(data)} bytes)"
        else:
            ls = lines_of(data)
            verb = "NEW file" if old is None else ("mode/type change in" if old[0] != new[0] else "large rewrite of")
            u.desc = f"{path}\n{verb} ({len(ls)} lines)\n" + "\n".join("+ " + _clip(x) for x in ls[:8])
    return u
