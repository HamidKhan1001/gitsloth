"""The model's two narrow jobs: (1) assign each change to a commit group, (2) word each group's commit message."""
from __future__ import annotations

from dataclasses import dataclass, field

from .similarity import similarity_matrix
from .units import Unit

GROUP_SYSTEM = """You help a developer split their uncommitted work into clean, logical commits.
You will see the changes one at a time. For each change, decide whether it belongs to one of the existing commit groups or starts a new one.

Rules:
- A group is ONE purpose: one feature, one bug fix, one refactor, or one docs/test/config update.
- Changes that serve the same purpose belong together even when they are in different files (code + its test, a function + its callers).
- Changes with different purposes must be in different groups, even in the same file.
- Pure whitespace, formatting or typo fixes unrelated to the other work get their own group.
- Answer with the group id, or NEW. When NEW, give a title of 3-8 words describing the purpose, not the file."""

MESSAGE_SYSTEM = """You write git commit messages. Given the changes in one commit, write:
- subject: imperative mood, at most 60 characters, no trailing period, say WHAT this specific commit does, using names that appear in the changes (a verb, then the thing it changes).
- body: one or two plain sentences on why or how, or an empty string if the subject says it all.
Describe only what the changes show. Do not invent details.
Pick the verb from the size: mostly added lines or new files = "Add"/"Document"/"Introduce"; mostly removed = "Remove"; small edits in place = "Fix"/"Update"/"Rename".
Never call a change a typo fix unless it changes only a few characters."""


# Chosen on the dev repos only (click, flask, requests, jinja, werkzeug); see eval/tune.py and the README.
HI, LO, SAME_FILE = 0.15, 0.07, 0.1


@dataclass
class Group:
    id: str
    title: str
    units: list[int] = field(default_factory=list)


def _group_schema(ids: list[str]) -> dict:
    return {"type": "object", "properties": {"group": {"type": "string", "enum": ids + ["NEW"]}, "title": {"type": "string"}}, "required": ["group", "title"]}


def _summary(groups: list[Group], units: list[Unit]) -> str:
    if not groups:
        return "(no groups yet)"
    rows = []
    for g in groups:
        files = sorted({units[i].path for i in g.units})
        shown = ", ".join(files[:4]) + (f" +{len(files) - 4} more" if len(files) > 4 else "")
        rows.append(f"{g.id}: {g.title}  [{len(g.units)} changes in {shown}]")
    return "\n".join(rows)


def cluster_online(units: list[Unit], client, max_groups: int = 10, on_step=None) -> list[Group]:
    groups: list[Group] = []
    for n, u in enumerate(units):
        if not groups:
            ans = {"group": "NEW", "title": ""}
        else:
            ids = [g.id for g in groups]
            prompt = f"Existing groups:\n{_summary(groups, units)}\n\nNext change:\n{u.desc}"
            ans = client.complete_json(GROUP_SYSTEM, prompt, _group_schema(ids))
        target = next((g for g in groups if g.id == ans["group"]), None)
        if target is None or (ans["group"] == "NEW" and len(groups) < max_groups):
            title = (ans.get("title") or "").strip() or f"Changes to {u.path}"
            target = Group(f"G{len(groups) + 1}", title[:80])
            groups.append(target)
        elif ans["group"] == "NEW":  # group cap reached: fall back to the group sharing this file, else the last one
            target = next((g for g in groups if any(units[i].path == u.path for i in g.units)), groups[-1])
        target.units.append(u.idx)
        if on_step:
            on_step(n + 1, len(units), target)
    return groups


JUDGE_SYSTEM = """You decide whether two sets of code changes belong in the same git commit.
Same commit = they serve one purpose (one feature, one bug fix, one refactor), e.g. a function change and the test or doc line that covers it.
Different commits = unrelated purposes, even if they touch the same file. Answer only with same_commit true or false."""


def _sample(g: list[int], units: list[Unit], k: int = 3) -> str:
    return "\n\n".join(units[i].desc[:450] for i in g[:k]) + (f"\n(+{len(g) - k} more changes)" if len(g) > k else "")


def cluster(units: list[Unit], client=None, hi: float = HI, lo: float = LO, same_file: float = SAME_FILE, max_asks: int = 25, on_step=None) -> list[Group]:
    """Agglomerative clustering on identifier overlap. Pairs above `hi` merge on their own; pairs between `lo` and `hi`
    are put to the model as a yes/no question; below `lo` they never merge. With client=None only `hi` applies."""
    n = len(units)
    sim = similarity_matrix(units)
    for i in range(n):
        for j in range(n):
            if i != j and units[i].path == units[j].path:
                sim[i][j] += same_file
    clusters = [[i] for i in range(n)]
    vetoed: set[tuple] = set()
    asks = 0
    while len(clusters) > 1:
        best, bi, bj = -1.0, -1, -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                key = (tuple(clusters[a]), tuple(clusters[b]))
                if key in vetoed:
                    continue
                v = sum(sim[i][j] for i in clusters[a] for j in clusters[b]) / (len(clusters[a]) * len(clusters[b]))
                if v > best:
                    best, bi, bj = v, a, b
        if bi < 0:
            break
        merge = best >= hi
        if not merge and client is not None and best >= lo and asks < max_asks:
            asks += 1
            ans = client.complete_json(JUDGE_SYSTEM, f"Set A:\n{_sample(clusters[bi], units)}\n\nSet B:\n{_sample(clusters[bj], units)}",
                                       {"type": "object", "properties": {"same_commit": {"type": "boolean"}}, "required": ["same_commit"]})
            merge = bool(ans.get("same_commit"))
            if not merge:
                vetoed.add((tuple(clusters[bi]), tuple(clusters[bj])))
        elif not merge and (client is None or best < lo or asks >= max_asks):
            break
        if merge:
            clusters[bi] = sorted(clusters[bi] + clusters[bj])
            del clusters[bj]
            if on_step:
                on_step(n - len(clusters), n, None)
    out = []
    for k, c in enumerate(sorted(clusters, key=min), 1):
        files = sorted({units[i].path for i in c})
        out.append(Group(f"G{k}", "Changes to " + ", ".join(files[:2]) + (" …" if len(files) > 2 else ""), c))
    return out


def write_message(group: Group, units: list[Unit], client) -> str:
    added = sum(len(units[i].new_lines) for i in group.units if units[i].kind == "hunk")
    removed = sum(len(units[i].old_lines) for i in group.units if units[i].kind == "hunk")
    new_files = sum(1 for i in group.units if units[i].kind == "file" and units[i].old is None)
    files = sorted({units[i].path for i in group.units})
    stats = f"Size: +{added}/-{removed} lines in {len(files)} file(s)" + (f", {new_files} new file(s)" if new_files else "") + f"\nFiles: {', '.join(files[:8])}"
    body = stats + "\n\n" + "\n\n".join(units[i].desc for i in group.units)[:2200]
    schema = {"type": "object", "properties": {"subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["subject", "body"]}
    try:
        ans = client.complete_json(MESSAGE_SYSTEM, f"Working title: {group.title}\n\n{body}", schema)
        subject = " ".join(str(ans.get("subject", "")).split()).rstrip(".")
        extra = " ".join(str(ans.get("body", "")).split())
    except Exception:
        subject, extra = "", ""
    if not subject or len(subject) > 72:
        subject = group.title.rstrip(".")[:72]
    return subject + (f"\n\n{extra}" if extra else "")
