from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from . import build, plan, report
from .gitio import GitError, git, snapshot_tree, text, toplevel
from .llm import LLMError, OllamaClient
from .units import extract_units


def propose(repo: str, client, judge: bool = False, say=lambda s: None):
    """Returns (units, groups, messages, trees, base_commit). Writes only unreferenced objects."""
    base_commit = text(repo, "rev-parse", "HEAD")
    base_tree = text(repo, "rev-parse", "HEAD^{tree}")
    work = snapshot_tree(repo)
    if work == base_tree:
        return None
    units = extract_units(repo, base_tree, work)
    say(f"{len(units)} changes to sort")
    groups = plan.cluster(units, client if judge else None, hi=plan.HI, lo=plan.LO, same_file=plan.SAME_FILE)
    say(f"{len(groups)} groups by identifier overlap" + (" (borderline pairs judged by the model)" if judge else ""))
    groups.sort(key=lambda g: min(g.units))
    trees = build.build_trees(repo, base_tree, work, units, [g.units for g in groups])
    keep = [k for k, t in enumerate(trees) if k == 0 and t != base_tree or k > 0 and t != trees[k - 1]]
    groups, trees = [groups[k] for k in keep], [trees[k] for k in keep]
    messages = [plan.write_message(g, units, client) for g in groups]
    return units, groups, messages, trees, base_commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gitsloth", description="Turn a messy working tree into logical commits on a NEW branch. Your working tree and index are never modified.")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--judge", action="store_true", help="also ask the model about borderline pairs (slower; see README for what it buys)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, create no branch")
    ap.add_argument("--report", metavar="FILE.html", help="also write a self-contained HTML review page (works offline)")
    ap.add_argument("--adopt", action="store_true", help="after creating the branch, move the current branch to it (git reset --mixed; working tree unchanged)")
    a = ap.parse_args(argv)
    try:
        repo = toplevel(a.repo)
        branch = text(repo, "rev-parse", "--abbrev-ref", "HEAD")
        client = OllamaClient(a.model)
        t0 = time.time()
        res = propose(repo, client, a.judge, say=lambda s: print(s, file=sys.stderr, flush=True))
        if res is None:
            print("Nothing to commit: working tree matches HEAD.")
            return 0
        units, groups, messages, trees, base_commit = res
        for g, m in zip(groups, messages):
            files = sorted({units[i].path for i in g.units})
            print(f"\n* {m.splitlines()[0]}   ({len(g.units)} changes, {len(files)} files)")
            for f in files[:6]:
                print(f"    {f}")
            if len(files) > 6:
                print(f"    … {len(files) - 6} more")
        print(f"\n{len(units)} changes -> {len(groups)} commits in {time.time() - t0:.0f}s ({client.calls} local model calls, {client.prompt_tokens + client.output_tokens} tokens)")
        name = None if a.dry_run else f"gitsloth/{branch}-{datetime.now():%Y%m%d-%H%M%S}"
        if a.report:
            with open(a.report, "w", encoding="utf-8") as fh:
                fh.write(report.render(units, groups, messages, model=a.model, seconds=time.time() - t0, calls=client.calls, tree=trees[-1], branch=name))
            print(f"Review page: {a.report}")
        if a.dry_run:
            return 0
        tip_chain = build.commit_chain(repo, base_commit, trees, messages)
        git(repo, "update-ref", f"refs/heads/{name}", tip_chain[-1], "0" * 40)
        print(f"Created branch {name}. Your working tree, index and '{branch}' are untouched.")
        print(f"  review:  git log --stat {branch}..{name}")
        if a.adopt:
            assert text(repo, "rev-parse", f"{name}^{{tree}}") == snapshot_tree(repo)
            git(repo, "reset", "--mixed", name)
            print(f"Adopted: '{branch}' now points at the new history (previous tip {base_commit[:10]}).")
        else:
            print(f"  adopt:   git reset --mixed {name}   (working tree stays exactly as it is)")
        return 0
    except (GitError, LLMError, build.InvariantError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
