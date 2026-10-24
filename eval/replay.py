"""Replay evaluation: squash k consecutive real commits into one 'working state', ask gitsloth to split it,
and score the grouping against the commits the authors actually made (ground truth from git blame).

Metric: pairwise agreement over units whose true commit is unambiguous. For every pair of units, did the
tool agree with history on 'same commit' vs 'different commit'?  Reported as precision / recall / F1.
Baselines are computed on exactly the same units.
"""
import argparse, itertools, json, os, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gitsloth import plan
from gitsloth.build import build_trees
from gitsloth.gitio import git, text
from gitsloth.llm import OllamaClient
from gitsloth.units import extract_units

ROOT = Path(__file__).resolve().parent


def commits_linear(repo, n=400):
    out = text(repo, "rev-list", "--first-parent", "--no-merges", "--reverse", f"--max-count={n}", "HEAD")
    return out.split()


def truth_for(repo, units, base, tip, window):
    pos = {c: i for i, c in enumerate(window)}
    truth = {}
    for u in units:
        if u.kind == "hunk" and u.nc > 0:
            out = text(repo, "blame", "--porcelain", "-L", f"{u.ns},+{u.nc}", tip, "--", u.path)
            cnt, cur = Counter(), None
            for l in out.splitlines():
                if l.startswith("\t"):
                    cnt[cur] += 1
                else:
                    f = l.split()
                    if len(f) >= 3 and len(f[0]) == 40 and f[1].isdigit() and f[2].isdigit():
                        cur = f[0]
            if not cnt:
                continue
            sha, n = cnt.most_common(1)[0]
            if sha in pos and n / sum(cnt.values()) >= 0.8:
                truth[u.idx] = pos[sha]
        elif u.kind == "file" and (u.old is None or u.new is None):
            flt = "A" if u.old is None else "D"
            out = text(repo, "log", f"--diff-filter={flt}", "--format=%H", f"{base}..{tip}", "--", u.path).split()
            hit = [c for c in out if c in pos]
            if len(hit) == 1:
                truth[u.idx] = pos[hit[0]]
    return truth


def ari(truth, pred):
    """Adjusted Rand index: 1.0 = identical partition, ~0 = no better than chance (a single big group or random scatter scores ~0)."""
    from math import comb
    ids = sorted(truth)
    n = len(ids)
    cont = Counter((truth[i], pred[i]) for i in ids)
    a = Counter(truth[i] for i in ids); b = Counter(pred[i] for i in ids)
    sij = sum(comb(v, 2) for v in cont.values()); sa = sum(comb(v, 2) for v in a.values()); sb = sum(comb(v, 2) for v in b.values())
    exp = sa * sb / comb(n, 2) if n > 1 else 0
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (sij - exp) / (mx - exp)


def pair_scores(truth, pred):
    ids = sorted(truth)
    tp = fp = fn = 0
    for a, b in itertools.combinations(ids, 2):
        t, p = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and p; fp += (not t) and p; fn += t and (not p)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def windows(repo, sizes, want, lo=6, hi=34):
    cs = commits_linear(repo)
    found, step = [], 3
    for start in range(3, len(cs) - 6, step):
        k = sizes[len(found) % len(sizes)]
        w = cs[start : start + k]
        if len(w) < k:
            break
        base_tree = text(repo, "rev-parse", f"{w[0]}^{{tree}}")
        parent = text(repo, "rev-parse", f"{w[0]}^")
        tip = w[-1]
        units = extract_units(repo, text(repo, "rev-parse", f"{parent}^{{tree}}"), text(repo, "rev-parse", f"{tip}^{{tree}}"))
        if not (lo <= len(units) <= hi):
            continue
        truth = truth_for(repo, units, parent, tip, w)
        if len(set(truth.values())) >= 2 and len(truth) >= 6:
            found.append((w, parent, units, truth))
        if len(found) >= want:
            break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--want", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["online", "lexical", "hybrid"], default="hybrid")
    ap.add_argument("--hi", type=float, default=0.30)
    ap.add_argument("--lo", type=float, default=0.12)
    ap.add_argument("--same-file", type=float, default=0.0)
    ap.add_argument("--max-asks", type=int, default=12)
    a = ap.parse_args()
    out = ROOT / "results" / a.out; out.mkdir(parents=True, exist_ok=True)
    for name in a.repos:
        repo = str(ROOT / "repos" / name)
        for n, (w, parent, units, truth) in enumerate(windows(repo, [3, 4, 5], a.want)):
            f = out / f"{name}-{n}.json"
            if f.exists():
                continue
            client = OllamaClient(a.model)
            t0 = time.time()
            if a.method == "online":
                groups = plan.cluster_online(units, client)
            else:
                groups = plan.cluster(units, client if a.method == "hybrid" else None, a.hi, a.lo, a.same_file, a.max_asks)
            secs = time.time() - t0
            pred = {i: g.id for g in groups for i in g.units}
            base_tree = text(repo, "rev-parse", f"{parent}^{{tree}}"); tip_tree = text(repo, "rev-parse", f"{w[-1]}^{{tree}}")
            build_trees(repo, base_tree, tip_tree, units, [g.units for g in groups])  # invariant must hold on real code
            per_file = {u.idx: u.path for u in units}
            res = {
                "repo": name, "commits": [text(repo, 'log', '-1', '--format=%h %s', c) for c in w], "units": len(units), "scored": len(truth),
                "true_commits": len(set(truth.values())), "pred_groups": len(groups), "secs": round(secs), "tokens": client.prompt_tokens + client.output_tokens,
                "invariant_ok": True, "titles": {g.id: g.title for g in groups},
                "method": a.method, "model_calls": client.calls,
                "model": {**dict(zip(("p", "r", "f1"), pair_scores(truth, pred))), "ari": ari(truth, pred)},
                "per_file": {**dict(zip(("p", "r", "f1"), pair_scores(truth, per_file))), "ari": ari(truth, per_file)},
                "one_commit": {**dict(zip(("p", "r", "f1"), pair_scores(truth, {i: 0 for i in range(len(units))}))), "ari": ari(truth, {i: 0 for i in range(len(units))})},
            }
            f.write_text(json.dumps(res, indent=1))
            m, b = res["model"], res["per_file"]
            print(f"{name}-{n}: units={len(units)} scored={len(truth)} true={res['true_commits']} pred={len(groups)}  ARI model={m['ari']:.2f} per-file={b['ari']:.2f} one-commit={res['one_commit']['ari']:.2f}  (pair-F1 {m['f1']:.2f})  {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
