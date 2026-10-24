"""Model-free sweep of the lexical clustering thresholds. Run on DEV repos only."""
import itertools, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay
from gitsloth import plan

repos = sys.argv[1:]
cache = Path("/tmp/windows-" + "-".join(repos) + ".pkl")
if cache.exists():
    cases = pickle.load(open(cache, "rb"))
else:
    cases = []
    for r in repos:
        for w, parent, units, truth in replay.windows(str(replay.ROOT / "repos" / r), [3, 4, 5], 8):
            cases.append((r, units, truth))
    pickle.dump(cases, open(cache, "wb"))
print(len(cases), "dev cases")
base = [replay.ari(t, {u.idx: u.path for u in us}) for _, us, t in cases]
print("per-file baseline mean ARI %.3f" % (sum(base) / len(base)))
rows = []
for hi, sf in itertools.product([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5], [0.0, 0.1, 0.2]):
    sc = []
    for _, us, t in cases:
        g = plan.cluster(us, None, hi, hi, sf)
        sc.append(replay.ari(t, {i: x.id for x in g for i in x.units}))
    rows.append((sum(sc) / len(sc), hi, sf, sc))
for m, hi, sf, sc in sorted(rows, reverse=True)[:10]:
    print(f"hi={hi:.2f} same_file={sf:.1f} mean ARI {m:.3f}  per-case {[round(x,2) for x in sc]}")
