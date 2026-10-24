"""Single-file HTML review page for a proposed split. No scripts, no external requests: it opens offline."""
from __future__ import annotations

from html import escape

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1b16;--mut:#6b675c;--card:#fff;--line:#e4e0d4;--add:#e6f4ea;--addf:#14532d;--del:#fdecea;--delf:#7f1d1d;--acc:#7c5a2b}
@media (prefers-color-scheme:dark){:root{--bg:#16140f;--fg:#ece8dc;--mut:#a09a88;--card:#1e1c15;--line:#38342a;--add:#12301c;--addf:#a7e3b9;--del:#3a1816;--delf:#f5b5ae;--acc:#d9b27a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}h1{font-size:1.5rem;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 24px}
.ok{display:inline-block;background:var(--add);color:var(--addf);border-radius:6px;padding:2px 10px;font-size:.85rem}
article{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:20px 0;overflow:hidden}
article>header{padding:14px 18px;border-bottom:1px solid var(--line)}article h2{font-size:1.05rem;margin:0}
.n{color:var(--acc);font-variant-numeric:tabular-nums;margin-right:8px}.meta{color:var(--mut);font-size:.85rem;margin-top:2px}
.body{color:var(--mut);font-size:.9rem;margin:6px 0 0}
.hunk{border-top:1px solid var(--line)}.hunk:first-of-type{border-top:0}.path{padding:6px 18px;font:600 .8rem ui-monospace,monospace;color:var(--mut)}
pre{margin:0;padding:0 0 8px;overflow-x:auto;font:.8rem/1.45 ui-monospace,SFMono-Regular,monospace}
pre span{display:block;padding:0 18px;white-space:pre}.a{background:var(--add);color:var(--addf)}.d{background:var(--del);color:var(--delf)}
code{font-family:ui-monospace,monospace;font-size:.85em}
"""


def _diff(u) -> str:
    if u.kind == "hunk":
        rows = [f'<span class="d">- {escape(l.decode("utf-8", "replace").rstrip(chr(10)))}</span>' for l in u.old_lines[:40]]
        rows += [f'<span class="a">+ {escape(l.decode("utf-8", "replace").rstrip(chr(10)))}</span>' for l in u.new_lines[:40]]
        more = max(0, len(u.old_lines) - 40) + max(0, len(u.new_lines) - 40)
        if more:
            rows.append(f"<span>… {more} more lines</span>")
        return "".join(rows)
    return "".join(f"<span>{escape(l)}</span>" for l in u.desc.splitlines()[1:])


def render(units, groups, messages, *, model: str, seconds: float, calls: int, tree: str, branch: str | None) -> str:
    cards = []
    for n, (g, m) in enumerate(zip(groups, messages), 1):
        subject, _, body = m.partition("\n\n")
        files = sorted({units[i].path for i in g.units})
        hunks = "".join(
            f'<div class="hunk"><div class="path">{escape(units[i].path)}</div><pre>{_diff(units[i])}</pre></div>' for i in g.units
        )
        cards.append(
            f'<article><header><h2><span class="n">{n}</span>{escape(subject)}</h2>'
            f'<div class="meta">{len(g.units)} changes in {len(files)} file{"s" * (len(files) != 1)}</div>'
            + (f'<p class="body">{escape(body)}</p>' if body else "")
            + f"</header>{hunks}</article>"
        )
    where = f" on branch <code>{escape(branch)}</code>" if branch else " (dry run, nothing written)"
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>gitsloth review</title><style>{CSS}</style></head><body><main>"
        f"<h1>{len(units)} changes became {len(groups)} commits</h1>"
        f'<p class="sub">Split locally by <code>{escape(model)}</code> in {seconds:.0f}s ({calls} model calls){where}.</p>'
        f'<p><span class="ok">Verified: last commit tree <code>{tree[:12]}</code> is byte-identical to your working tree</span></p>'
        + "".join(cards)
        + "</main></body></html>"
    )
