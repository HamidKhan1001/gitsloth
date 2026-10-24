import pytest

from gitsloth import cli, plan
from gitsloth.gitio import snapshot_tree, text
from gitsloth.llm import LLMError, OllamaClient

from .helpers import make_repo, write


def body(n, tag):
    return b"\n".join(f"{tag}{i} = {i}".encode() for i in range(n)) + b"\n"


class Writer:
    """Fake model: only ever asked for commit messages in the default (lexical) mode."""
    calls = 0
    prompt_tokens = output_tokens = 0

    def complete_json(self, system, user, schema):
        self.calls += 1
        if "subject" in schema["properties"]:
            return {"subject": "Update things.", "body": ""}
        return {"same_commit": False}


def messy_repo(tmp_path):
    """Two retry_limit changes in app.py + the test that covers them (one purpose), and an unrelated new module."""
    r = make_repo(tmp_path, {"app.py": body(30, "base_"), "tests/test_app.py": body(20, "t_")})
    a = body(30, "base_").split(b"\n"); a[1] = b"retry_limit = clamp_retry_limit(retry_limit)"; a[28] = b"def apply_retry_limit(): return retry_limit"
    t = body(20, "t_").split(b"\n"); t[10] = b"assert apply_retry_limit() == clamp_retry_limit(retry_limit)"
    write(r, {"app.py": b"\n".join(a), "tests/test_app.py": b"\n".join(t), "zebra.py": b"stripes = ['black', 'white']\nprint(stripes)\n"})
    return r


def test_related_changes_group_across_files(tmp_path):
    r = messy_repo(tmp_path)
    units, groups, msgs, trees, base = cli.propose(r, Writer())
    by_group = [sorted({units[i].path for i in g.units}) for g in groups]
    assert ["app.py", "tests/test_app.py"] in by_group and ["zebra.py"] in by_group
    assert len(groups) == 2 and trees[-1] == snapshot_tree(r)
    assert all(m.startswith("Update things") and not m.splitlines()[0].endswith(".") for m in msgs)


def test_cli_creates_branch_and_leaves_user_state(tmp_path, monkeypatch):
    r = messy_repo(tmp_path)
    status, head = text(r, "status", "--porcelain=v1"), text(r, "rev-parse", "HEAD")
    monkeypatch.setattr(cli, "OllamaClient", lambda m: Writer())
    assert cli.main(["--repo", r]) == 0
    assert text(r, "status", "--porcelain=v1") == status and text(r, "rev-parse", "HEAD") == head
    br = text(r, "for-each-ref", "--format=%(refname:short)", "refs/heads/gitsloth/")
    assert len(br.splitlines()) == 1 and text(r, "rev-list", "--count", f"HEAD..{br}") == "2"


def test_adopt_leaves_clean_status(tmp_path, monkeypatch):
    r = messy_repo(tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda m: Writer())
    assert cli.main(["--repo", r, "--adopt"]) == 0
    assert text(r, "status", "--porcelain=v1") == ""


def test_dry_run_writes_no_refs_and_report_is_self_contained(tmp_path, monkeypatch):
    r = messy_repo(tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda m: Writer())
    out = tmp_path / "review.html"
    assert cli.main(["--repo", r, "--dry-run", "--report", str(out)]) == 0
    assert text(r, "for-each-ref", "refs/heads/gitsloth/") == ""
    html = out.read_text()
    assert "2 commits" in html and "byte-identical" in html and "http" not in html.replace("http-equiv", "")
    assert "<script" not in html


def test_clean_tree_is_noop(tmp_path, monkeypatch, capsys):
    r = make_repo(tmp_path, {"a.txt": b"x\n"})
    monkeypatch.setattr(cli, "OllamaClient", lambda m: Writer())
    assert cli.main(["--repo", r]) == 0
    assert "Nothing to commit" in capsys.readouterr().out


def test_hostile_model_cannot_break_the_invariant(tmp_path):
    """--judge with a model that says yes to everything, and a message writer that returns junk."""
    r = messy_repo(tmp_path)

    class Hostile(Writer):
        def complete_json(self, system, user, schema):
            if "subject" in schema["properties"]:
                return {"subject": "", "body": 5}
            return {"same_commit": True}

    units, groups, msgs, trees, base = cli.propose(r, Hostile(), judge=True)
    assert all(m.strip() for m in msgs)  # propose() would have raised if the final tree differed


def test_judge_never_runs_unless_asked(tmp_path):
    r = messy_repo(tmp_path)

    class Counting(Writer):
        asked = 0
        def complete_json(self, system, user, schema):
            if "same_commit" in schema["properties"]:
                Counting.asked += 1
            return super().complete_json(system, user, schema)

    cli.propose(r, Counting())
    assert Counting.asked == 0


def test_refuses_remote_llm_host():
    with pytest.raises(LLMError):
        OllamaClient(base_url="https://api.example.com")
    OllamaClient(base_url="http://localhost:11434")
