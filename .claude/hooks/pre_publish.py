#!/usr/bin/env python3
"""PreToolUse(Bash) hook: gate everything that publishes from the xllama checkout.

Non-publishing commands exit 0 with no output. A publishing command gets:
  deny  inline PR/comment text (use --body-file so it can be checked and shown),
        check_claims.py hits (unless XLLAMA_SKIP_CLAIMS=1 is in the command),
        a d3d12 branch whose preflight stamp is not the pushed commit;
  ask   otherwise, with a summary to show the user before they approve.
Any failure of the hook itself turns into ask, never a silent allow.
"""

import json
import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "origin/main"
UPSTREAM = "gianlucamazza/xllama"

GIT_PUSH = re.compile(r"\bgit(?:\s+(?:-C|-c)\s+\S+)*\s+push\b")
GH_SUB = re.compile(r"\bgh\s+(pr\s+(?:create|edit|comment|review)|issue\s+(?:create|comment|edit))\b")
GH_API = re.compile(r"\bgh\s+api\b")
GH_API_WRITE = re.compile(
    r"-X\s*(?:POST|PATCH|PUT|DELETE)|--method[\s=]+(?:POST|PATCH|PUT|DELETE)"
    r"|(?:-[fF]|--(?:raw-)?field)[\s=]*body=|--input\b|\bmutation\b",
    re.IGNORECASE,
)
SKIP = re.compile(r"\bXLLAMA_SKIP_CLAIMS=1\b")

PREFLIGHT_PATHS = (
    "src/bridge/ggml_d3d12",
    "shaders/",
    "include/xllama/ggml_d3d12.h",
    "docs/gguf-gpu-decode.md",
    "CHANGELOG.md",
    "scripts/bench-",
    "scripts/paired-bench-summary.py",
)


def emit(decision, reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": "xllama pre_publish: " + reason,
    }}))
    return 0


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def tokens(command):
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        return command.split()


def option(toks, *names):
    """Value of the first --name value / --name=value / -n value among toks."""
    for i, t in enumerate(toks):
        for n in names:
            if t == n and i + 1 < len(toks):
                return toks[i + 1]
            if n.startswith("--") and t.startswith(n + "="):
                return t[len(n) + 1:]
    return None


def classify(command):
    """('push'|'pr create'|...|'api', toks) or (None, None)."""
    toks = tokens(command)
    if GIT_PUSH.search(command):
        return "push", toks
    m = GH_SUB.search(command)
    if m:
        return " ".join(m.group(1).split()), toks
    if GH_API.search(command) and GH_API_WRITE.search(command):
        return "api", toks
    return None, None


def repo_dir(toks, cwd):
    for i, t in enumerate(toks[:-1]):
        if t == "git" and toks[i + 1] == "-C" and i + 2 < len(toks):
            return os.path.normpath(os.path.join(cwd, toks[i + 2]))
    return cwd


def body_source(kind, toks, command, cwd):
    """(body_file or None, deny_reason or None)."""
    if kind == "push":
        return None, None
    # Only look at the gh invocation, not e.g. `git checkout -b x && gh ...`.
    starts = [i for i, t in enumerate(toks) if t == "gh"]
    if starts:
        toks = toks[starts[0]:]
        command = " ".join(toks)
    if kind == "api":
        for i, t in enumerate(toks):
            m = re.match(r"(?:(?:--field|--raw-field|-[fF])=?)?body=(.*)", t, re.S)
            if m:
                v = m.group(1)
                flag = toks[i - 1] if i else ""
                is_field = flag in ("-F", "--field") or t.startswith(("-F", "--field"))
                if is_field and v.startswith("@") and v != "@-":
                    return os.path.join(cwd, v[1:]), None
                return None, ("inline body= in gh api; write the text to a file in the "
                              "scratchpad and pass -F body=@<file>, so it can be checked "
                              "and shown to the user first")
        return None, None
    f = option(toks, "--body-file", "-F")
    if f == "-":
        return None, "--body-file - (stdin) cannot be checked; pass a file path"
    if f:
        return os.path.join(cwd, f), None
    if option(toks, "--body", "-b") is not None or re.search(r"--body[\s=]", command):
        return None, ("inline --body/-b; write the text to <scratchpad>/<name>.md and pass "
                      "--body-file <path>, so check_claims.py can scan it and the user can "
                      "read the exact text before approving")
    return None, None


def push_head(toks, repo):
    """The commit a git push sends: first refspec's source if it resolves, else HEAD."""
    try:
        i = toks.index("push")
    except ValueError:
        return "HEAD", None, None
    args = [t for t in toks[i + 1:] if not t.startswith("-")]
    stop = next((j for j, t in enumerate(args) if t in ("&&", "||", ";", "|")), len(args))
    args = args[:stop]
    remote = args[0] if args else None
    ref = args[1] if len(args) > 1 else None
    head = "HEAD"
    if ref:
        src = ref.lstrip("+").split(":")[0]
        if src:
            try:
                git(repo, "rev-parse", "--verify", "-q", src + "^{commit}")
                head = src
            except RuntimeError:
                pass
    return head, remote, ref


def format_hits(hits):
    lines = [f"  {h['where']}  {h['kind']}  {h['fragment']!r} -> {h['need']}" for h in hits[:20]]
    if len(hits) > 20:
        lines.append(f"  ... and {len(hits) - 20} more")
    return "\n".join(lines)


def decide(payload):
    if payload.get("tool_name", "Bash") != "Bash":
        return None
    command = (payload.get("tool_input") or {}).get("command") or ""
    kind, toks = classify(command)
    if kind is None:
        return None
    cwd = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    repo = repo_dir(toks, cwd)

    body, inline = body_source(kind, toks, command, cwd)
    if inline:
        return "deny", inline
    if body and not os.path.isfile(body):
        return "deny", f"body file {body} does not exist"

    import check_claims

    ships_branch = kind in ("push", "pr create", "pr edit")
    head, remote, ref = push_head(toks, repo) if kind == "push" else ("HEAD", None, None)
    head_sha = git(repo, "rev-parse", head)
    mb = git(repo, "merge-base", BASE, head_sha)

    if ships_branch:
        hits = check_claims.run(repo, BASE, head_sha, [body] if body else [])
    else:
        hits = check_claims.run(repo, BASE, BASE, [body]) if body else []
    hits = [dict(zip(("where", "kind", "fragment", "need"), h)) for h in hits]
    skip = bool(SKIP.search(command))
    if hits and not skip:
        return "deny", (
            f"check_claims.py found {len(hits)} unsupported claim(s):\n{format_hits(hits)}\n"
            "Fix: add the bench/results/<date>-<slug>/ directory in this PR, or remove the "
            "figures (submission counts may stay), or move Series X numbers to the PR body "
            "labelled operator-declared. Rules: .claude/rules/evidence.md. Bypass only if "
            "the user explicitly says to publish anyway: prefix XLLAMA_SKIP_CLAIMS=1.")

    if ships_branch:
        changed = git(repo, "diff", "--name-only", f"{mb}..{head_sha}").splitlines()
        d3d12 = [p for p in changed if p.startswith(PREFLIGHT_PATHS)]
        if d3d12:
            stamp_path = git(repo, "rev-parse", "--git-path", "claude-preflight-stamp")
            stamp_path = os.path.join(repo, stamp_path)
            try:
                with open(stamp_path, encoding="utf-8") as f:
                    stamp = f.read().strip()
            except OSError:
                stamp = ""
            if stamp != head_sha:
                what = "never ran" if not stamp else f"ran on {stamp[:12]}"
                return "deny", (
                    f"the branch touches {', '.join(d3d12[:4])}; run /d3d12-optin-preflight "
                    f"(it {what}, HEAD is {head_sha[:12]}). The skill writes the stamp only "
                    "when every item has evidence.")

    ahead = git(repo, "rev-list", "--count", f"{mb}..{head_sha}")
    if kind == "push":
        summary = (f"git push to remote {remote or '(default)'} ref {ref or '(current)'}; "
                   f"{head} = {head_sha[:12]}, {ahead} commit(s) over {BASE}.")
    else:
        title = option(toks, "--title", "-t")
        summary = f"gh {kind}"
        if title:
            summary += f", title {title!r}"
        if body:
            summary += f", body {body}"
        if kind == "pr create":
            target = option(toks, "--repo", "-R")
            base = option(toks, "--base", "-B")
            if target != UPSTREAM or base != "main":
                summary += (f". WARNING: PRs go to {UPSTREAM} main (got --repo {target}, "
                            f"--base {base})")
    note = " Claims check BYPASSED (XLLAMA_SKIP_CLAIMS=1)." if skip and hits else ""
    return "ask", (summary + note + " Show the user the exact text (body file or "
                   "commits) and get a 'tak' for THIS step before approving.")


def main():
    try:
        payload = json.load(sys.stdin)
        result = decide(payload)
    except Exception as e:  # fail safe: ask, never allow silently
        return emit("ask", f"hook error ({type(e).__name__}: {e}); verify by hand "
                           "(check_claims.py, preflight) before approving.")
    if result is None:
        return 0
    return emit(*result)


if __name__ == "__main__":
    sys.exit(main())
