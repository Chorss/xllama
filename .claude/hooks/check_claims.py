#!/usr/bin/env python3
"""Flag performance and parity claims that have no committed data behind them.

Scans only the lines a branch ADDS to *.md / *.markdown (git diff base..head),
plus an optional PR/comment body file. Local harness, not part of xllama.

Categories:
  PERF               tok/s, xN.NNN ratios, us/ms, +-N MB, B/A, ratio with CI.
                     Docs: OK only when the paragraph (or a table's caption)
                     cites an existing bench/results/<dir> in head.
                     Body file: also OK when it says operator-declared.
  PARITY             bit for bit, bit-identical, byte-identical, ...
                     Docs: OK with bench/results/... or #N (where the data
                     lives; maintainer's #504 CHANGELOG wording). Body file:
                     also OK when it says operator-declared.
                     Conditional/contract wording (whether, expected,
                     require) is not a claim.
  UNVERIFIED-FIGURE  a xN.NNN or N.NN tok/s figure that is neither on the
                     base ref nor in any bench/results file in head.
Submission/dispatch counts (421 -> 301 submissions) are never hits.

Exit 1 when there are hits, 0 when clean, 2 on usage/git errors.
"""

import argparse
import json
import re
import subprocess
import sys

PERF_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*tok/s"),
    re.compile(r"×\s?\d+\.\d+"),
    re.compile(r"\bx\d+\.\d+\b"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:µs|us|ms)\b"),
    re.compile(r"[+−-]\d+(?:\.\d+)?\s*(?:MB|MiB)\b"),
    re.compile(r"\bB/A\b"),
    re.compile(r"\d\.\d{3}\s*\(\d\.\d{3}\s*[–-]\s*\d\.\d{3}\)"),
]
# A table whose header names one of these units: every data row with a decimal is a hit.
PERF_TABLE_HEADER = re.compile(r"tok/s|\bB/A\b|µs|\bms\b|\bMB\b|×")
DECIMAL = re.compile(r"\d+\.\d+")

PARITY = re.compile(
    r"bit for bit|bit-for-bit|bit-identical|byte-identical|bit identity"
    r"|\bmatch(?:es|ed)?\b[^.\n]{0,80}\bexactly\b",
    re.IGNORECASE,
)
# Wording before the match in the same sentence that makes it a contract, not a claim.
PARITY_HEDGE = re.compile(r"\b(?:whether|if|expected|requires?|required)\b", re.IGNORECASE)

FIGURES = [
    re.compile(r"×\s?(\d+\.\d+)"),
    re.compile(r"\bx(\d+\.\d+)\b"),
    re.compile(r"(\d+\.\d+)\s*tok/s"),
]

BENCH_PATH = re.compile(r"bench/results/[A-Za-z0-9._-]+")
ISSUE_REF = re.compile(r"(?<![\w/])#\d+\b")
OPERATOR = re.compile(r"operator-declared", re.IGNORECASE)


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r


class Repo:
    def __init__(self, path, base, head):
        self.path = path
        self.base = base
        self.head = head
        self.merge_base = git(path, "merge-base", base, head).stdout.strip()
        self._exists = {}
        self._base_text = {}
        self._bench_text = None

    def exists_in_head(self, p):
        p = p.rstrip("/.")
        if p not in self._exists:
            r = git(self.path, "cat-file", "-e", f"{self.head}:{p}", check=False)
            self._exists[p] = r.returncode == 0
        return self._exists[p]

    def base_text(self, path):
        if path not in self._base_text:
            if path is None:
                r = git(self.path, "grep", "-h", "-I", "-e", ".", self.base, "--", "*.md",
                        check=False)
            else:
                r = git(self.path, "show", f"{self.base}:{path}", check=False)
            self._base_text[path] = r.stdout if r.returncode == 0 else ""
        return self._base_text[path]

    def bench_text(self):
        if self._bench_text is None:
            r = git(self.path, "grep", "-h", "-I", "-e", ".", self.head, "--", "bench/results",
                    check=False)
            self._bench_text = r.stdout if r.returncode == 0 else ""
        return self._bench_text

    def added_lines(self):
        """{path: {lineno: text}} of lines added by merge_base..head."""
        out = git(self.path, "diff", "-U0", "--no-color", "--no-ext-diff",
                  f"{self.merge_base}..{self.head}", "--", "*.md", "*.markdown",
                  ":(exclude).claude/").stdout
        added, path, lineno = {}, None, 0
        for line in out.splitlines():
            if line.startswith("+++ "):
                path = None if line == "+++ /dev/null" else line[6:]
            elif line.startswith("@@"):
                m = re.search(r"\+(\d+)", line)
                lineno = int(m.group(1)) if m else 0
            elif line.startswith("+") and path is not None:
                added.setdefault(path, {})[lineno] = line[1:]
                lineno += 1
        return added

    def head_file(self, path):
        return git(self.path, "show", f"{self.head}:{path}").stdout


def paragraphs(lines):
    """Yield (start_index, [lines]) blocks separated by blank lines."""
    block, start = [], 0
    for i, line in enumerate(lines):
        if line.strip():
            if not block:
                start = i
            block.append(line)
        elif block:
            yield start, block
            block = []
    if block:
        yield start, block


def number_present(num, text):
    return re.search(r"(?<![\d.])" + re.escape(num) + r"(?!\d)", text) is not None


def sentence_prefix(line, pos):
    cut = max(line.rfind(". ", 0, pos), line.rfind("; ", 0, pos))
    return line[cut + 1:pos]


def scan(repo, path, lines, added, body_mode):
    """lines: full text; added: set of 0-based indices to check."""
    hits = []
    blocks = list(paragraphs(lines))
    for bi, (start, block) in enumerate(blocks):
        idx = range(start, start + len(block))
        if not any(i in added for i in idx):
            continue
        context = "\n".join(block)
        is_table = block[0].lstrip().startswith("|")
        if is_table and bi > 0:
            context = "\n".join(blocks[bi - 1][1]) + "\n" + context
        bench_ok = any(repo.exists_in_head(p) for p in BENCH_PATH.findall(context))
        operator = bool(OPERATOR.search(context))
        perf_ok = bench_ok or (body_mode and operator)
        parity_ok = bench_ok or bool(ISSUE_REF.search(context)) or (body_mode and operator)
        table_perf = is_table and bool(PERF_TABLE_HEADER.search(block[0]))

        for i in idx:
            if i not in added:
                continue
            line = lines[i]
            where = f"{path}:{i + 1}"
            if not perf_ok:
                frag = None
                for pat in PERF_PATTERNS:
                    m = pat.search(line)
                    if m:
                        frag = m.group(0)
                        break
                if frag is None and table_perf and i > start + 1 and DECIMAL.search(line):
                    frag = line.strip()[:60]
                if frag is not None:
                    need = ("cite an existing bench/results/<dir> in this paragraph"
                            if not body_mode else
                            "cite bench/results/<dir> or label it operator-declared")
                    hits.append((where, "PERF", frag, need))
            if not parity_ok:
                for m in PARITY.finditer(line):
                    if PARITY_HEDGE.search(sentence_prefix(line, m.start())):
                        continue
                    hits.append((where, "PARITY", m.group(0),
                                 "cite bench/results/<dir> or the #N that holds the data"
                                 if not body_mode else
                                 "cite bench/results/<dir>, #N, or say (operator-declared, "
                                 "Series X) with models + ubatch"))
                    break
            if body_mode and operator:
                continue
            for pat in FIGURES:
                for m in pat.finditer(line):
                    num = m.group(1)
                    if number_present(num, repo.base_text(path if not body_mode else None)):
                        continue
                    if number_present(num, repo.bench_text()):
                        continue
                    hits.append((where, "UNVERIFIED-FIGURE", m.group(0),
                                 f"{num} is not on {repo.base} nor in bench/results/ "
                                 f"in {repo.head}; private notes are not a source"))
    return hits


def run(repo_path, base, head, body_files):
    repo = Repo(repo_path, base, head)
    hits = []
    for path, added in sorted(repo.added_lines().items()):
        lines = repo.head_file(path).splitlines()
        hits += scan(repo, path, lines, {n - 1 for n in added}, body_mode=False)
    for bf in body_files:
        with open(bf, encoding="utf-8") as f:
            lines = f.read().splitlines()
        hits += scan(repo, bf, lines, set(range(len(lines))), body_mode=True)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--file", action="append", default=[], help="PR/comment body (repeatable)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        hits = run(a.repo, a.base, a.head, a.file)
    except (RuntimeError, OSError) as e:
        if a.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"check_claims: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps({"hits": [dict(zip(("where", "kind", "fragment", "need"), h))
                                   for h in hits]}))
    else:
        for where, kind, frag, need in hits:
            print(f"{where}  {kind}  {frag!r}  -> {need}")
        print(f"check_claims: {len(hits)} hit(s)" if hits else "check_claims: clean",
              file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
