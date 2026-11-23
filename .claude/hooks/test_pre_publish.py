#!/usr/bin/env python3
"""Tests for pre_publish.py: synthetic PreToolUse JSON on stdin, throwaway git repos."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HOOK = os.path.join(HERE, "pre_publish.py")


def sh(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


class Repo:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name
        sh(self.path, "init", "-q", "-b", "main")
        sh(self.path, "config", "user.email", "t@example.invalid")
        sh(self.path, "config", "user.name", "t")
        self.write("docs/a.md", "# Doc\n")
        self.commit()
        sh(self.path, "update-ref", "refs/remotes/origin/main", "HEAD")
        sh(self.path, "checkout", "-q", "-b", "feat")

    def write(self, rel, text):
        full = os.path.join(self.path, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def commit(self):
        sh(self.path, "add", "-A")
        sh(self.path, "commit", "-q", "-m", "c")

    def stamp(self, sha=None):
        p = os.path.join(self.path, sh(self.path, "rev-parse", "--git-path",
                                       "claude-preflight-stamp"))
        with open(p, "w") as f:
            f.write((sha or sh(self.path, "rev-parse", "HEAD")) + "\n")


def run_hook(command, cwd):
    payload = {"tool_name": "Bash", "cwd": cwd, "tool_input": {"command": command}}
    r = subprocess.run([sys.executable, "-I", HOOK], input=json.dumps(payload),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    if not r.stdout.strip():
        return None, ""
    out = json.loads(r.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    return out["permissionDecision"], out["permissionDecisionReason"]


class PrePublish(unittest.TestCase):
    def setUp(self):
        self.r = Repo()

    def tearDown(self):
        self.r.tmp.cleanup()

    def hook(self, command):
        return run_hook(command, self.r.path)

    def add_claim(self):
        self.r.write("docs/a.md", "# Doc\n\nDecode 45.10 → 48.43 tok/s, ×1.071.\n")
        self.r.commit()

    def test_non_publishing_commands_pass_silently(self):
        for cmd in ("git status", "git log --oneline -3",
                    "gh api repos/x/y/pulls/1/reviews", "gh pr view 511", "gh pr list"):
            self.assertEqual(self.hook(cmd), (None, ""), cmd)

    def test_push_with_claims_is_denied(self):
        self.add_claim()
        decision, reason = self.hook("git -C . push fork feat")
        self.assertEqual(decision, "deny")
        self.assertIn("UNVERIFIED-FIGURE", reason)
        self.assertIn("docs/a.md:3", reason)

    def test_inline_body_is_denied(self):
        decision, reason = self.hook('gh pr create --title t --body "x"')
        self.assertEqual(decision, "deny")
        self.assertIn("--body-file", reason)
        decision, _ = self.hook("gh api -X POST repos/x/y/issues/1/comments -f body=hello")
        self.assertEqual(decision, "deny")

    def test_skip_env_turns_claims_into_ask(self):
        self.add_claim()
        decision, reason = self.hook("XLLAMA_SKIP_CLAIMS=1 git push fork feat")
        self.assertEqual(decision, "ask")
        self.assertIn("BYPASSED", reason)

    def test_clean_push_with_fresh_stamp_asks(self):
        self.r.write("src/bridge/ggml_d3d12.cpp", "// x\n")
        self.r.commit()
        self.r.stamp()
        decision, reason = self.hook("git push fork feat")
        self.assertEqual(decision, "ask")
        self.assertIn("1 commit(s)", reason)

    def test_stale_or_missing_stamp_is_denied(self):
        self.r.write("CHANGELOG.md", "- a\n")
        self.r.commit()
        decision, reason = self.hook("git push fork feat")
        self.assertEqual(decision, "deny")
        self.assertIn("never ran", reason)
        self.r.stamp()
        self.r.write("CHANGELOG.md", "- b\n")
        self.r.commit()
        decision, reason = self.hook("git push fork feat")
        self.assertEqual(decision, "deny")
        self.assertIn("/d3d12-optin-preflight", reason)

    def test_non_d3d12_branch_needs_no_stamp(self):
        self.r.write("docs/a.md", "# Doc\n\nPlain prose.\n")
        self.r.commit()
        self.assertEqual(self.hook("git push fork feat")[0], "ask")

    def test_body_file_is_checked(self):
        body = self.r.write("body.md", "Now 48.43 tok/s.\n")
        decision, reason = self.hook(f"gh pr comment 511 --body-file {body}")
        self.assertEqual(decision, "deny")
        self.assertIn("body.md:1", reason)
        clean = self.r.write("ok.md", "Fixed in abc123.\n")
        decision, reason = self.hook(f"gh pr comment 511 --body-file {clean}")
        self.assertEqual(decision, "ask")
        self.assertIn("ok.md", reason)

    def test_body_file_operator_declared_numbers_ask(self):
        body = self.r.write("body.md", "Series X (operator-declared): 45.10 → 48.43 tok/s.\n")
        self.assertEqual(self.hook(f"gh issue comment 503 -F {body}")[0], "ask")

    def test_missing_body_file_is_denied(self):
        decision, _ = self.hook("gh pr comment 1 --body-file /nonexistent/x.md")
        self.assertEqual(decision, "deny")

    def test_pr_create_warns_about_target(self):
        body = self.r.write("b.md", "What & why\n")
        decision, reason = self.hook(f"gh pr create --title t --body-file {body}")
        self.assertEqual(decision, "ask")
        self.assertIn("WARNING", reason)
        decision, reason = self.hook(f"gh pr create -R gianlucamazza/xllama -B main "
                                     f"--title t --body-file {body}")
        self.assertNotIn("WARNING", reason)

    def test_checkout_dash_b_is_not_inline_body(self):
        decision, _ = self.hook("git checkout -b x && gh pr create --fill")
        self.assertEqual(decision, "ask")

    def test_hook_error_fails_safe_to_ask(self):
        with tempfile.TemporaryDirectory() as d:
            decision, reason = run_hook("git push fork feat", d)
        self.assertEqual(decision, "ask")
        self.assertIn("hook error", reason)


if __name__ == "__main__":
    unittest.main()
