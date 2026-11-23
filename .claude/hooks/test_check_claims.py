#!/usr/bin/env python3
"""Tests for check_claims.py on throwaway git repos."""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_claims  # noqa: E402


def sh(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)


class GitRepo:
    """main has docs/a.md; branch adds whatever a test writes."""

    def __init__(self, main_doc="# Doc\n\nThe bias ADD gave ×1.050 on Bielik.\n"):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name
        sh(self.path, "init", "-q", "-b", "main")
        sh(self.path, "config", "user.email", "t@example.invalid")
        sh(self.path, "config", "user.name", "t")
        self.write("docs/a.md", main_doc)
        self.commit("main")
        sh(self.path, "checkout", "-q", "-b", "feat")

    def write(self, rel, text):
        full = os.path.join(self.path, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def append(self, rel, text):
        with open(os.path.join(self.path, rel), "a", encoding="utf-8") as f:
            f.write(text)

    def commit(self, msg="c"):
        sh(self.path, "add", "-A")
        sh(self.path, "commit", "-q", "-m", msg)

    def hits(self, body=None):
        files = [body] if body else []
        return check_claims.run(self.path, "main", "HEAD", files)

    def close(self):
        self.tmp.cleanup()


TABLE = """
Measured on Series X (operator-declared, not project evidence):

| Model  | Decode A → B tok/s | B/A decode          |
| ------ | ------------------ | ------------------- |
| Bielik | 15.19 → 16.09      | 1.056 (1.050–1.063) |
"""


class CheckClaims(unittest.TestCase):
    def setUp(self):
        self.r = GitRepo()

    def tearDown(self):
        self.r.close()

    def kinds(self, hits):
        return sorted({h[1] for h in hits})

    def test_clean_branch(self):
        self.r.append("docs/a.md", "\nA new paragraph without figures.\n")
        self.r.commit()
        self.assertEqual(self.r.hits(), [])

    def test_tok_s_table_without_data_is_perf(self):
        self.r.append("docs/a.md", TABLE)
        self.r.commit()
        hits = self.r.hits()
        self.assertIn("PERF", self.kinds(hits))
        self.assertTrue(any(h[0].endswith(":9") for h in hits), hits)  # data row

    def test_tok_s_table_with_existing_bench_dir_is_ok(self):
        self.r.write("bench/results/2026-11-20-x/README.md", "data 15.19 16.09\n")
        self.r.append("docs/a.md", TABLE.replace(
            "not project evidence)", "not project evidence; bench/results/2026-11-20-x/)"))
        self.r.commit()
        self.assertEqual(self.r.hits(), [])

    def test_bench_dir_must_exist_in_head(self):
        self.r.append("docs/a.md", TABLE.replace(
            "not project evidence)", "bench/results/2026-11-20-missing/)"))
        self.r.commit()
        self.assertIn("PERF", self.kinds(self.r.hits()))

    def test_operator_declared_is_not_enough_in_docs(self):
        self.r.append("docs/a.md", "\nDecode went 45.10 tok/s (operator-declared).\n")
        self.r.commit()
        self.assertIn("PERF", self.kinds(self.r.hits()))

    def test_parity_operator_declared_ok_in_body(self):
        body = self.r.write("body.md", "Logits matched bit for bit (operator-declared, "
                                       "Series X; Coder-3B, ubatch 512 and 1).\n")
        self.assertEqual(self.r.hits(body), [])

    def test_parity_in_docs_needs_issue_or_data(self):
        self.r.append("docs/a.md", "\nLogits matched bit for bit "
                                   "(operator-declared, Series X).\n")
        self.r.commit()
        self.assertEqual(self.kinds(self.r.hits()), ["PARITY"])

    def test_parity_with_issue_reference_ok(self):
        self.r.append("docs/a.md", "\nThe Series X comparisons reported in #503 matched "
                                   "logits bit for bit.\n")
        self.r.commit()
        self.assertEqual(self.r.hits(), [])

    def test_parity_contract_wording_is_not_a_claim(self):
        self.r.append("docs/a.md", "\nThe gate does not require bit identity. Whether "
                                   "the output is bit-identical goes in the error column. "
                                   "The expected result is the CPU's bit for bit.\n")
        self.r.commit()
        self.assertEqual(self.r.hits(), [])

    def test_ratio_not_on_main_is_unverified(self):
        self.r.append("docs/a.md", "\nThe bias ADD removed 120 submissions for ×1.107.\n")
        self.r.commit()
        self.assertEqual(self.kinds(self.r.hits()), ["PERF", "UNVERIFIED-FIGURE"])

    def test_ratio_already_on_main_is_not_unverified(self):
        self.r.append("docs/a.md", "\nAs above, ×1.050 on Bielik.\n")
        self.r.commit()
        self.assertEqual(self.kinds(self.r.hits()), ["PERF"])

    def test_submission_counts_are_ok(self):
        self.r.append("docs/a.md", "\nMeasured on Series X: Bielik 421 → 301 submissions "
                                   "per token (60 layers × 7 → 5, plus lm_head).\n")
        self.r.commit()
        self.assertEqual(self.r.hits(), [])

    def test_only_added_lines_count(self):
        # A figure already on main is not re-flagged when an unrelated line changes.
        r = GitRepo(main_doc="# Doc\n\nOld 12.5 tok/s line.\n")
        try:
            r.append("docs/a.md", "\nNew prose.\n")
            r.commit()
            self.assertEqual(r.hits(), [])
        finally:
            r.close()

    def test_body_figures_operator_declared_ok(self):
        body = self.r.write("body.md", "Measured on Series X (operator-declared):\n\n"
                                       "| Model | tok/s |\n| --- | --- |\n| A | 45.10 |\n")
        self.assertEqual(self.r.hits(body), [])

    def test_body_figures_unlabelled_hit(self):
        body = self.r.write("body.md", "Decode is 48.43 tok/s now, ×1.071.\n")
        self.assertIn("UNVERIFIED-FIGURE", self.kinds(self.r.hits(body)))
        self.assertIn("PERF", self.kinds(self.r.hits(body)))

    def test_harness_files_are_not_scanned(self):
        self.r.write(".claude/rules/evidence.md", "No ×1.107 and no bit for bit.\n")
        self.r.commit()
        self.assertEqual(self.r.hits(), [])

    def test_cli_exit_codes(self):
        self.assertEqual(check_claims.main(["--repo", self.r.path, "--base", "main"]), 0)
        self.r.append("docs/a.md", "\nNow 3.2 ms per token.\n")
        self.r.commit()
        self.assertEqual(check_claims.main(["--repo", self.r.path, "--base", "main"]), 1)
        self.assertEqual(check_claims.main(["--repo", self.r.path, "--base", "nope"]), 2)


if __name__ == "__main__":
    unittest.main()
