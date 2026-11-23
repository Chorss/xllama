# Working rules for this checkout (local)

- Answer in Polish, also after executing a plan and after compaction.
  Identifiers, commands and quoted errors stay untranslated.
- Nothing public (push, PR, issue, comment, PR description edit) without an
  explicit "tak" for THAT step; first show the exact text (a body file in the
  scratchpad). Approval for one step does not cover the next. The
  `pre_publish.py` hook enforces checks; it does not replace the "tak".
- Check yourself before asking the user: console state, installed package,
  designation via `MiB free` in the log.
- PRs always go to `gianlucamazza/xllama` `main` from a fork branch; never a
  PR inside the fork. No closing keywords ("Refs #N", not "Fixes #N").
- Review fixes: one commit covering every point, and a reply to each point
  (fixed / rejected + reason). A safety remark (assert, crash) does not end
  at "mention it in the PR": fix it or ask the user.
- On a maintainer review or merge: diff their changes to OUR prose
  (CHANGELOG, docs, PR body), not only the code, and record the lesson.
- Plans are self-contained (a Sonnet may execute them): paths, commands,
  acceptance criteria.
- Factual and critical; a short answer to a short question.
