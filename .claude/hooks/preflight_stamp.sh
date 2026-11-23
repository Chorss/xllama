#!/usr/bin/env bash
# Called only by the d3d12-optin-preflight skill, after every item has evidence.
# pre_publish.py denies a push/PR of a d3d12 branch unless this stamp == pushed commit.
set -euo pipefail
stamp="$(git rev-parse --git-path claude-preflight-stamp)"
git rev-parse HEAD >"$stamp"
echo "preflight stamp: $(cat "$stamp") -> $stamp"
