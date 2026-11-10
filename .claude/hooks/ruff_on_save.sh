#!/bin/bash
# After Claude edits a Python file: format it and apply safe lint fixes, then report what is left.
file=$(python3 -c 'import json, sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')
[[ "$file" == *.py && -f "$file" ]] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
ruff=.venv/bin/ruff
[ -x "$ruff" ] || exit 0
"$ruff" format --quiet "$file"
"$ruff" check --fix --quiet "$file" >/dev/null
remaining=$("$ruff" check --quiet --output-format concise "$file")
if [ -n "$remaining" ]; then
    echo "ruff found issues in $file:" >&2
    echo "$remaining" >&2
    exit 2
fi
