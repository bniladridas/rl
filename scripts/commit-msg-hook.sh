#!/bin/bash

# Check commit message
msg=$(cat "$1")
first_line=$(echo "$msg" | head -n1)

if ! echo "$first_line" | grep -qE '^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: '; then
    echo >&2 "Commit message must follow conventional commit format (start with type: )"
    exit 1
fi

if [ ${#first_line} -gt 40 ]; then
    echo >&2 "Commit message first line too long (>40 characters)"
    exit 1
fi

if echo "$first_line" | grep -q '[A-Z]'; then
    echo >&2 "Commit message first line must be lowercase"
    exit 1
fi

# This example catches duplicate Signed-off-by lines.
test "" = "$(grep '^Signed-off-by: ' "$1" |
	 sort | uniq -c | sed -e '/^[ 	]*1[ 	]/d')" || {
	echo >&2 Duplicate Signed-off-by lines.
	exit 1
}