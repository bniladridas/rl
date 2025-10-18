#!/bin/bash
# Docker dry-run equivalent using hadolint
# Usage: ./scripts/docker-dry-run.sh <dockerfile-path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dockerfile-path>"
    echo "Example: $0 deploy/Dockerfile"
    exit 1
fi

DOCKERFILE="$1"

if [ ! -f "$DOCKERFILE" ]; then
    echo "❌ Error: Dockerfile '$DOCKERFILE' not found!"
    exit 1
fi

echo "🔍 Validating Dockerfile syntax and best practices..."
if hadolint "$DOCKERFILE"; then
    echo "✅ Dockerfile syntax is valid!"
else
    echo "❌ Dockerfile has issues!"
    exit 1
fi
