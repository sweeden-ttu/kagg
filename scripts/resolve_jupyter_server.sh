#!/usr/bin/env bash
# Pick nb.local.com:8888, else ai-rig.local:8888. Prints the base URL (trailing slash).
set -euo pipefail

probe() {
  local code
  code="$(curl -sS --connect-timeout 5 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || true)"
  [[ "$code" =~ ^(200|302|401|403)$ ]]
}

if probe "http://nb.local.com:8888/api"; then
  echo "http://nb.local.com:8888/"
elif probe "http://ai-rig.local:8888/api"; then
  echo "http://ai-rig.local:8888/"
else
  echo "No Jupyter server reachable at nb.local.com:8888 or ai-rig.local:8888" >&2
  exit 1
fi
