#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_PY="real_world/robot_api/joint_jog_pykin.py"

if [[ ! -f "${REPO_ROOT}/${TARGET_PY}" ]]; then
  echo "Error: target script not found: ${TARGET_PY}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec python "${TARGET_PY}" "$@"
