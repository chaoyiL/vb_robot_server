#!/usr/bin/env bash
# 运行 policy/scripts/train.py
# 用法：./train.sh [CONFIG] [其他参数...]  CONFIG 默认 pi05_chaoyi
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 若 conda 环境中有 FFmpeg/libstdc++，优先让 PyAV 等依赖从该环境加载动态库，
# 避免系统库版本过低导致 CXXABI 或 av 相关导入失败。
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [[ -d "$CONDA_LIB" ]]; then
        export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

if [[ -n "${1:-}" ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="pi05_chaoyi"
fi

cd "$PROJECT_ROOT"
uv run python policy/scripts/train.py "$CONFIG" --exp-name my_experiment --overwrite "$@"
