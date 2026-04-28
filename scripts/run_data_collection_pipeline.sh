#!/usr/bin/env bash
# 运行 Data_collection/run_data_collection_pipeline.py
# 处理 conda/uv 混合环境下的动态库查找，避免 PyAV 导入时报 CXXABI/libstdc++ 错误。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 若在 conda 环境中，优先使用该环境的动态库（如 libstdc++ / tbb / ffmpeg）
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [[ -d "$CONDA_LIB" ]]; then
        export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

cd "$PROJECT_ROOT"
uv run python Data_collection/run_data_collection_pipeline_le.py "$@"
