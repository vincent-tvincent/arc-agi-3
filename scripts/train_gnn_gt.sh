#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG="${CONFIG:-configs/train_gnn_arc.yaml}"
GT_DIR="${GT_DIR:-/run/media/blue-lobster/disk3/CS274p_output/arc_gt/GT}"
CACHE_DIR="${CACHE_DIR:-/run/media/blue-lobster/disk3/CS274p_output/gnn_cache_gt_soft_v1}"
PRETRAINED_GNN="${PRETRAINED_GNN:-/run/media/blue-lobster/disk3/CS274p_output/runs/train_gnn_arc/checkpoints/gnn_latest.pt}"
WORKERS="${WORKERS:-8}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
MODE="${MODE:-cache}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/train_gnn_gt.sh [options]

Options:
  --cache                         Precompute/use cache, then train from cache. Default.
  --direct                        Train directly from GT records without cache.
  --rebuild-cache                 Rebuild cache before training.
  --python-bin PATH               Python executable. Default: python or $PYTHON_BIN.
  --config PATH                   GNN config YAML.
  --gt-dir PATH                   GT record directory.
  --cache-dir PATH                Precomputed cache directory.
  --pretrained-gnn PATH           Warm-start checkpoint.
  --workers N                     Cache precompute worker processes.
  --shard-size N                  Examples per cache shard.
  -h, --help                      Show this help.

Examples:
  scripts/train_gnn_gt.sh --python-bin ../arc_agi/bin/python
  scripts/train_gnn_gt.sh --rebuild-cache --workers 12
  scripts/train_gnn_gt.sh --direct
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache)
      MODE="cache"
      shift
      ;;
    --direct)
      MODE="direct"
      shift
      ;;
    --rebuild-cache)
      REBUILD_CACHE="1"
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --gt-dir)
      GT_DIR="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --pretrained-gnn)
      PRETRAINED_GNN="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --shard-size)
      SHARD_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3. Activate your venv or pass --python-bin PATH." >&2
    exit 127
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Activate your venv or pass --python-bin PATH." >&2
  exit 127
fi

if [[ "${MODE}" == "direct" ]]; then
  "${PYTHON_BIN}" scripts/train_gnn.py \
    --config "${CONFIG}" \
    --data-dir "${GT_DIR}" \
    --data-format arc \
    --pretrained-gnn "${PRETRAINED_GNN}"
  exit 0
fi

if [[ "${MODE}" != "cache" ]]; then
  echo "Unknown MODE='${MODE}'. Use MODE=cache or MODE=direct." >&2
  exit 2
fi

PRECOMPUTE_ARGS=(
  scripts/precompute_gnn_cache.py
  --config "${CONFIG}"
  --data-dir "${GT_DIR}"
  --data-format arc
  --out-dir "${CACHE_DIR}"
  --workers "${WORKERS}"
  --shard-size "${SHARD_SIZE}"
)

if [[ "${REBUILD_CACHE}" == "1" ]]; then
  PRECOMPUTE_ARGS+=(--overwrite)
fi

if [[ "${REBUILD_CACHE}" == "1" || ! -f "${CACHE_DIR}/manifest.json" ]]; then
  "${PYTHON_BIN}" "${PRECOMPUTE_ARGS[@]}"
else
  echo "Using existing cache: ${CACHE_DIR}"
  echo "Set REBUILD_CACHE=1 to rebuild it."
fi

"${PYTHON_BIN}" scripts/train_gnn.py \
  --config "${CONFIG}" \
  --data-dir "${CACHE_DIR}" \
  --data-format cache \
  --pretrained-gnn "${PRETRAINED_GNN}"
