#!/usr/bin/env bash
# run_eval.sh — Run the full evaluation pipeline
# Usage: bash scripts/run_eval.sh [mock|api|anthropic-only]

set -euo pipefail

MODE="${1:-mock}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> AI Assistant Evaluation Pipeline"
echo "==> Mode: ${MODE}"
echo "==> Project: ${PROJECT_DIR}"

cd "$PROJECT_DIR/evaluation"

# Install eval dependencies
pip install anthropic requests --quiet

echo ""
echo "==> Running evaluation (60 prompts)..."
python eval_runner.py \
  --mode "$MODE" \
  --oss-url "${OSS_URL:-http://localhost:7860}" \
  --frontier-url "${FRONTIER_URL:-http://localhost:7861}" \
  --delay 0.5

echo ""
echo "==> Results saved to evaluation/results/"
echo "==> Done! ✅"
