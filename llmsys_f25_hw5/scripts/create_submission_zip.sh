#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/llmsys_hw5_submission_${TIMESTAMP}.zip"

mkdir -p "${OUT_DIR}"

cd "${ROOT_DIR}"
zip -r "${OUT_FILE}" \
  README.md \
  requirements.txt \
  pytest.ini \
  data_parallel \
  pipeline \
  project \
  tests \
  scripts \
  submit_figures \
  -x "*/__pycache__/*" \
     "*.pyc" \
      "tests/model*_gradients.pth" \
     ".git/*" \
     ".venv/*" \
     ".pytest_cache/*" \
     "workdir/*" \
     "dist/*"

echo "Created submission zip: ${OUT_FILE}"