#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Prefer existing virtual env in this order: .venv, venv, env
if [ -d "$PROJECT_DIR/.venv" ]; then
  VENV_DIR="$PROJECT_DIR/.venv"
elif [ -d "$PROJECT_DIR/venv" ]; then
  VENV_DIR="$PROJECT_DIR/venv"
elif [ -d "$PROJECT_DIR/env" ]; then
  VENV_DIR="$PROJECT_DIR/env"
else
  VENV_DIR="$PROJECT_DIR/env"
  echo "Creating virtual environment at $VENV_DIR..."
  python -m venv "$VENV_DIR"
fi

if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || "${OS:-}" == "Windows_NT" ]]; then
  # Git Bash on Windows may still use Unix-style activation
  # Prefer POSIX activation if present
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate" 2>/dev/null || source "$VENV_DIR/bin/activate"
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

pip install -r "$PROJECT_DIR/requirements.txt"

mkdir -p "$PROJECT_DIR/pdf"

python "$PROJECT_DIR/main_rag_chat.py" "$@"


