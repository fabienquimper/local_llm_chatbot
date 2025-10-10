#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python -m venv "$VENV_DIR"
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OS" == "Windows_NT" ]]; then
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


