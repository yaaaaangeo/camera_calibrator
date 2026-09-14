#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

if [[ -x "${script_dir}/venv/bin/python" ]]; then
    python_cmd=("${script_dir}/venv/bin/python")
elif [[ -x "${script_dir}/.venv/bin/python" ]]; then
    python_cmd=("${script_dir}/.venv/bin/python")
elif command -v python3 >/dev/null 2>&1; then
    python_cmd=(python3)
elif command -v python >/dev/null 2>&1; then
    python_cmd=(python)
else
    echo "Python was not found."
    echo "Install Python 3.10 or newer, then try again."
    exit 1
fi

if ! "${python_cmd[@]}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is required."
    echo "Selected Python: ${python_cmd[*]}"
    exit 1
fi

"${python_cmd[@]}" -m app.main
