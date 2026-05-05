#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m biodcase_edge.cli.train --config-name debug "$@"

