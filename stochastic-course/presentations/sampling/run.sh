#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/src"

uv run ./acceptance_rejection_beta22.py
uv run ./acceptance_rejection_bimodal.py
uv run ./acceptance_rejection_pi.py
uv run ./importance_sampling_triangular.py
uv run ./importance_sampling_tail.py
uv run ./metropolis_hastings_bimodal.py
uv run ./metropolis_hastings_triangular.py
