#!/usr/bin/env bash
# =============================================================================
# fetch_data.sh — download the pre-built Edge-RAG stores into ./data/
# =============================================================================
# The CPU runtime image does NOT bake in the LanceDB / KuzuDB / BM25 stores;
# they are produced once by the GPU extraction stage and shipped as a DATA
# artifact (Zenodo, DOI 10.5281/zenodo.20807936). This script fetches that
# artifact and unpacks it so the per-dataset stores resolve as:
#
#     data/<dataset>/vector/        (LanceDB)
#     data/<dataset>/graph/         (KuzuDB)
#     data/<dataset>/questions.json
#     data/<dataset>/articles_info.json
#
# (<dataset> = hotpotqa | 2wikimultihop | musique | strategyqa). This is the
# layout demo_app.py and the evaluation suite read by default.
#
# Usage (no args — uses the pinned Zenodo URL below):
#     ./scripts/fetch_data.sh
#
# Override the source if needed:
#     ZENODO_URL="https://zenodo.org/records/<ID>/files/edge-rag-stores.zip?download=1" \
#         ./scripts/fetch_data.sh
# =============================================================================
set -euo pipefail

# Pinned Zenodo artifact (DOI 10.5281/zenodo.20807936). Override via $ZENODO_URL.
DEFAULT_ZENODO_URL="https://zenodo.org/records/20807936/files/edge-rag-stores.zip?download=1"
ZENODO_URL="${ZENODO_URL:-$DEFAULT_ZENODO_URL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/data"

if [[ -z "$ZENODO_URL" ]]; then
    echo "ERROR: no artifact URL set." >&2
    echo "  Set ZENODO_URL=... (or edit DEFAULT_ZENODO_URL in this script)." >&2
    echo "  The artifact must unpack to <dataset>/{vector,graph,questions.json}." >&2
    exit 1
fi

mkdir -p "$DEST"

echo "Fetching pre-built stores from:"
echo "  $ZENODO_URL"
echo "Into:"
echo "  $DEST"

ARCHIVE="$(mktemp -t edge-rag-stores.XXXXXX.zip)"
trap 'rm -f "$ARCHIVE"' EXIT

curl -fL --retry 3 -o "$ARCHIVE" "$ZENODO_URL"

# Optional integrity check: export ZENODO_SHA256 to verify the download.
if [[ -n "${ZENODO_SHA256:-}" ]]; then
    echo "Verifying SHA-256…"
    echo "${ZENODO_SHA256}  ${ARCHIVE}" | sha256sum -c -
fi

echo "Unpacking…"
# The artifact is a .zip whose top-level entries are the per-dataset folders
# (hotpotqa/, 2wikimultihop/, …); unpack them directly under data/.
unzip -oq "$ARCHIVE" -d "$DEST"

echo
echo "Done. Datasets now available under $DEST:"
ls -1 "$DEST"
echo
echo "Next: run the demo, e.g."
echo "  python -u demo_app.py --question \"Who directed the film Ed Wood?\""
