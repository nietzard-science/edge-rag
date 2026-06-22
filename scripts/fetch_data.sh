#!/usr/bin/env bash
# =============================================================================
# fetch_data.sh — download the pre-built Edge-RAG stores into ./data/indices
# =============================================================================
# The CPU runtime image does NOT bake in the LanceDB / KuzuDB / BM25 stores;
# they are produced once by the GPU extraction stage and shipped as a DATA
# artifact (Zenodo). This script fetches that artifact and lays it out the way
# StoreManager expects:
#
#     data/indices/<dataset>/vector/        (LanceDB)
#     data/indices/<dataset>/graph/         (KuzuDB)
#     data/indices/<dataset>/questions.json
#     data/indices/<dataset>/articles_info.json
#
# docker-compose.yml mounts data/indices -> /app/data/indices (= INDEX_DIR), so
# after this script + `docker compose up` a result reproduces with no fiddling.
#
# Usage:
#     ZENODO_URL="https://zenodo.org/records/<ID>/files/edge-rag-stores.tar.gz" \
#         ./scripts/fetch_data.sh
#
# Or pin the DOI once it is minted by editing DEFAULT_ZENODO_URL below.
# =============================================================================
set -euo pipefail

# TODO: replace with the minted Zenodo artifact URL (DOI 10.5281/zenodo.XXXX).
DEFAULT_ZENODO_URL=""
ZENODO_URL="${ZENODO_URL:-$DEFAULT_ZENODO_URL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/data/indices"

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

TARBALL="$(mktemp -t edge-rag-stores.XXXXXX.tar.gz)"
trap 'rm -f "$TARBALL"' EXIT

curl -fL --retry 3 -o "$TARBALL" "$ZENODO_URL"

# Optional integrity check: if SHA256.txt sits next to the tarball URL, verify.
if [[ -n "${ZENODO_SHA256:-}" ]]; then
    echo "Verifying SHA-256…"
    echo "${ZENODO_SHA256}  ${TARBALL}" | sha256sum -c -
fi

echo "Unpacking…"
tar -xzf "$TARBALL" -C "$DEST"

echo
echo "Done. Datasets now available under $DEST:"
ls -1 "$DEST"
echo
echo "Next: docker compose up   (then: docker compose run --rm app evaluate --dataset hotpotqa --range 0-5)"
