# =============================================================================
# fetch_data.ps1 — Windows companion to fetch_data.sh
# =============================================================================
# Downloads the pre-built Edge-RAG stores (LanceDB / KuzuDB / BM25 +
# questions.json) from the Zenodo artifact (DOI 10.5281/zenodo.20807936) and
# unpacks them so the per-dataset stores resolve as:
#   data\<dataset>\{vector,graph,questions.json}
# (<dataset> = hotpotqa | 2wikimultihop | musique | strategyqa) — the layout
# demo_app.py and the evaluation suite read by default.
#
# Usage (no args — uses the pinned Zenodo URL below):
#   .\scripts\fetch_data.ps1
#
# Override the source if needed:
#   $env:ZENODO_URL = "https://zenodo.org/records/<ID>/files/edge-rag-stores.zip?download=1"
#   .\scripts\fetch_data.ps1
# =============================================================================
$ErrorActionPreference = "Stop"

# Pinned Zenodo artifact (DOI 10.5281/zenodo.20807936). Override via $env:ZENODO_URL.
$DefaultZenodoUrl = "https://zenodo.org/records/20807936/files/edge-rag-stores.zip?download=1"
$ZenodoUrl = if ($env:ZENODO_URL) { $env:ZENODO_URL } else { $DefaultZenodoUrl }

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $RepoRoot "data"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Write-Host "Fetching pre-built stores from:`n  $ZenodoUrl"
Write-Host "Into:`n  $Dest"

$Archive = Join-Path $env:TEMP "edge-rag-stores.zip"
Invoke-WebRequest -Uri $ZenodoUrl -OutFile $Archive

if ($env:ZENODO_SHA256) {
    Write-Host "Verifying SHA-256..."
    $actual = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLower()
    if ($actual -ne $env:ZENODO_SHA256.ToLower()) {
        Remove-Item $Archive -Force
        Write-Error "SHA-256 mismatch: expected $($env:ZENODO_SHA256), got $actual"
        exit 1
    }
}

Write-Host "Unpacking..."
# The artifact is a .zip whose top-level entries are the per-dataset folders.
Expand-Archive -Path $Archive -DestinationPath $Dest -Force
Remove-Item $Archive -Force

Write-Host "`nDone. Datasets now available under $Dest:"
Get-ChildItem -Name $Dest
Write-Host "`nNext: run the demo, e.g."
Write-Host '  python -u demo_app.py --question "Who directed the film Ed Wood?"'
