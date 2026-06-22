# =============================================================================
# fetch_data.ps1 — Windows companion to fetch_data.sh
# =============================================================================
# Downloads the pre-built Edge-RAG stores (LanceDB / KuzuDB / BM25 +
# questions.json) from the Zenodo artifact into data\indices, laid out the way
# StoreManager expects:  data\indices\<dataset>\{vector,graph,questions.json}.
#
# Usage:
#   $env:ZENODO_URL = "https://zenodo.org/records/<ID>/files/edge-rag-stores.tar.gz"
#   .\scripts\fetch_data.ps1
# =============================================================================
$ErrorActionPreference = "Stop"

# TODO: replace with the minted Zenodo artifact URL (DOI 10.5281/zenodo.XXXX).
$DefaultZenodoUrl = ""
$ZenodoUrl = if ($env:ZENODO_URL) { $env:ZENODO_URL } else { $DefaultZenodoUrl }

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $RepoRoot "data\indices"

if ([string]::IsNullOrWhiteSpace($ZenodoUrl)) {
    Write-Error "No artifact URL set. Set `$env:ZENODO_URL=... (or edit `$DefaultZenodoUrl in this script). The artifact must unpack to <dataset>\{vector,graph,questions.json}."
    exit 1
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Write-Host "Fetching pre-built stores from:`n  $ZenodoUrl"
Write-Host "Into:`n  $Dest"

$Tarball = Join-Path $env:TEMP "edge-rag-stores.tar.gz"
Invoke-WebRequest -Uri $ZenodoUrl -OutFile $Tarball

if ($env:ZENODO_SHA256) {
    Write-Host "Verifying SHA-256..."
    $actual = (Get-FileHash -Algorithm SHA256 $Tarball).Hash.ToLower()
    if ($actual -ne $env:ZENODO_SHA256.ToLower()) {
        Remove-Item $Tarball -Force
        Write-Error "SHA-256 mismatch: expected $($env:ZENODO_SHA256), got $actual"
        exit 1
    }
}

Write-Host "Unpacking..."
# tar ships with Windows 10/11.
tar -xzf $Tarball -C $Dest
Remove-Item $Tarball -Force

Write-Host "`nDone. Datasets now available under $Dest:"
Get-ChildItem -Name $Dest
Write-Host "`nNext: docker compose up   (then: docker compose run --rm app evaluate --dataset hotpotqa --range 0-5)"
