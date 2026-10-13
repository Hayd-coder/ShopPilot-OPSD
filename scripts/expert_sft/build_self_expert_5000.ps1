param(
    [string]$PythonExe = "python",
    [string]$JavaHome = $env:JAVA_HOME,
    [int]$TargetTrajectories = 5000,
    [int]$SelectionSeed = 20261012
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ArtifactRoot = Join-Path $RepoRoot "datasets\generated"
$Generator = Join-Path $PSScriptRoot "generate_oracle_webshop.py"

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $PythonExe"
}
if (-not $JavaHome) {
    throw "JAVA_HOME is not set. Point it to a Java 11 installation."
}
if (-not (Test-Path -LiteralPath (Join-Path $JavaHome "bin\java.exe") -PathType Leaf)) {
    throw "Java executable not found under JAVA_HOME: $JavaHome"
}

New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
$env:JAVA_HOME = $JavaHome
$env:Path = "$JavaHome\bin;$env:Path"
$env:PYTHONUTF8 = "1"

& $PythonExe $Generator `
    --target-successes $TargetTrajectories `
    --max-attempts 6408 `
    --max-goals-per-asin 1000 `
    --max-search-pages 5 `
    --min-visible-title-overlap 0 `
    --max-visible-rank-gap 999 `
    --comparison-fraction 0 `
    --selection-seed $SelectionSeed `
    --allow-heldout-asin-overlap `
    --no-reference-strata-only `
    --output (Join-Path $ArtifactRoot "self_expert_5000.jsonl") `
    --failure-output (Join-Path $ArtifactRoot "self_expert_5000_failures.jsonl") `
    --manifest (Join-Path $ArtifactRoot "self_expert_5000_manifest.json")

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
