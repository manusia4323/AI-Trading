# Start the FastAPI development server with automatic reloads.
$venvUvicorn = Join-Path $PSScriptRoot "..\.venv\Scripts\uvicorn.exe"

if (-not (Test-Path $venvUvicorn)) {
    throw "Virtual-environment Uvicorn was not found: $venvUvicorn"
}

Set-Location $PSScriptRoot
& $venvUvicorn app.main:app --reload
