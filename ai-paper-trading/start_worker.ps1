# Start the worker using the project's sibling virtual environment.
$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual-environment Python was not found: $venvPython"
}

Set-Location $PSScriptRoot
& $venvPython run_worker.py
