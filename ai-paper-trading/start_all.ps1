# Start the worker and API together using the project's sibling virtual environment.
$venvPython = Join-Path $PSScriptRoot ".\.venv\Scripts\python.exe"
$venvUvicorn = Join-Path $PSScriptRoot ".\.venv\Scripts\uvicorn.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual-environment Python was not found: $venvPython"
}

if (-not (Test-Path $venvUvicorn)) {
    throw "Virtual-environment Uvicorn was not found: $venvUvicorn"
}

#$worker = Start-Process -FilePath $venvPython -ArgumentList "run_worker.py" -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru

try {
    & $venvUvicorn app.main:app --reload
}
finally {
    if (-not $worker.HasExited) {
        Stop-Process -Id $worker.Id
    }
}
