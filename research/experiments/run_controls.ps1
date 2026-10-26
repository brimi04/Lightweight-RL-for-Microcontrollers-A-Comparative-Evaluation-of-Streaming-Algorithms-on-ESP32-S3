<#
run_controls.ps1 -- Runs ONE reproducibility control with the original .venv.
It does NOT run full ablations (X1/X2) or X3.

Usage (from any folder):
  powershell -NoProfile -ExecutionPolicy Bypass -File <repository>\research\experiments\run_controls.ps1 -Control C0
  ... -Control C1      (requires a valid C0_PASSED.json)
  ... -Control C2      (requires a valid C1_PASSED.json)
  Add -DryRun to check configuration and paths without running episodes.
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet('C0', 'C1', 'C2')][string]$Control,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$Research = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Research '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { throw ".venv Python not found: $Py" }
$ver = (& $Py -c "import sys; print(sys.version.split()[0])").Trim()
if ($ver -ne '3.10.0') { throw "Unexpected .venv version: $ver (expected 3.10.0)" }
$np = (& $Py -c "import numpy, gymnasium; print(numpy.__version__ + ' ' + gymnasium.__version__)").Trim()
if ($np -ne '2.2.6 0.29.1') { throw "Unexpected numpy/gymnasium in the .venv: $np (expected '2.2.6 0.29.1')" }

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$cmdLine = "run_controls.ps1 -Control $Control" + $(if ($DryRun) { ' -DryRun' } else { '' })
$env:CONTROL_COMMAND = $cmdLine

switch ($Control) {
    'C0' { $pyArgs = @('-W', 'ignore', (Join-Path $PSScriptRoot 'run_controls.py'), '--control', 'C0') }
    'C1' { $pyArgs = @('-W', 'ignore', (Join-Path $PSScriptRoot 'run_ablations_cartpole.py'), '--stage', 'control') }
    'C2' { $pyArgs = @('-W', 'ignore', (Join-Path $PSScriptRoot 'run_controls.py'), '--control', 'C2') }
}
if ($DryRun) { $pyArgs += '--dry-run' }

$LogDir = Join-Path $Research 'results\derived\controls\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $LogDir ("{0}{1}_{2}.log" -f $Control, $(if ($DryRun) { '_dryrun' } else { '' }), $stamp)
if (Test-Path $Log) { throw "The log already exists: $Log" }
"# $cmdLine | start UTC $stamp | python $ver | numpy/gymnasium $np" | Out-File -FilePath $Log -Encoding utf8

Write-Host "Running $cmdLine  (log: $Log)"
$ErrorActionPreference = 'Continue'
& $Py @pyArgs 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -Path $Log -Value $line -Encoding utf8 }
$code = $LASTEXITCODE
"# end UTC $((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')) | exit code $code" | Add-Content -Path $Log -Encoding utf8
Write-Host "Control $Control finished with code $code"
exit $code
