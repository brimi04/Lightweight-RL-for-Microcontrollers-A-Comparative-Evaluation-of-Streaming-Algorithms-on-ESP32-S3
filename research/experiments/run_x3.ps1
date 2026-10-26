<#
run_x3.ps1 -- Runs ONLY X3 (Stream-AC/MountainCar, documented
configuration) with the original .venv. It accepts no other value: it cannot
run C0/C1/C2 or X1/X2 from this script.

Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File <repository>\research\experiments\run_x3.ps1
  Add -DryRun to check configuration, memory and FLOPs without running episodes.
#>
param(
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
$cmdLine = "run_x3.ps1" + $(if ($DryRun) { ' -DryRun' } else { '' })
$env:CONTROL_COMMAND = $cmdLine

$pyArgs = @('-W', 'ignore', (Join-Path $PSScriptRoot 'run_x3_streamac_mountaincar.py'), '--stage', 'run')
if ($DryRun) { $pyArgs += '--dry-run' }

$LogDir = Join-Path $Research 'results\derived\controls\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $LogDir ("X3{0}_{1}.log" -f $(if ($DryRun) { '_dryrun' } else { '' }), $stamp)
if (Test-Path $Log) { throw "The log already exists: $Log" }
"# $cmdLine | start UTC $stamp | python $ver | numpy/gymnasium $np" | Out-File -FilePath $Log -Encoding utf8

Write-Host "Running $cmdLine  (log: $Log)"
$ErrorActionPreference = 'Continue'
& $Py @pyArgs 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -Path $Log -Value $line -Encoding utf8 }
$code = $LASTEXITCODE
"# end UTC $((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')) | exit code $code" | Add-Content -Path $Log -Encoding utf8
Write-Host "X3 finished with code $code"
exit $code
