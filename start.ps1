param(
    [ValidateSet('clap', 'demo', 'aligned')][string]$Backend = 'clap',
    [int]$Port = 8765,
    [string]$Config = '',
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Create .venv and install the project first; see README.md.'
    }
    $launchArguments = @('-m', 'sonic_bridge', 'serve', '--backend', $Backend, '--port', "$Port")
    if ($Config) { $launchArguments += @('--config', $Config) }
    if ($Offline) { $launchArguments += '--offline' }
    & $pythonPath @launchArguments
    if ($LASTEXITCODE -ne 0) { throw "Sonic Bridge exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
