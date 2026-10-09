#Requires -Version 7.0
<# Packages an explicitly selected existing executable; does not build or modify target/release. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ExecutablePath,
    [string] $OutputDirectory = (Join-Path ([IO.Path]::GetTempPath()) ('quantus-stratum-kit-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff')))
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$executable = (Resolve-Path -LiteralPath $ExecutablePath).Path
if (Test-Path -LiteralPath $executable -PathType Container) { throw 'ExecutablePath must be a file.' }
if ([IO.Path]::GetFileName($executable) -cnotin @('quantus-miner','quantus-miner.exe')) { throw 'Select a quantus-miner or quantus-miner.exe executable built for the destination OS.' }
$destination = [IO.Path]::GetFullPath($OutputDirectory)
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
if ($destination.Equals($repository,$comparison) -or $destination.StartsWith($repository + [IO.Path]::DirectorySeparatorChar,$comparison)) { throw 'Package outside the repository to preserve baseline artifacts.' }
if (Test-Path -LiteralPath $destination) { throw 'OutputDirectory already exists; choose a new directory.' }
$null = New-Item -ItemType Directory -Path $destination
$null = New-Item -ItemType Directory -Path (Join-Path $destination 'scripts')
$null = New-Item -ItemType Directory -Path (Join-Path $destination 'examples')
$binaryName = [IO.Path]::GetFileName($executable)
Copy-Item -LiteralPath $executable -Destination (Join-Path $destination $binaryName)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'start_stratum.ps1') -Destination (Join-Path $destination 'scripts/start_stratum.ps1')
Copy-Item -LiteralPath (Join-Path $repository 'examples/stratum/config.example.json') -Destination (Join-Path $destination 'examples/config.example.json')
Copy-Item -LiteralPath (Join-Path $repository 'docs/stratum-operations.md') -Destination (Join-Path $destination 'README.md')
$entries = @(Get-ChildItem -LiteralPath $destination -File -Recurse | Sort-Object FullName | ForEach-Object {
    [ordered]@{ path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/'); bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$revision = $null
$dirty = $null
if (Get-Command git -ErrorAction SilentlyContinue) {
    $candidate = & git -C $repository rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $candidate -match '^[a-fA-F0-9]{40,64}$') {
        $revision = [string]$candidate
        $status = @(& git -C $repository status --porcelain 2>$null)
        if ($LASTEXITCODE -eq 0) { $dirty = ($status.Count -gt 0) }
    }
}
$manifest = [ordered]@{ formatVersion=1; createdUtc=[DateTime]::UtcNow.ToString('o'); executable=$binaryName; sourceRevision=$revision; sourceDirty=$dirty; packageHostOS=[Runtime.InteropServices.RuntimeInformation]::OSDescription; files=$entries }
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $destination 'manifest.json') -Encoding utf8NoBOM
foreach ($entry in $entries) {
    if ((Get-FileHash -LiteralPath (Join-Path $destination $entry.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Package checksum verification failed.' }
}
Write-Output "Verified distribution kit: $destination"
Write-Output "Executable SHA256: $($entries.Where({$_.path -eq $binaryName})[0].sha256)"
