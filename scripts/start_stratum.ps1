#Requires -Version 7.0
<# Starts one bounded TLS mining session. ConfigPath must be outside this kit/repository.
   Supply only a public payout address. No schedules or GPU/OS settings are changed. #>
[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Run')][string] $ConfigPath,
    [Parameter(ParameterSetName = 'Run')][string] $ExecutablePath = '',
    [Parameter(ParameterSetName = 'Run')][ValidateRange(10,120)][int] $ExitGraceSeconds = 60,
    [Parameter(ParameterSetName = 'Run')][switch] $DryRun,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')][switch] $SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-MinerArguments {
    param([System.Collections.IDictionary] $Config)
    $allowed = @('wallet','worker','duration','engine','poolHost','poolPort','fallbackPools','cudaDevices','gpuBatchSize','gpuThrottleMs','reconnectAttempts','gpuTelemetryInterval')
    foreach ($key in $Config.Keys) { if ($key -cnotin $allowed) { throw "Unknown configuration key: $key" } }
    if (-not $Config.Contains('wallet') -or $Config.wallet -isnot [string] -or $Config.wallet -cnotmatch '^qz[1-9A-HJ-NP-Za-km-z]{38,78}$') { throw 'wallet must be a public qz payout address, never a private key or recovery phrase.' }
    $defaults = @{ worker='rig1'; duration=120; engine='cuda'; poolHost='quantus.suprnova.cc'; poolPort=7074; fallbackPools=@(); cudaDevices='0'; gpuBatchSize=1000000; gpuThrottleMs=0; reconnectAttempts=3; gpuTelemetryInterval=0 }
    foreach ($key in $defaults.Keys) { if (-not $Config.Contains($key)) { $Config[$key] = $defaults[$key] } }
    if ($Config.worker -isnot [string] -or $Config.worker -cnotmatch '^[A-Za-z0-9_-]{1,32}$') { throw 'Invalid worker label.' }
    if ($Config.engine -cnotin @('cpu','cuda')) { throw 'engine must be cpu or cuda.' }
    if ($Config.poolHost -isnot [string] -or $Config.poolHost -cnotmatch '^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$') { throw 'poolHost must be a TLS hostname without scheme/path/port.' }
    $ranges = @{ duration=@(1,86400); poolPort=@(1,65535); gpuBatchSize=@(1,4294967295); gpuThrottleMs=@(0,60000); reconnectAttempts=@(0,20); gpuTelemetryInterval=@(0,3600) }
    foreach ($key in $ranges.Keys) {
        $value = $Config[$key]
        if (($value -isnot [int] -and $value -isnot [long]) -or $value -lt $ranges[$key][0] -or $value -gt $ranges[$key][1]) { throw "Invalid integer/range for $key" }
    }
    if ($Config.gpuTelemetryInterval -gt 0 -and $Config.gpuTelemetryInterval -lt 5) { throw 'gpuTelemetryInterval must be 0 or 5..3600.' }
    if ($Config.cudaDevices -isnot [string] -or $Config.cudaDevices -cnotmatch '^\d+(?:,\d+)*$') { throw 'cudaDevices must contain comma-separated visible CUDA ordinals.' }
    $devices = @($Config.cudaDevices.Split(','))
    if ($devices.Count -gt 64) { throw 'cudaDevices is limited to 64 ordinals.' }
    $seen = [Collections.Generic.HashSet[uint32]]::new()
    foreach ($device in $devices) {
        $ordinal = [uint32]0
        if (-not [uint32]::TryParse($device, [ref]$ordinal) -or -not $seen.Add($ordinal)) { throw 'cudaDevices must contain unique uint32 ordinals.' }
    }
    if ($Config.engine -eq 'cpu' -and $Config.cudaDevices -ne '0') { throw 'cudaDevices selection requires engine cuda.' }
    if ($Config.fallbackPools -isnot [array] -or $Config.fallbackPools.Count -gt 8) { throw 'fallbackPools must be an array with at most 8 TLS hostname:port endpoints.' }
    $arguments = [Collections.Generic.List[string]]::new()
    $arguments.Add('stratum')
    $mapping = [ordered]@{ wallet='wallet'; worker='worker'; duration='duration'; engine='engine'; poolHost='pool-host'; poolPort='pool-port'; cudaDevices='cuda-devices'; gpuBatchSize='gpu-batch-size'; gpuThrottleMs='gpu-throttle-ms'; reconnectAttempts='reconnect-attempts'; gpuTelemetryInterval='gpu-telemetry-interval' }
    foreach ($key in $mapping.Keys) { $arguments.Add("--$($mapping[$key])"); $arguments.Add([string]$Config[$key]) }
    foreach ($endpoint in $Config.fallbackPools) {
        if ($endpoint -isnot [string] -or $endpoint -cnotmatch '^(?=.{3,259}$)[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?:([0-9]{1,5})$' -or [int]$Matches[1] -notin 1..65535) { throw 'Each fallback must be an explicit TLS hostname:port endpoint.' }
        $arguments.Add('--fallback-pool'); $arguments.Add($endpoint)
    }
    return ,$arguments.ToArray()
}

if ($SelfTest) {
    $wallet = 'qz' + ('1' * 40)
    $valid = @{ wallet=$wallet; duration=120; fallbackPools=@('backup.example:7074') }
    $arguments = ConvertTo-MinerArguments $valid
    if ($arguments[0] -ne 'stratum' -or '--fallback-pool' -notin $arguments) { throw 'Argument mapping test failed.' }
    foreach ($bad in @(@{wallet=$wallet;duration=0}, @{wallet=$wallet;duration=86401}, @{wallet=$wallet;gpuTelemetryInterval=1}, @{wallet=$wallet;insecure=$true}, @{wallet=$wallet;poolHost='stratum://bad'}, @{wallet=$wallet;fallbackPools=@('bad:0')}, @{wallet='secret phrase'}, @{wallet=$wallet;cudaDevices='0,0'}, @{wallet=$wallet;cudaDevices='0,00'}, @{wallet=$wallet;engine='cpu';cudaDevices='1'}, @{wallet=$wallet;cudaDevices='4294967296'}, @{wallet=$wallet;cudaDevices=((0..64) -join ',')})) {
        $rejected = $false
        try { $null = ConvertTo-MinerArguments $bad } catch { $rejected = $true }
        if (-not $rejected) { throw 'Invalid configuration accepted.' }
    }
    Write-Output 'Launcher offline self-test passed (no process started).'
    return
}

$configFile = (Resolve-Path -LiteralPath $ConfigPath).Path
$kitRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
if ($configFile.StartsWith($kitRoot + [IO.Path]::DirectorySeparatorChar, $comparison)) { throw 'Keep the wallet configuration outside the repository/distribution kit.' }
$config = Get-Content -LiteralPath $configFile -Raw | ConvertFrom-Json -AsHashtable
$arguments = ConvertTo-MinerArguments $config
if ([string]::IsNullOrEmpty($ExecutablePath)) {
    $candidates = @('../quantus-miner.exe','../quantus-miner','../target/release/quantus-miner.exe','../target/release/quantus-miner')
    foreach ($candidate in $candidates) {
        $path = Join-Path $PSScriptRoot $candidate
        if (Test-Path -LiteralPath $path -PathType Leaf) { $ExecutablePath = $path; break }
    }
    if ([string]::IsNullOrEmpty($ExecutablePath)) { throw 'Specify the freshly built miner with -ExecutablePath.' }
}
$executable = (Resolve-Path -LiteralPath $ExecutablePath).Path
if (Test-Path -LiteralPath $executable -PathType Container) { throw 'ExecutablePath must be a file.' }
$hash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
Write-Output "Executable SHA256: $hash"
Write-Output "Bounded $($config.engine) session: $($config.duration)s; TLS pool $($config.poolHost):$($config.poolPort); worker $($config.worker)"
if ($DryRun) { Write-Output 'Configuration validated. DryRun starts no process and connects to no pool.'; return }
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $executable
$start.UseShellExecute = $false
$start.WorkingDirectory = [IO.Path]::GetDirectoryName($executable)
foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }
$process = [Diagnostics.Process]::new()
$process.StartInfo = $start
$started = $false
$timedOut = $false
$clock = [Diagnostics.Stopwatch]::new()
try {
    $started = $process.Start()
    if (-not $started) { throw 'Miner process did not start.' }
    $clock.Start()
    while (-not $process.WaitForExit(250)) {
        if ($clock.Elapsed.TotalSeconds -ge ([long]$config.duration + $ExitGraceSeconds)) { $timedOut = $true; break }
    }
} finally {
    if ($started -and -not $process.HasExited) {
        try { $process.Kill($true) } catch { Write-Warning 'Could not terminate miner tree; check Task Manager before another run.' }
        if (-not $process.WaitForExit(10000)) { Write-Warning 'Miner remains running; a hung driver can prevent termination.' }
    }
    $clock.Stop()
    $exitCode = if ($started -and $process.HasExited) { $process.ExitCode } else { $null }
    $process.Dispose()
}
if ($timedOut) { throw 'Miner exceeded the bounded process deadline.' }
if ($exitCode -ne 0) { throw "Miner failed with exit code $exitCode" }
