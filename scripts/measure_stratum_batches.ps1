#Requires -Version 7.0
<#
.SYNOPSIS
Runs a bounded, serial CUDA Stratum batch-size comparison on the user's GPU.
.DESCRIPTION
This sends real shares to Suprnova. Supply only a PUBLIC qz payout address.
It never changes clocks, voltage, power limits, drivers, or security settings.
Requires PowerShell 7 and a freshly built release quantus-miner executable.
Use -SelfTest for offline summary-parser tests without starting a process.
#>
[CmdletBinding(DefaultParameterSetName = 'Sweep')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Sweep')]
    [ValidatePattern('^qz[1-9A-HJ-NP-Za-km-z]{38,78}$')]
    [string] $Wallet,
    [Parameter(ParameterSetName = 'Sweep')]
    [ValidatePattern('^[A-Za-z0-9_-]{1,32}$')]
    [string] $Worker = 'yuunyan',
    [Parameter(ParameterSetName = 'Sweep')]
    [string] $ExecutablePath = (Join-Path $PSScriptRoot '../target/release/quantus-miner.exe'),
    [Parameter(ParameterSetName = 'Sweep')]
    [string] $OutputDirectory = (Join-Path ([IO.Path]::GetTempPath()) 'quantus-batch-measurements'),
    [Parameter(ParameterSetName = 'Sweep')]
    [ValidateRange(10, 120)]
    [int] $ExitGraceSeconds = 60,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')]
    [switch] $SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertFrom-MinerSummary {
    param([string] $Text)
    $summaries = [regex]::Matches($Text, '(?m)^.*Stratum session summary:\s*Stats\s*\{([^}\r\n]+)\}\s*$')
    if ($summaries.Count -ne 1) { throw 'Expected exactly one final Stratum Stats summary.' }
    if ($summaries[0].Groups[1].Value -notmatch '^\s*\w+:\s*\d+(?:,\s*\w+:\s*\d+)*\s*$') { throw 'Malformed Stats body.' }
    $fields = [ordered]@{}
    foreach ($match in [regex]::Matches($summaries[0].Groups[1].Value, '(\w+):\s*(\d+)')) {
        $name = $match.Groups[1].Value
        if ($fields.Contains($name)) { throw 'Duplicate Stats field.' }
        $fields[$name] = [uint64]::Parse($match.Groups[2].Value, [Globalization.CultureInfo]::InvariantCulture)
    }
    $required = @('hashes', 'attempted', 'submitted', 'accepted', 'rejected', 'unacknowledged', 'stale', 'reconnects')
    if ($fields.Count -ne $required.Count) { throw 'Unexpected Stats fields.' }
    foreach ($name in $required) {
        if (-not $fields.Contains($name)) { throw "Missing Stats field: $name" }
    }
    if ($fields['attempted'] -lt $fields['submitted'] -or
        $fields['submitted'] -lt ($fields['accepted'] + $fields['rejected'])) {
        throw 'Inconsistent share accounting.'
    }
    return [pscustomobject] $fields
}

function Invoke-BoundedProcess {
    param([string] $File, [string[]] $Arguments, [int] $DeadlineSeconds, [string] $WorkingDirectory,
        [string] $StdoutPath = '', [string] $StderrPath = '')
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $File
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.WorkingDirectory = $WorkingDirectory
    foreach ($argument in $Arguments) { $start.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    $clock = [Diagnostics.Stopwatch]::new()
    $started = $false
    $timedOut = $false
    $stdout = ''
    $stderr = ''
    $exitCode = $null
    try {
        $started = $process.Start()
        if (-not $started) { throw 'Could not start child process.' }
        $clock.Start()
        # Drain BOTH pipes concurrently; never pipeline the miner's output.
        $outTask = $process.StandardOutput.ReadToEndAsync()
        $errTask = $process.StandardError.ReadToEndAsync()
        while (-not $process.WaitForExit(250)) {
            if ($clock.Elapsed.TotalSeconds -ge $DeadlineSeconds) {
                $timedOut = $true
                break
            }
        }
    }
    finally {
        if ($started) {
            if (-not $process.HasExited) {
                # Also runs when Ctrl+C interrupts the script. Kill descendants
                # as a best-effort cleanup; an unresponsive driver can prevent exit.
                try { $process.Kill($true) } catch { Write-Warning 'Could not kill child tree; check Task Manager.' }
                if (-not $process.WaitForExit(10000)) {
                    Write-Warning 'Child did not exit after kill; check Task Manager.'
                }
            }
            $clock.Stop()
            if ($process.HasExited) { $exitCode = $process.ExitCode }
            $tasks = [Threading.Tasks.Task[]] @($outTask, $errTask)
            if ([Threading.Tasks.Task]::WaitAll($tasks, 10000)) {
                $stdout = $outTask.GetAwaiter().GetResult()
                $stderr = $errTask.GetAwaiter().GetResult()
            } else {
                $timedOut = $true
                $stderr = 'Output readers did not finish after child termination.'
            }
        }
        # Preserve partial raw logs even when Ctrl+C interrupts the caller.
        if ($StdoutPath) { $stdout | Set-Content -LiteralPath $StdoutPath -Encoding utf8 }
        if ($StderrPath) { $stderr | Set-Content -LiteralPath $StderrPath -Encoding utf8 }
        $process.Dispose()
    }
    return [pscustomobject]@{
        ExitCode = $exitCode; TimedOut = $timedOut
        ElapsedSeconds = $clock.Elapsed.TotalSeconds
        Stdout = $stdout; Stderr = $stderr
    }
}

function Read-GpuSnapshot {
    param([string] $WorkingDirectory)
    $command = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) { return [pscustomobject]@{ Available = $false; Reason = 'nvidia-smi not found' } }
    try {
        $capture = Invoke-BoundedProcess $command.Source @(
            '--query-gpu=name,uuid,driver_version,pci.bus_id,temperature.gpu,power.draw,power.limit',
            '--format=csv'
        ) 10 $WorkingDirectory
        return [pscustomobject]@{
            Available = ($capture.ExitCode -eq 0 -and -not $capture.TimedOut)
            Csv = $capture.Stdout; Error = $capture.Stderr
        }
    } catch { return [pscustomobject]@{ Available = $false; Reason = $_.Exception.Message } }
}

if ($SelfTest) {
    $valid = 'Stratum session summary: Stats { hashes: 12520000000, attempted: 3, submitted: 3, accepted: 3, rejected: 0, unacknowledged: 0, stale: 0, reconnects: 0 }'
    $parsed = ConvertFrom-MinerSummary $valid
    if ($parsed.hashes -ne 12520000000 -or $parsed.accepted -ne 3) { throw 'Valid parser fixture failed.' }
    foreach ($invalid in @('', 'Stats { hashes: 1 }', "$valid`n$valid", $valid.Replace('attempted: 3', 'attempted: 0'), $valid.Replace('stale: 0, ', ''), $valid.Replace('reconnects: 0', 'reconnects: 0 trailing-junk'))) {
        $rejected = $false
        try { $null = ConvertFrom-MinerSummary $invalid } catch { $rejected = $true }
        if (-not $rejected) { throw 'Invalid parser fixture was accepted.' }
    }
    Write-Host 'Offline summary parser self-test passed; no process or pool connection started.'
    return
}

$exe = (Resolve-Path -LiteralPath $ExecutablePath).Path
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
$revision = $null
$dirty = $null
$git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $git) {
    $revResult = Invoke-BoundedProcess $git.Source @('-C', $repo, 'rev-parse', 'HEAD') 10 $repo
    if ($revResult.ExitCode -eq 0) { $revision = $revResult.Stdout.Trim() }
    $dirtyResult = Invoke-BoundedProcess $git.Source @('-C', $repo, 'status', '--porcelain') 10 $repo
    if ($dirtyResult.ExitCode -eq 0) { $dirty = (-not [string]::IsNullOrWhiteSpace($dirtyResult.Stdout)) }
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 6)
$output = Join-Path ([IO.Path]::GetFullPath($OutputDirectory)) $stamp
$null = New-Item -ItemType Directory -Path $output -Force
$runs = [Collections.Generic.List[object]]::new()
$metadata = [ordered]@{
    CreatedUtc = [DateTime]::UtcNow.ToString('o'); PowerShellVersion = $PSVersionTable.PSVersion.ToString()
    Executable = $exe; ExecutableSha256 = $sha256; GitRevision = $revision; GitWorkingTreeDirty = $dirty
    CudaVisibleDevices = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES')
    PoolHost = 'quantus.suprnova.cc'; PoolPort = 7074; Worker = $Worker
    DurationSeconds = 120; ExitGraceSeconds = $ExitGraceSeconds
    Batches = @(1000000, 4000000, 8000000, 16000000, 1000000)
    InitialGpu = Read-GpuSnapshot $repo
    Runs = $runs
}
function Save-Results {
    $metadata | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'results.json') -Encoding utf8
    if ($runs.Count -gt 0) {
        $runs | Select-Object Run, BatchSize, Status, ElapsedSeconds, EffectivePhysicalMHps, Hashes, Attempted, Submitted, Accepted, Rejected, Unacknowledged, Stale, Reconnects, ExitCode, Failure |
            Export-Csv -LiteralPath (Join-Path $output 'results.csv') -NoTypeInformation -Encoding utf8
    }
}
Save-Results
Write-Host "Results: $output"
Write-Host 'Starting five serial 120-second real-mining trials. Ctrl+C stops the script and requests child-tree termination.'

foreach ($batch in $metadata.Batches) {
    $number = $runs.Count + 1
    $name = '{0:D2}-batch-{1}' -f $number, $batch
    $record = [ordered]@{
        Run = $number; BatchSize = $batch; Status = 'running'; StartedUtc = [DateTime]::UtcNow.ToString('o')
        ElapsedSeconds = $null; EffectivePhysicalMHps = $null; Hashes = $null; Attempted = $null; Submitted = $null
        Accepted = $null; Rejected = $null; Unacknowledged = $null; Stale = $null; Reconnects = $null
        ExitCode = $null; Failure = $null; GpuBefore = Read-GpuSnapshot $repo; GpuAfter = $null
        StdoutLog = "$name.stdout.log"; StderrLog = "$name.stderr.log"
    }
    $runs.Add([pscustomobject] $record)
    $run = $runs[$runs.Count - 1]
    Save-Results
    Write-Host "Run $number/5: batch $batch, duration 120s"
    try {
        if ((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash -ne $sha256) { throw 'Executable changed during the sweep.' }
        $capture = Invoke-BoundedProcess $exe @(
            'stratum', '--wallet', $Wallet, '--worker', $Worker, '--engine', 'cuda',
            '--pool-host', 'quantus.suprnova.cc', '--pool-port', '7074',
            '--duration', '120', '--gpu-batch-size', "$batch", '--reconnect-attempts', '0'
        ) (120 + $ExitGraceSeconds) $repo -StdoutPath (Join-Path $output $run.StdoutLog) -StderrPath (Join-Path $output $run.StderrLog)
        $capture.Stdout | Set-Content -LiteralPath (Join-Path $output $run.StdoutLog) -Encoding utf8
        $capture.Stderr | Set-Content -LiteralPath (Join-Path $output $run.StderrLog) -Encoding utf8
        $run.ExitCode = $capture.ExitCode
        $run.ElapsedSeconds = [Math]::Round($capture.ElapsedSeconds, 3)
        if ($capture.TimedOut) { throw 'Child exceeded the hard deadline; its process tree was terminated.' }
        if ($capture.ExitCode -ne 0) { throw "Miner exited unsuccessfully: $($capture.ExitCode)" }
        $text = $capture.Stdout + "`n" + $capture.Stderr
        $stats = ConvertFrom-MinerSummary $text
        foreach ($field in $stats.PSObject.Properties) { $run.($field.Name) = $field.Value }
        if ($stats.reconnects -ne 0 -or $stats.rejected -ne 0 -or $stats.unacknowledged -ne 0) {
            throw 'Reconnects, rejected shares, or unacknowledged shares make this trial unsuitable for comparison.'
        }
        if ($text -match '(?i)invalid share|refusing invalid|incorrect nonce|duplicate share|worker panicked|keepalive rejected') {
            throw 'The raw log contains a correctness/protocol failure.'
        }
        if ($stats.hashes -eq 0 -or $text -notmatch 'Session duration reached; stopping pool worker') {
            throw 'Missing successful bounded mining completion.'
        }
        $run.EffectivePhysicalMHps = [Math]::Round($stats.hashes / $capture.ElapsedSeconds / 1000000.0, 3)
        $run.Status = 'passed'
        Write-Host "  $($run.EffectivePhysicalMHps) effective physical MH/s; accepted $($run.Accepted), stale $($run.Stale)"
    } catch {
        $run.Status = 'failed'; $run.Failure = $_.Exception.Message
        throw
    } finally {
        $run.GpuAfter = Read-GpuSnapshot $repo
        Save-Results
    }
}
Write-Host "Five trials completed. Compare results.csv and the two 1M baselines; raw logs remain in $output"
