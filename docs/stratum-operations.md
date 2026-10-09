# Bounded operational kit

This launcher starts **real mining** only when invoked without `-DryRun` or
`-SelfTest`. It never installs software, schedules future launches, changes GPU
clocks/power/voltage, or weakens TLS. No wallet, private key or recovery phrase
is included. Use only your public payout address.

## Build and distribute

Build into a separate target directory to preserve a working baseline:

```powershell
$env:CARGO_TARGET_DIR = "$HOME\Documents\QuantusBuild"
cargo build -p miner-cli --release --locked
pwsh -File .\scripts\package_stratum.ps1 -ExecutablePath "$env:CARGO_TARGET_DIR\release\quantus-miner.exe" -OutputDirectory "$HOME\Documents\QuantusKit"
```

Packaging takes an existing executable; it does not build or overwrite
`target/release`. The manifest contains SHA256 and size for every delivered
file, including the executable, launcher, example and documentation. The
manifest itself is not signed: hashes detect file changes but do not prove
publisher identity. Obtain the kit through a trusted channel and compare its
executable checksum with the sender's checksum. Required NVIDIA driver/NVRTC
libraries remain external; no driver or library installation is performed.

## Configure and start

Copy `examples/stratum/config.example.json` (repository) or
`examples/config.example.json` (kit) outside the repository/kit, e.g.
`$HOME\Documents\quantus-session.json`. Replace the placeholder with your
public qz address. Unknown configuration keys and invalid ranges fail closed.
Do not put credentials, payout recovery material, or wallet config into git.

```powershell
pwsh -File .\scripts\start_stratum.ps1 -SelfTest
pwsh -File .\scripts\start_stratum.ps1 -ConfigPath "$HOME\Documents\quantus-session.json" -ExecutablePath .\quantus-miner.exe -DryRun
pwsh -File .\scripts\start_stratum.ps1 -ConfigPath "$HOME\Documents\quantus-session.json" -ExecutablePath .\quantus-miner.exe
```

For a repository build, use the actual separate build executable path. The
launcher prints its executable checksum but never echoes the wallet. The
wallet still appears in the miner process arguments; only a public address
belongs here. stdout/stderr go directly to the invoking console.

`duration` is 1..86400 seconds, default 120. The launcher adds a 60-second
initialization/exit allowance (configurable as `-ExitGraceSeconds` 10..120),
then attempts child-tree termination. Ctrl+C also triggers best-effort
cleanup. An unresponsive GPU driver can prevent exit; check Task Manager if
cleanup warns before launching again. No automatic repeat or daily schedule
is created. The miner's duration timer starts after CUDA initialization and includes pool
connection/exit, so it does not guarantee that many seconds of useful hashing.
The launcher's separate wall-clock deadline includes all initialization.

All primary and explicit `fallbackPools` (`hostname:port`) endpoints use
certificate-verified TLS, with no plaintext downgrade. Empty fallbacks mean
no alternate pool. Choose fallback operators intentionally: your public
address, worker and shares will be sent there. `reconnectAttempts` is 0..20.
`cudaDevices` uses visible CUDA ordinals such as `0` or `0,1`; selecting several
GPUs requires this updated miner. `engine: cpu` is an explicit troubleshooting
choice, never an automatic CUDA fallback.

## Read-only continuous telemetry

`gpuTelemetryInterval` is 0 (disabled) or 5..3600 seconds. It polls independently
of share arrival and completed CUDA batches. The optional `nvidia-smi` query
reads GPU index/UUID, utilization percent, temperature C, power W, used/total
VRAM MiB, SM/memory clocks MHz, driver version and configured power limit W. It changes no settings and sends no external API requests. All
NVIDIA GPUs returned by the tool are reported; its GPU indices can differ
from CUDA visible ordinals under `CUDA_VISIBLE_DEVICES`.

Each query has a five-second timeout, 16KiB output cap and 64-GPU row cap.
Unsupported values are recorded as absent. Invalid/error/timeout results warn
and are retried at the next interval. Missing `nvidia-smi` disables this
optional monitor without stopping mining. Shutdown cancels any active query;
its process is killed on drop. Telemetry is diagnostic, not a safety interlock
or proof of GPU correctness, speed, accepted work, or successful payout.

Cloud validation uses offline mocks only. A Windows GPU run and verified live
pool failover remain user-run validation, never inferred from cloud tests.

Primary reference: [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#selective-query-options)
explains selective `--query-gpu` CSV output and unsupported `N/A` values.
NVIDIA warns that command output can change across driver releases; this
parser rejects unexpected output rather than guessing values.

The package preserves the executable's original filename and records the
source repository revision/dirty status and packaging host OS in the manifest.
These describe the packaging checkout/host; they do not attest that the supplied
binary was built from that revision or for that OS. Supply a binary built for
the destination OS/architecture: a Linux binary does not become a Windows binary
by renaming it `.exe`. Windows builds normally use `quantus-miner.exe`; Linux
builds use `quantus-miner` and may require executable file permissions after
transfer. The launcher discovers either name in the kit root, or a repository
`target/release` binary; `-ExecutablePath` explicitly selects another build.
CUDA device lists reject duplicate ordinals and more than 64 devices. CPU mode
rejects an explicit device selection other than the default `0`.


### Approximate selected-GPU efficiency

For CUDA sessions only, the monitor compares completed physical hash counts
between successful telemetry samples. The first sample establishes a baseline;
a later positive hash-count interval can produce aggregate physical MH/s and
approximate physical MH/s/W. Current watts are summed **only** for the selected
CUDA GPUs whose canonical UUIDs match exactly one telemetry row apiece. Other
GPUs' power is never included. Missing/ambiguous UUID mapping, duplicate selected
UUIDs, MIG identity, unavailable/zero power, counter reset or no completed hashes
makes efficiency unavailable rather than producing a guessed ratio. CPU sessions
can show diagnostic NVIDIA fields but never report GPU mining efficiency.

The ratio is interval completed physical hash rate divided by current sampled
selected-GPU watts. It is approximate: sampling jitter and variable workload or
power can affect it. It is not accepted-share/payout efficiency, whole-system
wall power or integrated energy consumption. SM and memory clocks, driver version
and the configured power limit help interpret changes; all are read-only fields.
No clock/power setting is applied, and there is no temperature/power safety
interlock. Keep the same hardware/driver/settings when comparing runs.
