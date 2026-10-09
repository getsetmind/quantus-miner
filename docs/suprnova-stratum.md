# Experimental Suprnova Stratum client

This fork adds a separate `stratum` command. Existing `serve` (native QUIC)
and `benchmark` commands remain available. The first target is one worker and
one visible NVIDIA GPU, with verified TLS and bounded sessions.

## What is verified

A user-run, non-mining TLS probe received a successful login and an initial
`qpow-poseidon2` job from `quantus.suprnova.cc:7074`. The job contained a
32-byte mining hash, a 4-byte extranonce, a 64-byte big-endian target and an
integer difficulty. The captured target equals `U512::MAX / difficulty`.

A subsequent user-run 120-second Windows RTX 3060 trial received three
accepted-share acknowledgements, with no rejected, unacknowledged or stale
shares, and a screenshot showed one worker reflected in the pool. It stopped
cleanly at the requested duration. The run also had three keepalive-triggered
reconnections, so stable ongoing-session behavior is not yet established.

### Validation snapshot (2026-10-09)

On the cloud Linux host, Rust 1.93.0:

- Workspace build and tests with `--locked`: passed
- Workspace Clippy, all targets/features, warnings denied: passed
- Rustfmt, Taplo checks and `git diff --check`: passed
- Linux `miner-cli` release build and `stratum --help`: passed
- 17 new offline Stratum tests: passed, including an in-memory mock pool
  receiving two CPU-verified shares and returning two acceptance responses
- The mock integration test passed 50 consecutive runs after acknowledgement
  window backpressure was added
- CUDA-dependent tests skip on this host because no NVIDIA CUDA device is
  available. Cloud GPU execution remains unavailable; the later user-run Windows
  results below provide the hardware and pool observations

### User-run Windows RTX 3060 results (2026-10-09)

- All 17 engine-cuda tests passed on the GPU-equipped host in 10.02 seconds
- Local benchmark: 142.84 MH/s
- Bounded 120-second Suprnova session: 12.52 billion physical hashes,
  3 accepted, 0 rejected, 0 unacknowledged, 0 stale, and 3 reconnects
- Clean timed stop; pool screenshot reflected one worker
- The pool's 535.71 MH/s estimate is a short-window accounting estimate,
  separate from the measured local benchmark

Each reconnect followed a keepalive rejection around the 30-second interval.
The actual reply payload was not captured. The pinned [NOMP Quantus wire
reference](https://pkg.go.dev/github.com/mining-pool/not-only-mining-pool@v0.0.0-20260912023102-dbfed907ce52/engine/quantus)
documents `KEEPALIVED` for keepalive success, while this client previously
required the share/login success status `OK`. The dedicated keepalive parser
now accepts object status `KEEPALIVED` or `OK` only with an absent/null error;
share success parsing remains unchanged and does not accept `KEEPALIVED`.
This is a reference-supported compatibility fix, not proof of the uncaptured
live response. A bounded retry is needed to verify that the reconnects stop.
The keepalive patch passes all 19 focused offline Stratum tests and focused
Clippy with warnings denied; malformed/error responses remain rejected.

## Build and offline validation

Use the toolchain pinned in `rust-toolchain` (currently Rust 1.93.0):

```sh
cargo fmt --all -- --check
cargo test -p stratum-service -p engine-cpu -p pow-core --locked
cargo test -p engine-cuda --locked -- --nocapture
cargo build -p miner-cli --release --locked
```

The CUDA crate loads the NVIDIA driver and NVRTC dynamically. CUDA tests may
skip when these are unavailable. A successful test harness with skipped GPU
tests is **not a GPU correctness pass**. Run the CUDA tests on the intended
GPU and retain their output before treating GPU changes as verified.

### NVRTC compilation without a GPU

`scripts/check_nvrtc.py` compiles the embedded kernel for `compute_86` using
an explicitly selected NVRTC library, without executing any GPU code:

```sh
python scripts/check_nvrtc.py --library /path/to/libnvrtc.so.12
```

On Windows, supply the actual NVRTC DLL path instead. Keep its matching
builtins library alongside it. Compilation success is not GPU correctness or
performance validation.

A Windows RTX 3060 run rejected the upstream `=&r` asm constraints. The fix
uses block-local PTX registers and only publishes output operands after all
inputs have been consumed, rather than simply removing early-clobber markers.
On Linux both the original and corrected source compiled with NVRTC 12.4.127
and 13.4.92; the reported failure was not reproduced on Linux. The subsequent Windows retry and engine-cuda tests passed. The runtime now logs NVRTC's
major/minor version to help distinguish actual compiler configurations.

## Bounded Windows GPU trial

Only run this after choosing to start real mining. This command sends shares
to the pool; it is different from the earlier login-only PowerShell probe.
Use the public payout address from your wallet, never a recovery phrase,
private key, or wormhole inner hash. No system clock, voltage, firewall or
power settings are changed by the command.

```powershell
.\target\release\quantus-miner.exe stratum --wallet YOUR_PUBLIC_QZ_ADDRESS --worker rig1 --engine cuda --duration 120
```

The default endpoint is `quantus.suprnova.cc:7074`, using system-independent
public root certificates and normal hostname validation. There is no
certificate-bypass flag and no plaintext fallback. A TLS failure is a failure
to investigate, not a reason to turn off verification.

If more than one CUDA GPU is visible, select one in the invoking process
with `CUDA_VISIBLE_DEVICES` before launching. Multi-GPU coordination is outside
this first implementation's scope. `--engine cpu` selects a single CPU worker
for troubleshooting, not an automatic fallback when CUDA fails.

The duration defaults to 120 seconds and is capped at 24 hours per invocation.
Ctrl+C also requests shutdown. A GPU/driver call that is already blocked can
delay shutdown; this implementation does not claim to recover a hung driver.

## Correctness and operational limits

- Reject malformed, oversized or inconsistent job data before hashing
- Keep every nonce inside the pool-assigned prefix
- Partition the remaining nonce space by a random per-run salt and checked
  work generation, so reconnects and target changes do not restart the same
  nonce scan. This does not promise zero collision probability across runs
- Recompute proposed shares with the exact CPU hash before submission
- Resume after a found candidate; do not stop the job after its first share
- Replay overflowed CUDA candidate batches at a smaller size before selecting
  a candidate, so the fixed result buffer does not silently discard earlier hits
- Cancel obsolete work on a changed job or a disconnected session
- Do not blindly resend unacknowledged submissions after reconnecting
- Separate attempted/submitted/accepted/rejected/stale/unacknowledged counts
- Bound outstanding requests and pause new submissions while waiting for
  acknowledgements instead of flooding the connection

The upstream CUDA kernel documents rare arithmetic false negatives. Its math
is unchanged here. CPU verification prevents those candidates from becoming
invalid submitted shares, but does not recover valid shares the kernel failed
to detect. Raw H/s and accepted work are therefore separate measurements.

The reference protocol implementations use the same general login/job/submit
dialect, but do not prove every Suprnova extension. Unsupported difficulty
updates must stop/cancel work rather than continue with a stale target.

## First live-test acceptance criteria

1. TLS and login succeed, then the worker receives a valid job
2. The pool explicitly acknowledges a submitted share as accepted
3. That work is visible in the pool's own worker accounting
4. No repeated invalid shares, duplicate submissions, uncaught panic or
   unexpected continuing mining after the duration/stop request

Pool difficulty and luck can prevent a short session from finding a share.
Zero accepted shares in a short run is inconclusive, not proof of a bug or a
successful integration. Do not automatically extend a trial indefinitely.

No speedup is claimed. Start with stock clocks, record the exact source
revision, driver/NVRTC versions, GPU temperature, actual power, duration,
accepted/rejected work and pool difficulty. Compare alternatives on the same
machine and conditions before making optimization claims.
