# Experimental Suprnova Stratum client

This fork adds a separate `stratum` command. Existing `serve` (native QUIC)
and `benchmark` commands remain available. The first target is one worker and
one visible NVIDIA GPU, with verified TLS and bounded sessions.

## What is verified

A user-run, non-mining TLS probe received a successful login and an initial
`qpow-poseidon2` job from `quantus.suprnova.cc:7074`. The job contained a
32-byte mining hash, a 4-byte extranonce, a 64-byte big-endian target and an
integer difficulty. The captured target equals `U512::MAX / difficulty`.

This is **not** evidence that this implementation has submitted an accepted
share. The submit format and ongoing notification handling remain provisional
until a bounded live test confirms both the server acknowledgement and the
pool's worker accounting. Offline tests cannot establish that compatibility.

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
  available. GPU validation, Windows compilation, and real pool acceptance
  remain outstanding

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
