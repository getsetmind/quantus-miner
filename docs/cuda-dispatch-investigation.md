# CUDA dispatch investigation

Source investigation of baseline `5e268bd`, recorded 2026-10-09, followed by
the minimal defensive readback-completion fix described below. The kernel and
baseline binary were not changed. No GPU execution was available.

## Measurement context

The user's RTX 3060 sustained 134.667 MH/s with 4M batches during a 120-second
actual-mining run on driver 617.42, with no reported reconnects or errors.
The user explicitly closed background apps for the separate 32M-batch,
10-second benchmark that reached 146.52 MH/s; continued absence of competing
GPU load during the later pool run was not independently verified. These
different batch sizes and workloads do not establish a speedup or prove the
sustained 140 MH/s goal. Earlier batch sweeps did not control background GPU
load, so their rankings remain provisional.

For a full 4,000,000-nonce batch, 134.667 MH/s corresponds to approximately
29.70 ms, while 140 MH/s corresponds to 28.57 ms. Closing the entire gap through
host overhead alone would require saving approximately 1.13 ms per full batch.
This arithmetic is illustrative, not a measured decomposition of mining time.

## Synchronization evidence

- `crates/engine-cuda/src/lib.rs:516-544` starts the dispatch timer, clears the
  device results, launches the kernel, explicitly synchronizes at line 530,
  accumulates busy time at line 536, and then calls `clone_dtoh` at line 538
- `Cargo.lock:528-533` pins cudarc 0.19.9
- In cudarc 0.19.9, `src/driver/safe/core.rs:1634-1644`, `clone_dtoh` allocates
  a fresh Vec, calls `memcpy_dtoh`, and returns the Vec
- `core.rs:1648-1657` obtains device/host pointers and calls the asynchronous
  D2H helper. `src/driver/result.rs:1030-1052` implements that helper with
  `cuMemcpyDtoHAsync_v2` and documents that destination mutation can occur
  after the call returns
- Vec host-memory guards return `SyncOnDrop::Sync(None)` at
  `core.rs:1377-1381`; ordinary array guards do the same at lines 1341-1345.
  Their destruction does not introduce a host synchronization
- Device-pointer guards wait for previous writes and record device events
  (`core.rs:1170-1180`). Device ordering is not a host completion fence

NVIDIA's [driver API synchronization documentation](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html)
states that asynchronous transfers involving pageable host memory might be
host-synchronous. It does not guarantee that they are. The
[D2H API documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
likewise describes `cuMemcpyDtoHAsync` as asynchronous for most uses.

Therefore, the installed cudarc implementation does not provide evidence of a
duplicate explicit host synchronization. The existing pre-copy fence establishes
kernel completion and reports asynchronous kernel failures, but it does not
formally establish completion of the subsequently submitted copy before the host
reads its result. This is a source-level completion-guarantee concern, unconfirmed
on the user's hardware. No real-world race or incorrect result was observed.

## Minimal correctness hardening

The implementation retains the existing pre-copy synchronization and busy-time
boundary. After a successful `clone_dtoh`, it explicitly synchronizes the same
stream before inspecting the returned Vec. A failed completion fence maps to
`DeviceLost`, with an explicit result-copy completion error message. Nonce
ownership is not advanced, the batch is not accounted as completed, and candidates
are not validated before successful completion.

This additional post-copy fence is a correctness hardening change, not
a performance claim. Removing the pre-copy synchronization alone is unsupported.

A later performance candidate could queue D2H before one explicit completion
fence and reuse a pinned nine-word host buffer. It must preserve error handling
and a meaningful duty-cycle metric. Recording busy time immediately after an
asynchronous launch would incorrectly measure submission time. Moving that
boundary after readback changes the current metric's meaning; preserving the
kernel-completion boundary would require suitable event timing or an explicitly
documented metric change. An ordinary stack array removes Vec allocation but
does not supply the missing completion guarantee.

## Next measurements and acceptance gates

1. Measure unchanged baseline per-batch preparation, kernel execution, host wait,
   D2H, and CPU validation separately on the same RTX 3060. Keep instrumentation
   separate from final release-rate comparisons
2. Correctness-gate any candidate on all five fixed vectors and target boundaries
   (`hash - 1`, `hash`, `hash + 1`), existing overflow/resume and cancellation
   tests, candidate accounting, lowest-valid-nonce selection, prefix equality,
   zero-hit/U512 maximum, and nonce carries. Repeated hit/no-hit batches should
   check for stale result counts. A skipped GPU test is not a correctness pass
3. Compare unchanged baseline and candidate with identical driver, app state,
   batch size, clocks/power, and workload. Warm up and collect at least three
   alternating sustained samples; record raw rates, binary/kernel identity,
   temperature, clocks, and GPU identity. Confirm actual mining, not only a short
   benchmark. Keep an optimization only after repeatable improvement

`pow-core/src/lib.rs:38-69` also shows that prestate excludes the low 64 nonce
bits, while `crates/engine-cuda/src/lib.rs:505-508` recomputes it per batch. A
search-local cache must refresh at upper-nonce changes and preserve carry
splitting. Its benefit is limited when a Stratum search range contains only one
batch, so prioritize it only if measurements show meaningful repeated work.

Hash math, candidate validation, overflow replay, nonce ownership, and the
existing kernel's arithmetic contract remain outside the scope of these host
dispatch proposals.

## Offline validation

A GPU-gated regression reuses the same buffers for 16 alternating dense-hit and
zero-hit rounds. It checks CPU-verified lowest nonce, candidate work/hash, and
physical batch hash count. It requires actual CUDA execution to validate readback;
offline tests cannot prove driver synchronization behavior.

With the repository-pinned Rust 1.93.0, release-profile `engine-cuda` tests report
20 harness successes: nine offline tests executed, while 11 CUDA-gated tests
returned early because the CUDA driver was unavailable, including the new
regression. These skips are not GPU correctness passes.

Repository-wide validation subsequently passed with the same pinned compiler,
locked offline dependencies, a separate operational target directory, and
incremental compilation disabled. Release-profile checks reused the existing
operational cache to limit disk use:

- `cargo +1.93.0 build --workspace --release --locked --offline`
- `cargo +1.93.0 test --workspace --release --locked --offline -- --nocapture`
- `cargo +1.93.0 clippy --workspace --all-targets --all-features --release --locked --offline -- -D warnings`
- `cargo +1.93.0 fmt --all -- --check`
- `taplo fmt --check`
- `git diff --check`

Each command completed with exit status zero. The workspace test harness reports
125 successes across 24 unit/doc-test result groups, zero failures, and zero
formally ignored tests. Of those successes, 11 CUDA tests returned early with
explicit driver-unavailable skip messages. The remaining 114 harness successes
do not substitute for actual GPU correctness or synchronization validation. No
runtime mining was performed and no clock settings were changed.

The existing baseline release binary's SHA256 remained
`83379d8e7219b161f3709dcf14d9054860959f8f6fb2071be3641a37abd062cb`.
