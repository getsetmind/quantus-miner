# Quantus Stratum protocol evidence and conservative updates

Checked 2026-10-09. These references identify a shared dialect, not every
extension deployed by Suprnova. The synthetic offline tests are not pool captures.

## Pinned primary references

- [NOMP Quantus package and wire documentation, version v0.0.0-20260912023102-dbfed907ce52](https://pkg.go.dev/github.com/mining-pool/not-only-mining-pool@v0.0.0-20260912023102-dbfed907ce52/engine/quantus)
- [NOMP dialect implementation, commit dbfed907ce52483bf34b2aca2fb1e78e77e39060](https://github.com/mining-pool/not-only-mining-pool/blob/dbfed907ce52483bf34b2aca2fb1e78e77e39060/engine/quantus/stratum.go)
- [Official Quantus node mining guide, pinned by that package](https://github.com/Quantus-Network/chain/blob/f1176cea6a6d08ea437710dcd45cae6717b773df/MINING.md)
- [Official canonical miner wire types, linked by that package](https://github.com/Quantus-Network/chain/blob/f1176cea6a6d08ea437710dcd45cae6717b773df/miner-api/src/lib.rs)
  The canonical-types link could not be fetched during this review; it is not
  evidence for any additional Stratum notification. The fetched official mining
  guide describes native QUIC; do not conflate that with this JSON dialect.

The pinned package documents newline JSON login/job/submit objects. Login returns
a session ID, status, keepalive extension, and initial job (possibly null while
waiting). Jobs carry algorithm, job ID, mining hash, extranonce, target, integer
difficulty, and sequence. Subsequent `job` notifications contain an object with
`clean_jobs: true` and a full `job`. Keepalive success is `KEEPALIVED`; share
success is `OK`. It documents fixed difficulty with varDiff disabled. It does
not document standalone difficulty, target, or extranonce update messages.

The dialect source emits `job`, JSON-RPC version 2.0, object params, and the full
job wrapper. It derives target from integer difficulty and preserves each
session's prefix. Its explicit uint64 difficulty bound applies to its Stratum
listener. No additional supported Suprnova method is inferred from this source.

## Implemented client policy

- Accept fully validated `job` notifications. A changed target/difficulty in a
  full job is a work change even if the ID, hash and prefix remain identical
- Validate target equals `U512::MAX / difficulty`, with both positive
- Preserve the sequence high-water mark across updates that omit a sequence
- Reject decreasing sequences, even after a sequence-less work change
- Sequence-only and identical-work updates do not restart nonce scanning
- Reject every unknown method; an unfamiliar method can change work without
  any field or method name matching a heuristic blacklist
- Reject unknown envelope, job-wrapper and job fields and unsupported
  `clean_jobs` policies. Reconnect and cancel old work on parser failure

The existing direct-job params, omitted JSON-RPC version, optional sequence,
decimal-string difficulty, and narrowly bounded `notice` message are local
compatibility choices, not new upstream-supported wireformats. `notice` allows
only a nonempty string message of at most 4096 bytes; structured or extra fields
are rejected. No standalone difficulty conversion or partial-job merge is added.
A real new method needs a read-only captured example or authoritative protocol
source before support can safely be implemented.
