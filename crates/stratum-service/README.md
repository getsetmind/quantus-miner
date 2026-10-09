# Experimental Quantus Stratum service

One TLS-verified worker serves Suprnova's observed Quantus `login`/`job`
protocol. The host and certificate name are always checked against WebPKI
roots; plaintext transport and certificate bypass are not supported.

`run(Config, Arc<dyn MinerEngine>, watch::Receiver<bool>)` connects only when
called. A true shutdown value or a closed shutdown channel stops the session.
The reconnect budget is finite and counts total reconnections, with a constant
configured delay. Request IDs increase across sessions. Unacknowledged shares
are counted and never retransmitted after a disconnect.

The bounded latest-job slot cancels old engine work by generation. A bounded
share queue applies cancellation-aware backpressure. The nonce's high four
bytes are the pool-assigned extranonce, while the remaining bytes use a local namespace: 16 random per-run salt bytes,
8 big-endian generation bytes, and a 36-byte scanned counter. Generation
allocation is checked and cannot wrap. Within one run the disjoint generation
ranges prevent reconnects or target changes from re-mining previously submitted
nonces; independent runs have negligible random-salt collision probability.
Neither generation nor counter can carry outside its assigned range. Each engine winner is independently verified
against `pow-core`, its original nonce bytes, and the strict hash < target
comparison before submission. The physical-hash atomic counter includes every engine-returned hash count,
including cancellation and CUDA replay work; progress is logged every ten seconds
without an event per batch. Search resumes at the returned nonce + 1 so a
job can yield many shares. The engine must return its earliest valid winner;
the CUDA caller's overflow replay/splitting contract is important here.

The observed difficulty/target relationship is checked exactly as
floor(MAX_U512 / integer difficulty). Unknown algorithm, malformed fields,
out-of-order job sequence, partial target/difficulty changes, frame overflow,
invalid request state and acknowledgement timeout end the session and cancel
mining. Unrelated notifications are ignored. Sequence-only job updates retain
the same nonce cursor. Keepalive is sent only when the login advertises it.
The 64-request acknowledgement window applies backpressure to both new shares
and keepalive requests; incoming acknowledgements can still drain the window.
Credentials, session IDs, wire JSON, and pool rejection messages are not logged.

## Unverified compatibility

The `submit` shape is provisional: params contain session `id`, `job_id`,
a 128-character big-endian `nonce`, and a 128-character big-endian `result`.
This matches the inspected public Quantus reference interfaces, but no real
share was submitted or accepted during implementation. No source code from
those references was copied. This implementation does not establish CUDA
hardware performance or remove the CUDA engine's documented approximate-field
arithmetic miss risk. No payout address appears in fixtures.

## Offline checks

`cargo test -p stratum-service --lib` covers the observed sanitized login,
invalid fields, strict target boundary, reference hash verification,
extranonce/carry, multiple shares, framing bounds, job updates, sequence
idempotence, request acknowledgements, reconnect session clearing, blocked
write shutdown, and idle-worker shutdown. Tests never connect to a pool.
