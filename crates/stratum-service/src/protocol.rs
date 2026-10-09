//! Quantus JSON-line protocol. A bounded user-run Suprnova trial received
//! accepted-share acknowledgements. No upstream implementation was copied.
use anyhow::{bail, ensure, Context, Result};
use engine_cpu::Candidate;
use pow_core::JobContext;
use primitive_types::U512;
use serde_json::{json, Value};

pub const MAX_FRAME_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub job_id: String,
    pub header: [u8; 32],
    pub prefix: [u8; 4],
    pub difficulty: U512,
    pub target: U512,
    pub sequence: Option<u64>,
}

fn fixed_hex<const N: usize>(value: &Value, field: &str) -> Result<[u8; N]> {
    let s = value
        .get(field)
        .and_then(Value::as_str)
        .context("missing hexadecimal field")?;
    ensure!(s.len() == N * 2, "incorrect hexadecimal field length");
    let mut out = [0; N];
    hex::decode_to_slice(s, &mut out).map_err(|_| anyhow::anyhow!("invalid hexadecimal field"))?;
    Ok(out)
}

impl Job {
    pub fn parse(value: &Value) -> Result<Self> {
        let fields = value.as_object().context("job must be an object")?;
        ensure!(
            fields.keys().all(|key| matches!(
                key.as_str(),
                "algo" | "job_id" | "mining_hash" | "extranonce" | "difficulty" | "target" | "seq"
            )),
            "unsupported job field"
        );
        ensure!(
            value.get("algo").and_then(Value::as_str) == Some("qpow-poseidon2"),
            "unsupported Stratum algorithm"
        );
        let id = value
            .get("job_id")
            .and_then(Value::as_str)
            .context("missing job ID")?;
        ensure!(!id.is_empty() && id.len() <= 128, "invalid job ID length");
        let difficulty = match value.get("difficulty") {
            Some(Value::String(s)) => {
                U512::from_dec_str(s).map_err(|_| anyhow::anyhow!("invalid difficulty"))?
            }
            Some(v) => U512::from(
                v.as_u64()
                    .context("difficulty must be a positive integer")?,
            ),
            None => bail!("missing difficulty"),
        };
        ensure!(!difficulty.is_zero(), "difficulty is zero");
        let target = U512::from_big_endian(&fixed_hex::<64>(value, "target")?);
        ensure!(
            !target.is_zero() && target == U512::MAX / difficulty,
            "target/difficulty mismatch"
        );
        let sequence = value
            .get("seq")
            .map(|v| v.as_u64().context("invalid job sequence"))
            .transpose()?;
        Ok(Self {
            job_id: id.into(),
            header: fixed_hex(value, "mining_hash")?,
            prefix: fixed_hex(value, "extranonce")?,
            difficulty,
            target,
            sequence,
        })
    }

    /// Sequence-only updates must not restart nonce scanning.
    pub fn same_work(&self, other: &Self) -> bool {
        self.job_id == other.job_id
            && self.header == other.header
            && self.prefix == other.prefix
            && self.difficulty == other.difficulty
            && self.target == other.target
    }

    /// Validate ordering and retain the sequence high-water mark when a
    /// compatible notification omits it, including across work changes.
    /// Return a prepared job; compare `same_work` before assigning a new
    /// worker generation so sequence-only updates preserve the nonce cursor.
    pub fn reconcile_update(&self, mut next: Self) -> Result<Self> {
        if let (Some(previous), Some(incoming)) = (self.sequence, next.sequence) {
            ensure!(incoming >= previous, "out-of-order job sequence");
        }
        next.sequence = next.sequence.or(self.sequence);
        Ok(next)
    }

    pub fn context(&self) -> JobContext {
        JobContext {
            header: self.header,
            difficulty: self.difficulty,
            target: self.target,
        }
    }

    /// The pool owns the high four bytes; never carry into another prefix.
    pub fn nonce_bounds(&self) -> (U512, U512) {
        let mut low = [0u8; 64];
        let mut high = [255u8; 64];
        low[..4].copy_from_slice(&self.prefix);
        high[..4].copy_from_slice(&self.prefix);
        (U512::from_big_endian(&low), U512::from_big_endian(&high))
    }

    pub fn verify(&self, candidate: &Candidate) -> bool {
        let (low, high) = self.nonce_bounds();
        candidate.nonce >= low
            && candidate.nonce <= high
            && candidate.work == candidate.nonce.to_big_endian()
            && candidate.hash < self.target
            && pow_core::hash_from_nonce(&self.context(), candidate.nonce) == candidate.hash
    }
}

/// Full validated work, or a narrowly bounded local informational extension.
/// No standalone difficulty/target method is documented by the pinned dialect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Notification {
    Job(Box<Job>),
    Notice,
}

/// Fail closed on every unknown notification, rather than guessing whether a
/// method name or recursively hidden field changes work. The documented job
/// wrapper and the client's existing direct-job compatibility are supported.
/// `notice` is local compatibility, not a claimed Suprnova protocol extension.
pub fn parse_notification(value: &Value) -> Result<Notification> {
    let envelope = value
        .as_object()
        .context("notification must be an object")?;
    ensure!(
        envelope
            .keys()
            .all(|key| matches!(key.as_str(), "jsonrpc" | "id" | "method" | "params")),
        "unsupported notification envelope field"
    );
    ensure!(
        value.get("id").is_none_or(Value::is_null),
        "notification has a response ID"
    );
    ensure!(
        value
            .get("jsonrpc")
            .is_none_or(|v| v.as_str() == Some("2.0")),
        "unsupported JSON-RPC notification version"
    );
    let method = value
        .get("method")
        .and_then(Value::as_str)
        .context("missing notification method")?;
    let params = value.get("params").context("missing notification params")?;
    let object = params
        .as_object()
        .context("notification params must be an object")?;
    match method {
        "job" => {
            let job = if let Some(job) = object.get("job") {
                ensure!(
                    object
                        .keys()
                        .all(|key| matches!(key.as_str(), "job" | "clean_jobs")),
                    "unsupported job notification field"
                );
                ensure!(
                    object
                        .get("clean_jobs")
                        .is_none_or(|v| v == &Value::Bool(true)),
                    "unsupported clean_jobs policy"
                );
                job
            } else {
                params
            };
            Ok(Notification::Job(Box::new(Job::parse(job)?)))
        }
        "notice" => {
            ensure!(object.len() == 1, "unsupported notice fields");
            let message = object
                .get("message")
                .and_then(Value::as_str)
                .context("invalid notice message")?;
            ensure!(
                !message.is_empty() && message.len() <= 4096,
                "invalid notice message length"
            );
            Ok(Notification::Notice)
        }
        _ => bail!("unsupported Stratum notification; reconnecting safely"),
    }
}

pub fn login_request(id: u64, login: &str, pass: &str, agent: &str) -> Value {
    json!({"id":id,"method":"login","params":{"login":login,"pass":pass,"agent":agent}})
}

pub fn submit_request(id: u64, session: &str, job: &Job, candidate: &Candidate) -> Result<Value> {
    ensure!(job.verify(candidate), "refusing invalid share");
    Ok(
        json!({"id":id,"method":"submit","params":{"id":session,"job_id":job.job_id,"nonce":hex::encode(candidate.work),"result":hex::encode(candidate.hash.to_big_endian())}}),
    )
}

pub fn response_ok(value: &Value) -> bool {
    value.get("error").is_none_or(Value::is_null)
        && (value.get("result") == Some(&Value::Bool(true))
            || value.pointer("/result/status").and_then(Value::as_str) == Some("OK"))
}

/// Keepalive replies have their own success status, which must never count as
/// share acceptance. The pinned NOMP Quantus wire reference documents
/// `KEEPALIVED`; retain object-status `OK` compatibility, while rejecting
/// missing/malformed results, boolean results, and every non-null error.
/// Reference: https://pkg.go.dev/github.com/mining-pool/not-only-mining-pool@v0.0.0-20260912023102-dbfed907ce52/engine/quantus
pub fn keepalive_response_ok(value: &Value) -> bool {
    value.get("error").is_none_or(Value::is_null)
        && matches!(
            value.pointer("/result/status").and_then(Value::as_str),
            Some("KEEPALIVED" | "OK")
        )
}

/// Increment without wrapping through a pool-controlled prefix.
pub fn next_nonce(nonce: U512, end: U512) -> Option<U512> {
    if nonce < end {
        Some(nonce + U512::one())
    } else {
        None
    }
}

/// Cancellation-safe bounded framing: incomplete bytes persist across select!
/// cancellation. Never use read_line with a growing, unbounded allocation.
#[derive(Default)]
pub struct FrameDecoder {
    bytes: Vec<u8>,
}
impl FrameDecoder {
    pub fn push(&mut self, byte: u8) -> Result<Option<Value>> {
        if byte == b'\n' {
            ensure!(!self.bytes.is_empty(), "empty Stratum frame");
            let result = serde_json::from_slice(&self.bytes)
                .map_err(|_| anyhow::anyhow!("invalid Stratum JSON frame"));
            self.bytes.clear();
            return result.map(Some);
        }
        ensure!(
            self.bytes.len() < MAX_FRAME_BYTES,
            "Stratum frame too large"
        );
        self.bytes.push(byte);
        Ok(None)
    }
}

/// Disjoint per-generation subranges under the pool's fixed prefix. The run
/// salt prevents predictable overlap between independent miner restarts.
pub fn nonce_partition(prefix: [u8; 4], salt: [u8; 16], generation: u64) -> (U512, U512) {
    let mut low = [0u8; 64];
    low[..4].copy_from_slice(&prefix);
    low[4..20].copy_from_slice(&salt);
    low[20..28].copy_from_slice(&generation.to_be_bytes());
    let mut high = low;
    high[28..].fill(255);
    (U512::from_big_endian(&low), U512::from_big_endian(&high))
}
