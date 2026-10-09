#![forbid(unsafe_code)]
//! Experimental verified-TLS Quantus Stratum client. Real accepted-share
//! compatibility is unverified. Single correctness-first CPU/CUDA worker.
pub mod protocol;
use anyhow::{bail, ensure, Context, Result};
use engine_cpu::{Candidate, EngineStatus, JobIdCancelCheck, MinerEngine, Range};
use primitive_types::U512;
use protocol::{FrameDecoder, Job};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex,
    },
    thread,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    sync::{mpsc, watch},
    time::{timeout, Instant},
};
use tokio_rustls::{
    rustls::{pki_types::ServerName, ClientConfig, RootCertStore},
    TlsConnector,
};

/// No Debug implementation: credentials are never logged.
#[derive(Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub login: String,
    pub password: String,
    pub agent: String,
    pub reconnect_attempts: u32,
    pub reconnect_delay: Duration,
    pub connect_timeout: Duration,
    pub response_timeout: Duration,
    pub range_size: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            host: "quantus.suprnova.cc".into(),
            port: 7074,
            login: String::new(),
            password: "x".into(),
            agent: "quantus-miner-stratum/0.1".into(),
            reconnect_attempts: 5,
            reconnect_delay: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(15),
            response_timeout: Duration::from_secs(60),
            range_size: 1_048_576,
        }
    }
}
impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.host.is_empty() && self.port > 0,
            "missing Stratum host/port"
        );
        ensure!(
            !self.login.is_empty() && self.login.len() <= 256,
            "invalid Stratum login length"
        );
        ensure!(
            self.password.len() <= 256 && self.agent.len() <= 256,
            "Stratum credential/agent too long"
        );
        ensure!(self.range_size > 0, "range size must be positive");
        ensure!(
            !self.connect_timeout.is_zero() && !self.response_timeout.is_zero(),
            "timeouts must be positive"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub hashes: u64,
    pub attempted: u64,
    pub submitted: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub unacknowledged: u64,
    pub stale: u64,
    pub reconnects: u32,
}
#[derive(Clone)]
struct Assigned {
    generation: u64,
    job: Job,
}
struct WorkerState {
    salt: [u8; 16],
    hashes: AtomicU64,
    epoch: AtomicU64,
    latest: Mutex<Option<Assigned>>,
    wake: Condvar,
}
enum WorkerEvent {
    Share(Box<(Assigned, Candidate)>),
    Failed(&'static str),
}
struct Worker {
    state: Arc<WorkerState>,
    handle: Option<thread::JoinHandle<()>>,
}
impl Worker {
    fn new(engine: Arc<dyn MinerEngine>, range_size: u64) -> (Self, mpsc::Receiver<WorkerEvent>) {
        Self::with_salt(engine, range_size, rand::random())
    }
    fn with_salt(
        engine: Arc<dyn MinerEngine>,
        range_size: u64,
        salt: [u8; 16],
    ) -> (Self, mpsc::Receiver<WorkerEvent>) {
        let state = Arc::new(WorkerState {
            salt,
            hashes: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            latest: Mutex::new(None),
            wake: Condvar::new(),
        });
        let (tx, rx) = mpsc::channel(16);
        let shared = state.clone();
        let handle = thread::spawn(move || worker_loop(shared, engine, range_size, tx));
        (
            Self {
                state,
                handle: Some(handle),
            },
            rx,
        )
    }
    fn cancel(&self) {
        let mut latest = self.state.latest.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .state
            .epoch
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_add(1))
            });
        *latest = None;
        self.state.wake.notify_one();
    }
    fn assign(&self, job: Job) -> Result<Assigned> {
        let generation = self
            .state
            .epoch
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                n.checked_add(1).filter(|next| *next < u64::MAX)
            })
            .map_err(|_| anyhow::anyhow!("mining generation exhausted"))?
            + 1;
        let assigned = Assigned { generation, job };
        *self.state.latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(assigned.clone());
        self.state.wake.notify_one();
        Ok(assigned)
    }
    fn stop(&self) {
        let _lock = self.state.latest.lock().unwrap_or_else(|e| e.into_inner());
        self.state.epoch.store(u64::MAX, Ordering::SeqCst);
        self.state.wake.notify_one();
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}
fn send_event(
    tx: &mpsc::Sender<WorkerEvent>,
    state: &WorkerState,
    generation: u64,
    mut event: WorkerEvent,
) -> bool {
    loop {
        if state.epoch.load(Ordering::SeqCst) != generation {
            return false;
        }
        match tx.try_send(event) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(e)) => {
                event = e;
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
fn worker_loop(
    state: Arc<WorkerState>,
    engine: Arc<dyn MinerEngine>,
    size: u64,
    tx: mpsc::Sender<WorkerEvent>,
) {
    loop {
        let assigned = {
            let mut lock = state.latest.lock().unwrap_or_else(|e| e.into_inner());
            while lock.is_none() && state.epoch.load(Ordering::SeqCst) != u64::MAX {
                lock = state.wake.wait(lock).unwrap_or_else(|e| e.into_inner());
            }
            if state.epoch.load(Ordering::SeqCst) == u64::MAX {
                return;
            }
            lock.take().expect("worker job present")
        };
        let cancel = JobIdCancelCheck {
            current_job_id: &state.epoch,
            my_job_id: assigned.generation,
        };
        let mut ctx = engine.prepare_context(assigned.job.header, assigned.job.difficulty);
        ctx.target = assigned.job.target;
        let (mut start, last) =
            protocol::nonce_partition(assigned.job.prefix, state.salt, assigned.generation);
        while state.epoch.load(Ordering::SeqCst) == assigned.generation {
            let end = start.saturating_add(U512::from(size - 1)).min(last);
            let result = engine.search_range(&ctx, Range { start, end }, &cancel);
            let hash_count = match &result {
                EngineStatus::Found { hash_count, .. }
                | EngineStatus::Running { hash_count }
                | EngineStatus::Exhausted { hash_count }
                | EngineStatus::Cancelled { hash_count }
                | EngineStatus::DeviceLost { hash_count } => *hash_count,
            };
            let _ = state
                .hashes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    Some(n.saturating_add(hash_count))
                });
            let next = match result {
                EngineStatus::Found { candidate, .. } => {
                    if candidate.nonce < start
                        || candidate.nonce > end
                        || !assigned.job.verify(&candidate)
                    {
                        let _ = send_event(
                            &tx,
                            &state,
                            assigned.generation,
                            WorkerEvent::Failed("engine returned invalid share"),
                        );
                        return;
                    }
                    let next = protocol::next_nonce(candidate.nonce, last);
                    if !send_event(
                        &tx,
                        &state,
                        assigned.generation,
                        WorkerEvent::Share(Box::new((assigned.clone(), candidate))),
                    ) {
                        break;
                    }
                    next
                }
                EngineStatus::Exhausted { .. } => protocol::next_nonce(end, last),
                EngineStatus::Cancelled { .. } => break,
                EngineStatus::DeviceLost { .. } | EngineStatus::Running { .. } => {
                    let _ = send_event(
                        &tx,
                        &state,
                        assigned.generation,
                        WorkerEvent::Failed("mining engine stopped unexpectedly"),
                    );
                    return;
                }
            };
            match next {
                Some(n) => start = n,
                None => break,
            }
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum RequestKind {
    Login,
    Submit,
    Keepalive,
}
struct Pending {
    kind: RequestKind,
    sent: Instant,
}
struct Session {
    token: Option<String>,
    active: Option<Assigned>,
    keepalive: bool,
    pending: BTreeMap<u64, Pending>,
}
impl Session {
    fn new() -> Self {
        Self {
            token: None,
            active: None,
            keepalive: false,
            pending: BTreeMap::new(),
        }
    }
    fn register(&mut self, id: u64, kind: RequestKind) -> Result<()> {
        ensure!(self.pending.len() < 64, "too many unacknowledged requests");
        ensure!(!self.pending.contains_key(&id), "duplicate request ID");
        ensure!(
            self.pending
                .insert(
                    id,
                    Pending {
                        kind,
                        sent: Instant::now()
                    }
                )
                .is_none(),
            "duplicate request ID"
        );
        Ok(())
    }
    fn handle(&mut self, v: Value, worker: &Worker, stats: &mut Stats) -> Result<()> {
        if let Some(id) = v.get("id").and_then(Value::as_u64) {
            let pending = self
                .pending
                .remove(&id)
                .context("unexpected Stratum response ID")?;
            match pending.kind {
                RequestKind::Login => {
                    ensure!(protocol::response_ok(&v), "Stratum login rejected");
                    let result = v.get("result").context("missing login result")?;
                    let token = result
                        .get("id")
                        .and_then(Value::as_str)
                        .context("missing session ID")?;
                    ensure!(
                        !token.is_empty() && token.len() <= 256,
                        "invalid session ID length"
                    );
                    let job = Job::parse(result.get("job").context("missing login job")?)?;
                    self.keepalive = result
                        .get("extensions")
                        .and_then(Value::as_array)
                        .is_some_and(|a| a.iter().any(|s| s.as_str() == Some("keepalive")));
                    self.token = Some(token.into());
                    self.active = Some(worker.assign(job)?);
                    log::info!("Stratum authenticated; mining job received");
                }
                RequestKind::Submit => {
                    if protocol::response_ok(&v) {
                        stats.accepted += 1;
                        log::info!("Stratum share accepted ({})", stats.accepted);
                    } else {
                        stats.rejected += 1;
                        log::warn!("Stratum share rejected ({})", stats.rejected);
                    }
                }
                RequestKind::Keepalive => {
                    ensure!(protocol::response_ok(&v), "Stratum keepalive rejected")
                }
            }
            return Ok(());
        }
        let method = v
            .get("method")
            .and_then(Value::as_str)
            .context("missing Stratum method/response ID")?;
        if method == "job" {
            ensure!(self.token.is_some(), "job before authentication");
            let params = v.get("params").context("missing job params")?;
            let job = Job::parse(params.get("job").unwrap_or(params))?;
            if let Some(old) = &mut self.active {
                if let (Some(a), Some(b)) = (old.job.sequence, job.sequence) {
                    ensure!(b >= a, "out-of-order job sequence");
                }
                if old.job.same_work(&job) {
                    old.job.sequence = job.sequence.or(old.job.sequence);
                    return Ok(());
                }
            }
            self.active = Some(worker.assign(job)?);
            log::info!("Stratum job updated");
        } else {
            let normalized = method.to_ascii_lowercase();
            ensure!(
                !normalized.contains("diff")
                    && !normalized.contains("target")
                    && !normalized.contains("extranonce")
                    && !has_work_update(v.get("params")),
                "unsupported target/difficulty update; reconnecting safely"
            );
            log::debug!("Ignoring unrelated Stratum notification");
        }
        Ok(())
    }
    fn disconnect(&mut self, worker: &Worker, stats: &mut Stats) {
        worker.cancel();
        stats.unacknowledged += self
            .pending
            .values()
            .filter(|p| p.kind == RequestKind::Submit)
            .count() as u64;
        self.pending.clear();
        self.token = None;
        self.active = None;
    }
}
fn has_work_update(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Object(map)) => map.iter().any(|(key, value)| {
            matches!(
                key.as_str(),
                "difficulty" | "target" | "extranonce" | "mining_hash" | "job"
            ) || has_work_update(Some(value))
        }),
        Some(Value::Array(values)) => values.iter().any(|v| has_work_update(Some(v))),
        _ => false,
    }
}
fn allocate_id(next: &mut u64) -> Result<u64> {
    let id = *next;
    *next = next.checked_add(1).context("request ID exhausted")?;
    Ok(id)
}
async fn write_request<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    value: Value,
    limit: Duration,
    shutdown: &mut watch::Receiver<bool>,
) -> Result<bool> {
    let mut bytes = serde_json::to_vec(&value)?;
    ensure!(
        bytes.len() <= protocol::MAX_FRAME_BYTES,
        "outgoing frame too large"
    );
    bytes.push(b'\n');
    tokio::select! {
        _ = shutdown_signal(shutdown) => return Ok(false),
        result = timeout(limit, writer.write_all(&bytes)) => { result.context("Stratum write timed out")??; }
    }
    Ok(true)
}
async fn shutdown_signal(shutdown: &mut watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}
async fn connection(
    config: &Config,
    worker: &Worker,
    shares: &mut mpsc::Receiver<WorkerEvent>,
    shutdown: &mut watch::Receiver<bool>,
    stats: &mut Stats,
    next_id: &mut u64,
) -> Result<()> {
    let roots = RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let provider = Arc::new(tokio_rustls::rustls::crypto::ring::default_provider());
    let tls = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = ServerName::try_from(config.host.clone()).context("invalid TLS hostname")?;
    let stream = tokio::select! {
        _ = shutdown_signal(shutdown) => return Ok(()),
        result = timeout(config.connect_timeout, async { let tcp = TcpStream::connect((config.host.as_str(), config.port)).await?; TlsConnector::from(Arc::new(tls)).connect(name, tcp).await }) => result.context("Stratum connection timed out")??,
    };
    session_loop(config, worker, shares, shutdown, stats, next_id, stream).await
}

/// Shared production/mock JSON-line lifecycle; TLS is established exclusively
/// by connection() before this helper is called in production.
async fn session_loop<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    config: &Config,
    worker: &Worker,
    shares: &mut mpsc::Receiver<WorkerEvent>,
    shutdown: &mut watch::Receiver<bool>,
    stats: &mut Stats,
    next_id: &mut u64,
    stream: S,
) -> Result<()> {
    let (read, mut write) = tokio::io::split(stream);
    let mut read = BufReader::new(read);
    let mut decoder = FrameDecoder::default();
    let mut session = Session::new();
    let id = allocate_id(next_id)?;
    session.register(id, RequestKind::Login)?;
    let result = async {
        if !write_request(&mut write, protocol::login_request(id, &config.login, &config.password, &config.agent), config.response_timeout, shutdown).await? { return Ok(()); }
        let mut timer = tokio::time::interval(Duration::from_secs(1));
        let mut last_keepalive = Instant::now();
        let mut last_progress = Instant::now();
        let mut last_hashes = worker.state.hashes.load(Ordering::Relaxed);
        loop {
            tokio::select! {
                _ = shutdown_signal(shutdown) => return Ok(()),
                byte = read.read_u8() => {
                    let byte = byte.context("Stratum disconnected")?;
                    if let Some(frame) = decoder.push(byte)? { session.handle(frame, worker, stats)?; }
                }
                // Backpressure at the acknowledgement window: keep reading
                // replies instead of treating a fast worker as a disconnect.
                event = shares.recv(), if session.pending.len() < 64 => {
                    match event.context("mining worker disconnected")? {
                        WorkerEvent::Failed(message) => bail!(message),
                        WorkerEvent::Share(share) => {
                            let (assigned, candidate) = *share;
                            if session.active.as_ref().map(|a| a.generation) != Some(assigned.generation) { stats.stale += 1; continue; }
                            stats.attempted += 1;
                            let id = allocate_id(next_id)?;
                            let value = protocol::submit_request(id, session.token.as_deref().context("share before login")?, &assigned.job, &candidate)?;
                            session.register(id, RequestKind::Submit)?;
                            if !write_request(&mut write, value, config.response_timeout, shutdown).await? { return Ok(()); } stats.submitted += 1;
                        }
                    }
                }
                _ = timer.tick() => {
                    if last_progress.elapsed() >= Duration::from_secs(10) {
                        let hashes = worker.state.hashes.load(Ordering::Relaxed);
                        let rate = hashes.saturating_sub(last_hashes) as f64 / last_progress.elapsed().as_secs_f64();
                        log::info!("Stratum progress: {rate:.0} physical H/s; {} hashes, {} accepted, {} rejected", hashes, stats.accepted, stats.rejected);
                        last_hashes = hashes; last_progress = Instant::now();
                    }
                    ensure!(!session.pending.values().any(|p| p.sent.elapsed() >= config.response_timeout), "Stratum acknowledgement timed out");
                    if session.keepalive && session.pending.len() < 64 && last_keepalive.elapsed() >= Duration::from_secs(30) && !session.pending.values().any(|p| p.kind == RequestKind::Keepalive) {
                        let id = allocate_id(next_id)?; session.register(id, RequestKind::Keepalive)?;
                        if !write_request(&mut write, json!({"id":id,"method":"keepalived","params":{"id":session.token}}), config.response_timeout, shutdown).await? { return Ok(()); }
                        last_keepalive = Instant::now();
                    }
                }
            }
        }
    }.await;
    session.disconnect(worker, stats);
    result
}
/// Finite reconnection budget. Shares are never replayed across login sessions.
pub async fn run(
    config: Config,
    engine: Arc<dyn MinerEngine>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<Stats> {
    config.validate()?;
    let (mut worker, mut shares) = Worker::new(engine, config.range_size);
    let mut stats = Stats::default();
    let mut next_id = 1;
    let mut result = loop {
        if *shutdown.borrow() {
            break Ok(stats);
        }
        match connection(
            &config,
            &worker,
            &mut shares,
            &mut shutdown,
            &mut stats,
            &mut next_id,
        )
        .await
        {
            Ok(()) => break Ok(stats),
            Err(e) => {
                worker.cancel();
                log::warn!("Stratum connection stopped: {e}");
                if stats.reconnects >= config.reconnect_attempts {
                    break Err(e.context("Stratum reconnect budget exhausted"));
                }
                stats.reconnects += 1;
                tokio::select! { _ = shutdown_signal(&mut shutdown) => break Ok(stats), _ = tokio::time::sleep(config.reconnect_delay) => {} }
            }
        }
    };
    worker.stop();
    if let Some(handle) = worker.handle.take() {
        tokio::task::spawn_blocking(move || handle.join())
            .await
            .context("worker join task failed")?
            .map_err(|_| anyhow::anyhow!("mining worker panicked"))?;
    }
    if let Ok(stats) = &mut result {
        stats.hashes = worker.state.hashes.load(Ordering::Relaxed);
    }
    result
}
#[cfg(test)]
mod tests;
