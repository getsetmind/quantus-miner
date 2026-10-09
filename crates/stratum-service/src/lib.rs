#![forbid(unsafe_code)]
//! Experimental verified-TLS Quantus Stratum client. A bounded live trial
//! received accepted shares; ongoing stability still needs validation.
//! Explicit correctness-first CPU/CUDA workers with one authenticated session.
pub mod protocol;
use anyhow::{ensure, Context, Result};
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
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}
impl Endpoint {
    pub fn parse(value: &str) -> Result<Self> {
        let (host, port) = value
            .rsplit_once(':')
            .context("TLS endpoint must be hostname:port")?;
        let endpoint = Self {
            host: host.into(),
            port: port.parse().context("invalid TLS endpoint port")?,
        };
        endpoint.validate()?;
        Ok(endpoint)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.host.is_empty()
                && self.host.len() <= 253
                && self
                    .host
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
                && self.port > 0,
            "TLS endpoint must be a hostname and positive port"
        );
        ServerName::try_from(self.host.clone()).context("invalid TLS hostname")?;
        Ok(())
    }
}

/// No Debug implementation: credentials are never logged.
#[derive(Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub fallback_pools: Vec<Endpoint>,
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
            fallback_pools: Vec::new(),
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
    fn endpoint(&self, index: usize) -> Endpoint {
        let index = index % (self.fallback_pools.len() + 1);
        if index == 0 {
            Endpoint {
                host: self.host.clone(),
                port: self.port,
            }
        } else {
            self.fallback_pools[index - 1].clone()
        }
    }
    fn validate(&self) -> Result<()> {
        Endpoint {
            host: self.host.clone(),
            port: self.port,
        }
        .validate()?;
        ensure!(
            self.fallback_pools.len() <= 8,
            "at most eight explicit TLS fallback endpoints"
        );
        for endpoint in &self.fallback_pools {
            endpoint.validate()?;
        }
        ensure!(self.reconnect_attempts <= 20, "reconnect budget exceeds 20");
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
    hashes: Arc<AtomicU64>,
    failure: Mutex<Option<&'static str>>,
    epoch: AtomicU64,
    latest: Mutex<Option<Assigned>>,
    wake: Condvar,
}
#[derive(Debug)]
struct WorkerFailure(&'static str);
impl std::fmt::Display for WorkerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for WorkerFailure {}

enum WorkerEvent {
    Share(Box<(Assigned, Candidate)>),
    Failed(&'static str),
}
struct Worker {
    state: Arc<WorkerState>,
    handle: Option<thread::JoinHandle<()>>,
    extra_handles: Vec<thread::JoinHandle<()>>,
}
impl Worker {
    #[cfg(test)]
    fn new(engine: Arc<dyn MinerEngine>, range_size: u64) -> (Self, mpsc::Receiver<WorkerEvent>) {
        Self::with_salt(engine, range_size, rand::random())
    }
    #[cfg(test)]
    fn with_salt(
        engine: Arc<dyn MinerEngine>,
        range_size: u64,
        salt: [u8; 16],
    ) -> (Self, mpsc::Receiver<WorkerEvent>) {
        Self::with_engines(vec![engine], range_size, salt)
    }
    #[cfg(test)]
    fn with_engines(
        engines: Vec<Arc<dyn MinerEngine>>,
        range_size: u64,
        salt: [u8; 16],
    ) -> (Self, mpsc::Receiver<WorkerEvent>) {
        Self::with_hash_counter(engines, range_size, salt, Arc::new(AtomicU64::new(0)))
    }
    fn with_hash_counter(
        engines: Vec<Arc<dyn MinerEngine>>,
        range_size: u64,
        salt: [u8; 16],
        hashes: Arc<AtomicU64>,
    ) -> (Self, mpsc::Receiver<WorkerEvent>) {
        let state = Arc::new(WorkerState {
            salt,
            hashes,
            failure: Mutex::new(None),
            epoch: AtomicU64::new(0),
            latest: Mutex::new(None),
            wake: Condvar::new(),
        });
        let (tx, rx) = mpsc::channel(16);
        let mut handles = engines
            .into_iter()
            .enumerate()
            .map(|(index, engine)| {
                let shared = state.clone();
                let sender = tx.clone();
                thread::spawn(move || worker_loop(shared, engine, range_size, index as u32, sender))
            })
            .collect::<Vec<_>>();
        let handle = handles.remove(0);
        (
            Self {
                state,
                handle: Some(handle),
                extra_handles: handles,
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
        self.state.wake.notify_all();
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
        self.state.wake.notify_all();
        Ok(assigned)
    }
    fn stop(&self) {
        let _lock = self.state.latest.lock().unwrap_or_else(|e| e.into_inner());
        self.state.epoch.store(u64::MAX, Ordering::SeqCst);
        self.state.wake.notify_all();
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
fn report_worker_failure(
    tx: &mpsc::Sender<WorkerEvent>,
    state: &WorkerState,
    _generation: u64,
    message: &'static str,
) {
    // Persist independently of the bounded share queue. Backpressure or a pool
    // reconnect must never erase a fatal device failure. Cancel every worker
    // immediately; an already in-flight driver call can still delay exit.
    *state.failure.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
    let mut latest = state.latest.lock().unwrap_or_else(|e| e.into_inner());
    let _ = state
        .epoch
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
            Some(n.saturating_add(1))
        });
    *latest = None;
    state.wake.notify_all();
    let _ = tx.try_send(WorkerEvent::Failed(message));
}
fn check_worker_failure(worker: &Worker) -> Result<()> {
    if let Some(message) = *worker
        .state
        .failure
        .lock()
        .unwrap_or_else(|e| e.into_inner())
    {
        return Err(WorkerFailure(message).into());
    }
    Ok(())
}
fn worker_loop(
    state: Arc<WorkerState>,
    engine: Arc<dyn MinerEngine>,
    size: u64,
    worker_index: u32,
    tx: mpsc::Sender<WorkerEvent>,
) {
    let mut previous_generation = 0;
    loop {
        let assigned = {
            let mut lock = state.latest.lock().unwrap_or_else(|e| e.into_inner());
            while lock
                .as_ref()
                .is_none_or(|a| a.generation == previous_generation)
                && state.epoch.load(Ordering::SeqCst) != u64::MAX
            {
                lock = state.wake.wait(lock).unwrap_or_else(|e| e.into_inner());
            }
            if state.epoch.load(Ordering::SeqCst) == u64::MAX {
                return;
            }
            lock.as_ref().expect("worker job present").clone()
        };
        previous_generation = assigned.generation;
        let cancel = JobIdCancelCheck {
            current_job_id: &state.epoch,
            my_job_id: assigned.generation,
        };
        let mut ctx = engine.prepare_context(assigned.job.header, assigned.job.difficulty);
        ctx.target = assigned.job.target;
        let (mut start, last) = worker_nonce_partition(
            assigned.job.prefix,
            state.salt,
            assigned.generation,
            worker_index,
        );
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
                        report_worker_failure(
                            &tx,
                            &state,
                            assigned.generation,
                            "engine returned invalid share",
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
                    report_worker_failure(
                        &tx,
                        &state,
                        assigned.generation,
                        "mining engine stopped unexpectedly",
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
/// Fixed worker namespace inside each run/generation namespace. Even when
/// another device fails or reconnects, no worker is reassigned another range.
fn worker_nonce_partition(
    prefix: [u8; 4],
    salt: [u8; 16],
    generation: u64,
    worker: u32,
) -> (U512, U512) {
    let (start, _) = protocol::nonce_partition(prefix, salt, generation);
    let mut low = start.to_big_endian();
    low[28..32].copy_from_slice(&worker.to_be_bytes());
    let mut high = low;
    high[32..].fill(255);
    (U512::from_big_endian(&low), U512::from_big_endian(&high))
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
                    let initial_job = result.get("job").context("missing login job")?;
                    let job = if initial_job.is_null() {
                        None
                    } else {
                        Some(Job::parse(initial_job)?)
                    };
                    self.keepalive = result
                        .get("extensions")
                        .and_then(Value::as_array)
                        .is_some_and(|a| a.iter().any(|s| s.as_str() == Some("keepalive")));
                    self.token = Some(token.into());
                    self.active = job.map(|job| worker.assign(job)).transpose()?;
                    log::info!(
                        "Stratum authenticated; {}",
                        if self.active.is_some() {
                            "mining job received"
                        } else {
                            "waiting for pool job"
                        }
                    );
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
                    ensure!(
                        protocol::keepalive_response_ok(&v),
                        "Stratum keepalive rejected"
                    )
                }
            }
            return Ok(());
        }
        match protocol::parse_notification(&v)? {
            protocol::Notification::Job(job) => {
                let mut job = *job;
                ensure!(self.token.is_some(), "job before authentication");
                if let Some(old) = &mut self.active {
                    job = old.job.reconcile_update(job)?;
                    if old.job.same_work(&job) {
                        old.job.sequence = job.sequence;
                        return Ok(());
                    }
                }
                self.active = Some(worker.assign(job)?);
                log::info!("Stratum job/target updated");
            }
            protocol::Notification::Notice => log::debug!("Received bounded compatibility notice"),
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
            check_worker_failure(worker)?;
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
                        WorkerEvent::Failed(message) => return Err(WorkerFailure(message).into()),
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
                        log::info!("Stratum progress: {rate:.0} physical H/s; {} hashes, {} attempted, {} submitted, {} accepted, {} rejected, {} stale, {} unacknowledged, {} reconnects", hashes, stats.attempted, stats.submitted, stats.accepted, stats.rejected, stats.stale, stats.unacknowledged, stats.reconnects);
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
    shutdown: watch::Receiver<bool>,
) -> Result<Stats> {
    run_multi(config, vec![engine], shutdown).await
}

/// Explicit engine per worker; all workers share one authenticated pool session.
pub async fn run_multi(
    config: Config,
    engines: Vec<Arc<dyn MinerEngine>>,
    shutdown: watch::Receiver<bool>,
) -> Result<Stats> {
    run_multi_observed(config, engines, shutdown, Arc::new(AtomicU64::new(0))).await
}

/// Shared read-only physical-hash counter for an optional local monitor. The
/// caller supplies a new zeroed counter; no listener or API server is created.
pub async fn run_multi_observed(
    config: Config,
    engines: Vec<Arc<dyn MinerEngine>>,
    mut shutdown: watch::Receiver<bool>,
    hashes: Arc<AtomicU64>,
) -> Result<Stats> {
    config.validate()?;
    ensure!(
        !engines.is_empty() && engines.len() <= 64,
        "expected 1..64 explicit mining engines"
    );
    let (mut worker, mut shares) =
        Worker::with_hash_counter(engines, config.range_size, rand::random(), hashes);
    let mut stats = Stats::default();
    let mut next_id = 1;
    let mut endpoint_index = 0;
    let mut selected = config.clone();
    let result = loop {
        if *shutdown.borrow() {
            break Ok(());
        }
        if let Err(error) = check_worker_failure(&worker) {
            break Err(error.context("mining worker failure; all devices stopped"));
        }
        match connection(
            &selected,
            &worker,
            &mut shares,
            &mut shutdown,
            &mut stats,
            &mut next_id,
        )
        .await
        {
            Ok(()) => break Ok(()),
            Err(e) => {
                worker.cancel();
                log::warn!("Stratum connection stopped: {e}");
                if e.downcast_ref::<WorkerFailure>().is_some() {
                    break Err(e.context("mining worker failure; all devices stopped"));
                }
                if stats.reconnects >= config.reconnect_attempts {
                    break Err(e.context("Stratum reconnect budget exhausted"));
                }
                stats.reconnects += 1;
                endpoint_index = (endpoint_index + 1) % (config.fallback_pools.len() + 1);
                let endpoint = config.endpoint(endpoint_index);
                selected.host = endpoint.host;
                selected.port = endpoint.port;
                log::info!(
                    "Retrying explicitly configured TLS endpoint {}:{}",
                    selected.host,
                    selected.port
                );
                tokio::select! { _ = shutdown_signal(&mut shutdown) => break Ok(()), _ = tokio::time::sleep(config.reconnect_delay) => {} }
            }
        }
    };
    worker.stop();
    if let Some(handle) = worker.handle.take() {
        worker.extra_handles.push(handle);
    }
    let handles = std::mem::take(&mut worker.extra_handles);
    let panicked = tokio::task::spawn_blocking(move || {
        let mut panicked = false;
        for handle in handles {
            panicked |= handle.join().is_err();
        }
        panicked
    })
    .await
    .context("worker join task failed")?;
    ensure!(!panicked, "mining worker panicked");
    stats.hashes = worker.state.hashes.load(Ordering::Relaxed);
    if result.is_err() {
        log::error!("Stratum failed-session summary: {stats:?}");
    }
    result.map(|()| stats)
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod operational_tests;

#[cfg(test)]
mod protocol_update_tests;
