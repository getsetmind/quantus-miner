//! Optional, read-only NVIDIA telemetry. Never modifies the driver or hardware.
use std::{
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{io::AsyncReadExt, process::Command, sync::watch};

const MAX_OUTPUT: u64 = 16 * 1024;
const QUERY_TIMEOUT: Duration = Duration::from_secs(5);
const QUERY: &str = "index,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total,uuid,clocks.sm,clocks.mem,driver_version,power.limit";

#[derive(Debug, PartialEq)]
pub(crate) struct Sample {
    pub index: u32,
    pub utilization_percent: Option<f64>,
    pub temperature_c: Option<f64>,
    pub power_w: Option<f64>,
    pub memory_used_mib: Option<f64>,
    pub memory_total_mib: Option<f64>,
    pub uuid: Option<String>,
    pub sm_clock_mhz: Option<f64>,
    pub memory_clock_mhz: Option<f64>,
    pub driver_version: Option<String>,
    pub power_limit_w: Option<f64>,
}

pub(crate) struct MiningMonitor {
    pub selected_uuids: Vec<String>,
    pub hashes: Arc<AtomicU64>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Efficiency {
    pub physical_mh_s: f64,
    pub selected_current_power_w: f64,
    pub approximate_mh_s_per_w: f64,
}

fn unsupported(value: &str) -> bool {
    matches!(value, "N/A" | "[N/A]" | "[Not Supported]" | "Not Supported")
}

fn canonical_uuid(value: &str, prefix: &str) -> bool {
    let Some(body) = value.strip_prefix(prefix) else {
        return false;
    };
    body.len() == 36
        && body.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

fn uuid(value: &str) -> Result<Option<String>, String> {
    if unsupported(value) {
        return Ok(None);
    }
    // Keep MIG identity diagnostic, but never map it to a physical GPU's power.
    let mig_legacy = value.strip_prefix("MIG-").is_some_and(|body| {
        let pieces: Vec<_> = body.split('/').collect();
        pieces.len() == 3
            && canonical_uuid(pieces[0], "GPU-")
            && pieces[1..].iter().all(|piece| {
                !piece.is_empty()
                    && piece.len() <= 10
                    && piece.bytes().all(|byte| byte.is_ascii_digit())
            })
    });
    if value.len() > 80
        || !(canonical_uuid(value, "GPU-") || canonical_uuid(value, "MIG-") || mig_legacy)
    {
        return Err("invalid bounded GPU UUID".into());
    }
    Ok(Some(value.to_owned()))
}

fn driver_version(value: &str) -> Result<Option<String>, String> {
    if unsupported(value) {
        return Ok(None);
    }
    if value.len() > 32
        || value
            .split('.')
            .any(|piece| piece.is_empty() || !piece.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err("invalid bounded driver version".into());
    }
    Ok(Some(value.to_owned()))
}

/// Approximate interval physical hash rate divided by current selected-GPU watts.
/// This is neither accepted-work efficiency nor an integrated-energy estimate.
pub(crate) fn efficiency(
    selected: &[String],
    samples: &[Sample],
    hashes: u64,
    elapsed: Duration,
) -> Result<Efficiency, &'static str> {
    if selected.is_empty() || selected.len() > 64 {
        return Err("no valid selected GPU UUIDs");
    }
    if hashes == 0 || elapsed.is_zero() {
        return Err("no completed physical-hash interval");
    }
    let mut power = 0.0;
    for (index, selected_uuid) in selected.iter().enumerate() {
        if !canonical_uuid(selected_uuid, "GPU-") || selected[..index].contains(selected_uuid) {
            return Err("selected GPU UUIDs are noncanonical or duplicated");
        }
        let matches: Vec<_> = samples
            .iter()
            .filter(|sample| sample.uuid.as_deref() == Some(selected_uuid.as_str()))
            .collect();
        if matches.len() != 1 {
            return Err("selected GPU UUID mapping is missing or ambiguous");
        }
        match matches[0].power_w {
            Some(watts) if watts.is_finite() && watts > 0.0 => power += watts,
            _ => return Err("selected GPU power is unavailable or zero"),
        }
    }
    let physical_mh_s = hashes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let approximate_mh_s_per_w = physical_mh_s / power;
    if !power.is_finite() || !physical_mh_s.is_finite() || !approximate_mh_s_per_w.is_finite() {
        return Err("non-finite efficiency inputs");
    }
    Ok(Efficiency {
        physical_mh_s,
        selected_current_power_w: power,
        approximate_mh_s_per_w,
    })
}

fn number(value: &str) -> Result<Option<f64>, String> {
    let value = value.trim();
    if unsupported(value) {
        return Ok(None);
    }
    let number: f64 = value.parse().map_err(|_| "invalid numeric GPU telemetry")?;
    if !number.is_finite() || number < 0.0 {
        return Err("non-finite or negative GPU telemetry".into());
    }
    Ok(Some(number))
}

pub(crate) fn parse_samples(bytes: &[u8]) -> Result<Vec<Sample>, String> {
    if bytes.len() as u64 > MAX_OUTPUT {
        return Err("GPU telemetry output exceeded limit".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "GPU telemetry is not UTF-8")?;
    let mut samples = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if samples.len() >= 64 {
            return Err("too many GPU telemetry rows".into());
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 11 {
            return Err("unexpected GPU telemetry columns".into());
        }
        let sample = Sample {
            index: fields[0].parse().map_err(|_| "invalid GPU index")?,
            utilization_percent: number(fields[1])?,
            temperature_c: number(fields[2])?,
            power_w: number(fields[3])?,
            memory_used_mib: number(fields[4])?,
            memory_total_mib: number(fields[5])?,
            uuid: uuid(fields[6])?,
            sm_clock_mhz: number(fields[7])?,
            memory_clock_mhz: number(fields[8])?,
            driver_version: driver_version(fields[9])?,
            power_limit_w: number(fields[10])?,
        };
        if sample
            .utilization_percent
            .is_some_and(|value| value > 100.0)
            || sample.temperature_c.is_some_and(|value| value > 200.0)
            || matches!((sample.memory_used_mib, sample.memory_total_mib), (Some(used), Some(total)) if used > total)
            || samples
                .iter()
                .any(|previous: &Sample| previous.index == sample.index)
        {
            return Err("inconsistent GPU telemetry".into());
        }
        samples.push(sample);
    }
    if samples.is_empty() {
        return Err("empty GPU telemetry".into());
    }
    Ok(samples)
}

#[derive(Debug)]
pub(crate) enum QueryError {
    Missing,
    Failed(String),
}

pub(crate) async fn query_program(
    program: &str,
    arguments: &[&str],
    timeout: Duration,
) -> Result<Vec<Sample>, QueryError> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                QueryError::Missing
            } else {
                QueryError::Failed(format!("could not start telemetry query: {error}"))
            }
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| QueryError::Failed("missing query stdout".into()))?;
    let mut limited = stdout.take(MAX_OUTPUT + 1);
    let mut bytes = Vec::new();
    let result = tokio::time::timeout(timeout, async {
        limited
            .read_to_end(&mut bytes)
            .await
            .map_err(|error| QueryError::Failed(error.to_string()))?;
        if bytes.len() as u64 > MAX_OUTPUT {
            return Err(QueryError::Failed(
                "GPU telemetry output exceeded limit".into(),
            ));
        }
        let status = child
            .wait()
            .await
            .map_err(|error| QueryError::Failed(error.to_string()))?;
        if !status.success() {
            return Err(QueryError::Failed(
                "nvidia-smi query returned an error".into(),
            ));
        }
        parse_samples(&bytes).map_err(QueryError::Failed)
    })
    .await;
    let result = match result {
        Ok(value) => value,
        Err(_) => Err(QueryError::Failed("GPU telemetry query timed out".into())),
    };
    if result.is_err() {
        // Explicitly kill and reap ordinary failures; cancellation retains kill_on_drop.
        let _ = tokio::time::timeout(Duration::from_secs(1), child.kill()).await;
    }
    result
}

/// Run independently from batch completion. Stop promptly on session shutdown.
pub(crate) async fn run(
    interval: Duration,
    mut shutdown: watch::Receiver<bool>,
    monitor: Option<MiningMonitor>,
) {
    if interval.is_zero() {
        return;
    }
    let mut previous: Option<(Instant, u64)> = None;
    let mut tick = tokio::time::interval(interval.max(Duration::from_secs(5)));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        if *shutdown.borrow() {
            break;
        }
        tokio::select! {
            result = shutdown.changed() => { if result.is_err() || *shutdown.borrow() { break; } }
            _ = tick.tick() => {
                let arguments = [format!("--query-gpu={QUERY}"), "--format=csv,noheader,nounits".into()];
                let refs: Vec<_> = arguments.iter().map(String::as_str).collect();
                let result = tokio::select! {
                    _ = shutdown.changed() => break,
                    result = query_program("nvidia-smi", &refs, QUERY_TIMEOUT) => result,
                };
                match result {
                    Ok(samples) => {
                        for sample in &samples { log::info!("GPU telemetry: {sample:?}"); }
                        if let Some(monitor) = &monitor {
                            let now = Instant::now();
                            let hashes = monitor.hashes.load(Ordering::Relaxed);
                            let estimate = previous.and_then(|(then, before)| hashes.checked_sub(before).map(|delta| (delta, now.duration_since(then))));
                            match estimate {
                                Some((delta, elapsed)) => match efficiency(&monitor.selected_uuids, &samples, delta, elapsed) {
                                    Ok(value) => log::info!("Approximate GPU interval physical-hash efficiency (not accepted-work or integrated energy): interval_s={:.3} physical_mh_s={:.3} selected_current_power_w={:.3} approximate_mh_s_per_w={:.6}", elapsed.as_secs_f64(), value.physical_mh_s, value.selected_current_power_w, value.approximate_mh_s_per_w),
                                    Err(reason) => log::info!("GPU efficiency unavailable: {reason}"),
                                },
                                None => log::info!("GPU efficiency unavailable: awaiting completed-hash interval baseline"),
                            }
                            previous = Some((now, hashes));
                        }
                    },
                    Err(QueryError::Missing) => {
                        log::warn!("GPU telemetry disabled: optional nvidia-smi was not found");
                        break;
                    }
                    Err(QueryError::Failed(message)) => log::warn!("GPU telemetry unavailable: {message}"),
                }
            }
        }
    }
}
