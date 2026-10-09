//! Bounded, explicit pool mining. No wallet keys, clock changes, or metrics listener.

use anyhow::{anyhow, bail, Result};
use clap::{Args, ValueEnum};
use engine_cpu::{FastCpuEngine, MinerEngine};
use std::sync::{atomic::AtomicU64, Arc};
use std::time::Duration;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum Engine {
    Cpu,
    Cuda,
}

#[derive(Args, Debug)]
pub(crate) struct StratumArgs {
    /// Public qz payout address only; never a recovery phrase or private key
    #[arg(long, value_parser = validate_wallet)]
    pub wallet: String,

    /// Non-secret label identifying this machine
    #[arg(long, default_value = "rig1", value_parser = validate_worker)]
    worker: String,

    /// TLS pool hostname (no URL scheme); certificate verification is mandatory
    #[arg(long, default_value = "quantus.suprnova.cc")]
    pool_host: String,

    /// TLS Stratum port; this command does not support unencrypted TCP
    #[arg(long, default_value_t = 7074, value_parser = clap::value_parser!(u16).range(1..))]
    pool_port: u16,

    /// One CPU worker or explicitly selected CUDA GPUs; no automatic engine fallback
    #[arg(long, value_enum, default_value = "cuda")]
    engine: Engine,

    /// Explicit CUDA-visible ordinals, comma-separated (default: device 0)
    #[arg(long, value_delimiter = ',', num_args = 1.., default_value = "0")]
    cuda_devices: Vec<usize>,

    /// Explicit alternate TLS hostname:port; repeat for an ordered failover list
    #[arg(long, value_parser = validate_endpoint)]
    fallback_pool: Vec<String>,

    /// Read-only nvidia-smi sampling interval in seconds; 0 disables telemetry
    #[arg(long, default_value_t = 0, value_parser = validate_telemetry_interval)]
    gpu_telemetry_interval: u64,

    /// Session duration in seconds, then stop (also stop with Ctrl+C)
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u64).range(1..=86400))]
    duration: u64,

    /// CUDA dispatch size; independent of pool difficulty
    #[arg(long, default_value_t = 1_000_000, value_parser = clap::value_parser!(u32).range(1..))]
    gpu_batch_size: u32,

    /// Delay between CUDA batches; does not change GPU clocks or power limits
    #[arg(long, default_value_t = 0)]
    gpu_throttle_ms: u64,

    /// Maximum additional connection attempts after the first attempt
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(0..=20))]
    reconnect_attempts: u32,

    #[arg(short, long)]
    pub verbose: bool,
}

fn validate_endpoint(value: &str) -> std::result::Result<String, String> {
    stratum_service::Endpoint::parse(value).map_err(|e| e.to_string())?;
    Ok(value.to_owned())
}
fn validate_telemetry_interval(value: &str) -> std::result::Result<u64, String> {
    let seconds: u64 = value.parse().map_err(|_| "expected seconds".to_owned())?;
    if seconds != 0 && !(5..=3600).contains(&seconds) {
        return Err("telemetry interval must be 0 or 5..3600 seconds".into());
    }
    Ok(seconds)
}

fn validate_wallet(value: &str) -> std::result::Result<String, String> {
    const BASE58: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    if !value.starts_with("qz")
        || !(40..=80).contains(&value.len())
        || !value.chars().all(|c| BASE58.contains(c))
    {
        return Err(
            "expected a public qz payout address, not a phrase, URL, or private key".into(),
        );
    }
    // Structural validation only; address ownership and chain checksum are not inferred.
    Ok(value.to_owned())
}

fn validate_worker(value: &str) -> std::result::Result<String, String> {
    if value.is_empty()
        || value.len() > 32
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("worker must be 1-32 ASCII letters, digits, '-' or '_'".into());
    }
    Ok(value.to_owned())
}

pub(crate) async fn run(args: StratumArgs) -> Result<()> {
    if args.pool_host.is_empty()
        || args.pool_host.contains(['/', ':', '@'])
        || args.pool_host.chars().any(char::is_whitespace)
    {
        bail!("pool-host must be a TLS hostname without a scheme, path, or port");
    }
    if args.cuda_devices.is_empty()
        || args.cuda_devices.len() > 64
        || args
            .cuda_devices
            .iter()
            .enumerate()
            .any(|(i, n)| args.cuda_devices[..i].contains(n))
    {
        bail!("select 1..64 distinct CUDA-visible ordinals");
    }
    if args.fallback_pool.len() > 8 {
        bail!("at most eight explicit TLS fallback pools");
    }
    let mut selected_uuids = Vec::new();
    let engines: Vec<Arc<dyn MinerEngine>> = match args.engine {
        Engine::Cpu => {
            if args.cuda_devices != [0] {
                bail!("cuda-devices is only supported with the CUDA engine");
            }
            vec![Arc::new(FastCpuEngine::new(256))]
        }
        Engine::Cuda => {
            let cuda = engine_cuda::CudaEngine::try_new_on_devices(
                args.gpu_batch_size,
                args.gpu_throttle_ms,
                &args.cuda_devices,
            )
            .map_err(|error| anyhow!("CUDA initialization failed: {error}"))?;
            let devices = cuda.into_device_engines();
            let uuids = devices
                .iter()
                .map(|device| device.device_uuid())
                .collect::<std::result::Result<Vec<_>, _>>();
            match uuids {
                Ok(uuids) => selected_uuids = uuids,
                Err(_) => log::warn!(
                    "Selected GPU UUID mapping unavailable; efficiency monitoring disabled"
                ),
            }
            devices
                .into_iter()
                .map(|engine| Arc::new(engine) as Arc<dyn MinerEngine>)
                .collect()
        }
    };
    let config = stratum_service::Config {
        host: args.pool_host,
        port: args.pool_port,
        fallback_pools: args
            .fallback_pool
            .iter()
            .map(|s| stratum_service::Endpoint::parse(s))
            .collect::<Result<Vec<_>>>()?,
        login: format!("{}.{}", args.wallet, args.worker),
        password: "x".to_owned(),
        agent: format!("quantus-miner-stratum/{}", env!("CARGO_PKG_VERSION")),
        reconnect_attempts: args.reconnect_attempts,
        reconnect_delay: Duration::from_secs(2),
        connect_timeout: Duration::from_secs(10),
        response_timeout: Duration::from_secs(45),
        range_size: u64::from(args.gpu_batch_size),
    };
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let hashes = Arc::new(AtomicU64::new(0));
    let monitor = (!selected_uuids.is_empty()).then(|| crate::gpu_telemetry::MiningMonitor {
        selected_uuids,
        hashes: hashes.clone(),
    });
    let telemetry = (args.gpu_telemetry_interval > 0).then(|| {
        tokio::spawn(crate::gpu_telemetry::run(
            Duration::from_secs(args.gpu_telemetry_interval),
            shutdown_rx.clone(),
            monitor,
        ))
    });
    let duration = args.duration;
    let timer = tokio::spawn(async move {
        tokio::select! {
            () = tokio::time::sleep(Duration::from_secs(duration)) => {
                log::info!("Session duration reached; stopping pool worker");
            }
            result = tokio::signal::ctrl_c() => {
                if let Err(error) = result {
                    log::warn!("Ctrl+C handler failed: {error}; stopping safely");
                }
            }
        }
        let _ = shutdown_tx.send(true);
    });
    log::info!("Starting experimental TLS Stratum session; stop requested after {duration}s (in-flight GPU work may delay exit)");
    let result = stratum_service::run_multi_observed(config, engines, shutdown_rx, hashes).await;
    timer.abort();
    if let Some(telemetry) = telemetry {
        telemetry.abort();
        let _ = telemetry.await;
    }
    match result {
        Ok(stats) => {
            log::info!("Stratum session summary: {stats:?}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn reject_secrets_and_worker_injection() {
        assert!(validate_wallet("one two three four five six seven eight nine ten").is_err());
        assert!(validate_wallet("https://example.org/qz123").is_err());
        assert!(validate_worker("worker\nsubmit").is_err());
        assert!(validate_worker("rig-1_ok").is_ok());
    }

    #[test]
    fn stratum_help_and_duration_are_bounded() {
        let wallet = format!("qz{}", "1".repeat(46));
        assert!(crate::Args::try_parse_from(["miner", "stratum", "--wallet", &wallet]).is_ok());
        assert!(crate::Args::try_parse_from([
            "miner",
            "stratum",
            "--wallet",
            &wallet,
            "--duration",
            "0"
        ])
        .is_err());
        assert!(crate::Args::try_parse_from([
            "miner",
            "stratum",
            "--wallet",
            &wallet,
            "--duration",
            "86401"
        ])
        .is_err());
    }
}
