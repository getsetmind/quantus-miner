//! Bounded, explicit pool mining. No wallet keys, clock changes, or metrics listener.

use anyhow::{anyhow, bail, Result};
use clap::{Args, ValueEnum};
use engine_cpu::{FastCpuEngine, MinerEngine};
use std::sync::Arc;
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

    /// One CPU worker or one visible CUDA GPU; no automatic fallback
    #[arg(long, value_enum, default_value = "cuda")]
    engine: Engine,

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
    let engine: Arc<dyn MinerEngine> = match args.engine {
        Engine::Cpu => Arc::new(FastCpuEngine::new(256)),
        Engine::Cuda => {
            let cuda = engine_cuda::CudaEngine::try_new(args.gpu_batch_size, args.gpu_throttle_ms)
                .map_err(|error| anyhow!("CUDA initialization failed: {error}"))?;
            if cuda.device_count() != 1 {
                bail!("experimental Stratum mode requires exactly one visible CUDA GPU; use CUDA_VISIBLE_DEVICES to select one");
            }
            Arc::new(cuda)
        }
    };
    let config = stratum_service::Config {
        host: args.pool_host,
        port: args.pool_port,
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
    log::info!("Starting experimental TLS Stratum session; stop requested after {duration}s (in-flight GPU work may delay exit); share acceptance is not yet live-validated");
    let result = stratum_service::run(config, engine, shutdown_rx).await;
    timer.abort();
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
