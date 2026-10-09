use clap::Parser;

fn args(extra: &[&str]) -> Vec<String> {
    let mut args = vec![
        "miner".into(),
        "stratum".into(),
        "--wallet".into(),
        format!("qz{}", "1".repeat(46)),
    ];
    args.extend(extra.iter().map(|s| (*s).into()));
    args
}

#[test]
fn explicit_multi_device_and_tls_fallback_options_parse() {
    assert!(crate::Args::try_parse_from(args(&[
        "--cuda-devices",
        "2,0",
        "--fallback-pool",
        "backup.example:7074",
        "--fallback-pool",
        "another.example:443",
        "--gpu-telemetry-interval",
        "5"
    ]))
    .is_ok());
}

#[test]
fn malformed_endpoints_and_unsafe_telemetry_intervals_fail_before_execution() {
    for endpoint in [
        "tcp://pool.example:80",
        "https://pool.example:443",
        "pool.example:0",
        "pool.example",
        "pool.example:99999",
        "user@pool.example:7074",
    ] {
        assert!(crate::Args::try_parse_from(args(&["--fallback-pool", endpoint])).is_err());
    }
    for interval in ["1", "4", "3601", "-1"] {
        assert!(
            crate::Args::try_parse_from(args(&["--gpu-telemetry-interval", interval])).is_err()
        );
    }
    assert!(crate::Args::try_parse_from(args(&["--gpu-telemetry-interval", "0"])).is_ok());
}
