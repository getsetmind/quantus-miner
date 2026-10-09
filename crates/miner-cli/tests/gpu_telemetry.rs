#[path = "../src/gpu_telemetry.rs"]
mod gpu_telemetry;

#[test]
fn parses_numeric_and_optional_values() {
    let samples = gpu_telemetry::parse_samples(
        b"0, 99, 65, 112.50, 4096, 12288, GPU-11111111-1111-1111-1111-111111111111, 1500, 7000, 555.85, 170\n1, N/A, [Not Supported], N/A, 0, 8192, N/A, N/A, N/A, N/A, N/A\n",
    )
    .unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].power_w, Some(112.5));
    assert_eq!(samples[1].utilization_percent, None);
    assert_eq!(samples[0].sm_clock_mhz, Some(1500.0));
    assert_eq!(samples[0].memory_clock_mhz, Some(7000.0));
    assert_eq!(samples[0].driver_version.as_deref(), Some("555.85"));
    assert_eq!(samples[0].power_limit_w, Some(170.0));
}

#[test]
fn rejects_malformed_and_unbounded_output() {
    for value in [
        "",
        "0,1,2",
        "0, NaN, 50, 1, 0, 1",
        "0,101,50,1,0,1",
        "0,1,50,1,2,1",
        "0,1,50,1,0,1\n0,1,50,1,0,1",
    ] {
        assert!(
            gpu_telemetry::parse_samples(value.as_bytes()).is_err(),
            "{value}"
        );
    }
    assert!(gpu_telemetry::parse_samples(&vec![b' '; 16385]).is_err());
}

#[test]
fn gpu_row_count_is_bounded() {
    let sixty_four = (0..64)
        .map(|index| format!("{index},1,50,1,0,1,N/A,1,1,555.85,170\n"))
        .collect::<String>();
    assert_eq!(
        gpu_telemetry::parse_samples(sixty_four.as_bytes())
            .unwrap()
            .len(),
        64
    );
    assert!(gpu_telemetry::parse_samples(
        format!("{sixty_four}64,1,50,1,0,1,N/A,1,1,555.85,170\n").as_bytes()
    )
    .is_err());
}

mod process_tests {
    use super::gpu_telemetry::{query_program, run, QueryError};
    use std::time::Duration;
    use tokio::sync::watch;
    const QUERY_TIMEOUT: Duration = Duration::from_secs(5);
    #[tokio::test]
    async fn missing_optional_program_is_safe() {
        assert!(matches!(
            query_program("quantus-no-such-telemetry-program", &[], QUERY_TIMEOUT).await,
            Err(QueryError::Missing)
        ));
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn query_timeout_and_output_cap() {
        assert!(
            matches!(query_program("sh", &["-c", "exec sleep 10"], Duration::from_millis(30)).await, Err(QueryError::Failed(message)) if message.contains("timed out"))
        );
        assert!(
            matches!(query_program("sh", &["-c", "exec head -c 20000 /dev/zero"], QUERY_TIMEOUT).await, Err(QueryError::Failed(message)) if message.contains("limit"))
        );
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn successful_query_and_failed_exit() {
        let samples = query_program(
            "sh",
            &["-c", "printf '0,99,65,100,1024,4096,GPU-11111111-1111-1111-1111-111111111111,1500,7000,555.85,170\\n'"],
            QUERY_TIMEOUT,
        )
        .await
        .unwrap();
        assert_eq!(samples[0].temperature_c, Some(65.0));
        assert!(matches!(
            query_program("sh", &["-c", "exit 7"], QUERY_TIMEOUT).await,
            Err(QueryError::Failed(_))
        ));
    }

    #[tokio::test]
    async fn stopped_monitor_does_not_query() {
        let (_sender, receiver) = watch::channel(true);
        tokio::time::timeout(
            Duration::from_millis(100),
            run(Duration::from_secs(5), receiver, None),
        )
        .await
        .unwrap();
    }
}

fn gpu_uuid(index: u32) -> String {
    format!("GPU-11111111-1111-1111-1111-{index:012x}")
}
fn samples(powers: &[&str]) -> Vec<gpu_telemetry::Sample> {
    let text: String = powers
        .iter()
        .enumerate()
        .map(|(index, power)| {
            format!(
                "{index},99,65,{power},1024,4096,{},1500,7000,555.85,170\n",
                gpu_uuid(index as u32)
            )
        })
        .collect();
    gpu_telemetry::parse_samples(text.as_bytes()).unwrap()
}

#[test]
fn aggregate_efficiency_excludes_unselected_gpu_power() {
    let selected = vec![gpu_uuid(0), gpu_uuid(2)];
    let samples = samples(&["100", "999", "200"]);
    let value = gpu_telemetry::efficiency(
        &selected,
        &samples,
        600_000_000,
        std::time::Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(value.physical_mh_s, 300.0);
    assert_eq!(value.selected_current_power_w, 300.0);
    assert_eq!(value.approximate_mh_s_per_w, 1.0);
}

#[test]
fn efficiency_requires_every_selected_gpu_mapping_and_positive_power() {
    use std::time::Duration;
    let elapsed = Duration::from_secs(1);
    let selected = vec![gpu_uuid(0), gpu_uuid(1)];
    for powers in [&["100", "0"][..], &["100", "N/A"][..], &["100"][..]] {
        assert!(
            gpu_telemetry::efficiency(&selected, &samples(powers), 1_000_000, elapsed).is_err()
        );
    }
    let valid = samples(&["100", "200"]);
    for selection in [
        vec![],
        vec![gpu_uuid(0), gpu_uuid(0)],
        vec!["MIG-11111111-1111-1111-1111-111111111111".into()],
        vec![gpu_uuid(7)],
    ] {
        assert!(gpu_telemetry::efficiency(&selection, &valid, 1_000_000, elapsed).is_err());
    }
    assert!(gpu_telemetry::efficiency(&selected, &valid, 0, elapsed).is_err());
    assert!(gpu_telemetry::efficiency(&selected, &valid, 1_000_000, Duration::ZERO).is_err());
    let mut ambiguous = samples(&["100", "200"]);
    ambiguous[1].uuid = ambiguous[0].uuid.clone();
    assert!(gpu_telemetry::efficiency(&[gpu_uuid(0)], &ambiguous, 1_000_000, elapsed).is_err());
    let mut invalid = samples(&["100"]);
    for power in [f64::NAN, f64::INFINITY, -1.0] {
        invalid[0].power_w = Some(power);
        assert!(gpu_telemetry::efficiency(&[gpu_uuid(0)], &invalid, 1_000_000, elapsed).is_err());
    }
}

#[test]
fn rejects_invalid_identity_driver_and_numeric_fields() {
    let valid = format!("0,99,65,100,1024,4096,{},1500,7000,555.85,170", gpu_uuid(0));
    for value in [
        valid.replace(&gpu_uuid(0), "GPU-bad"),
        valid.replace("555.85", "555.85 injected"),
        valid.replace("555.85", "555..85"),
        valid.replace("555.85", &"1".repeat(33)),
        valid.replace(",99,", ",101,"),
        valid.replace(",100,", ",NaN,"),
        valid.replace(",1500,", ",-1,"),
        valid.replace(",1024,4096,", ",5000,4096,"),
    ] {
        assert!(
            gpu_telemetry::parse_samples(value.as_bytes()).is_err(),
            "{value}"
        );
    }
    let mig = valid.replace(&gpu_uuid(0), "MIG-11111111-1111-1111-1111-111111111111");
    let sample = gpu_telemetry::parse_samples(mig.as_bytes()).unwrap();
    assert!(gpu_telemetry::efficiency(
        &[gpu_uuid(0)],
        &sample,
        1_000_000,
        std::time::Duration::from_secs(1)
    )
    .is_err());
}
