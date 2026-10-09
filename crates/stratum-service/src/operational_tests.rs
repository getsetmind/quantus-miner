use super::*;
use engine_cpu::FastCpuEngine;

#[test]
fn explicit_endpoints_are_tls_hostnames_with_bounded_ports() {
    for value in ["pool.example:7074", "localhost:1234"] {
        assert!(Endpoint::parse(value).is_ok());
    }
    for value in [
        "http://pool.example:80",
        "stratum+tcp://pool.example:1",
        "user@pool.example:1",
        "pool.example:0",
        "pool.example:65536",
        "pool.example:1/path",
        "pool.example",
        "pool.example:1\n",
        "[::1]:7074",
    ] {
        assert!(Endpoint::parse(value).is_err(), "accepted {value}");
    }
    let config = Config {
        login: "synthetic".into(),
        fallback_pools: vec![Endpoint {
            host: "bad/url".into(),
            port: 1,
        }],
        ..Config::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn all_worker_generation_partitions_are_disjoint_and_prefix_bounded() {
    let prefix = [255; 4];
    let salt = [42; 16];
    let mut previous = None;
    for generation in 1..=3 {
        for device in 0..64 {
            let (start, end) = worker_nonce_partition(prefix, salt, generation, device);
            assert!(start < end);
            assert_eq!(&start.to_big_endian()[..4], &prefix);
            assert_eq!(&end.to_big_endian()[..4], &prefix);
            assert_eq!(&start.to_big_endian()[28..32], &device.to_be_bytes());
            if let Some(previous) = previous {
                assert!(previous < start);
            }
            previous = Some(end);
        }
    }
}

#[tokio::test]
async fn mock_two_devices_emit_cpu_verified_unique_shares_and_cancel_together() {
    let engines: Vec<Arc<dyn MinerEngine>> = vec![
        Arc::new(FastCpuEngine::new(1)),
        Arc::new(FastCpuEngine::new(1)),
    ];
    let (mut worker, mut shares) = Worker::with_engines(engines, 4, [31; 16]);
    let job = tests::easy_job();
    let assigned = worker.assign(job).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    let mut devices = std::collections::BTreeSet::new();
    for _ in 0..1000 {
        let event = timeout(Duration::from_secs(10), shares.recv())
            .await
            .unwrap()
            .unwrap();
        match event {
            WorkerEvent::Share(share) => {
                let (a, c) = *share;
                assert_eq!(a.generation, assigned.generation);
                assert!(a.job.verify(&c));
                assert!(seen.insert(c.nonce), "duplicate multi-device nonce");
                let device =
                    u32::from_be_bytes(c.nonce.to_big_endian()[28..32].try_into().unwrap());
                assert!(device < 2);
                devices.insert(device);
            }
            WorkerEvent::Failed(e) => panic!("{e}"),
        }
        if devices.len() == 2 {
            break;
        }
    }
    assert_eq!(devices.len(), 2);
    worker.cancel();
    assert!(worker.state.latest.lock().unwrap().is_none());
    assert_ne!(
        worker.state.epoch.load(Ordering::SeqCst),
        assigned.generation
    );
    let newer = worker.assign(assigned.job).unwrap();
    assert!(newer.generation > assigned.generation);
    worker.stop();
    let mut handles = std::mem::take(&mut worker.extra_handles);
    handles.push(worker.handle.take().unwrap());
    timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || {
            for handle in handles {
                handle.join().unwrap();
            }
        }),
    )
    .await
    .unwrap()
    .unwrap();
}

#[tokio::test]
async fn invalid_worker_counts_fail_without_a_network_connection() {
    let (_, shutdown) = watch::channel(false);
    let config = Config {
        login: "synthetic".into(),
        ..Config::default()
    };
    assert!(run_multi(config, Vec::new(), shutdown).await.is_err());
}

#[test]
fn failover_rotates_only_explicit_endpoints_and_preserves_primary() {
    let config = Config {
        host: "primary.example".into(),
        fallback_pools: vec![
            Endpoint::parse("alternate.example:1234").unwrap(),
            Endpoint::parse("second.example:2345").unwrap(),
        ],
        ..Config::default()
    };
    assert_eq!(config.endpoint(0).host, "primary.example");
    assert_eq!(config.endpoint(1).host, "alternate.example");
    assert_eq!(config.endpoint(1).port, 1234);
    assert_eq!(config.endpoint(2).host, "second.example");
    assert_eq!(config.endpoint(3).host, "primary.example");
    let default = Config::default();
    for retry in 0..21 {
        assert_eq!(default.endpoint(retry).host, "quantus.suprnova.cc");
    }
}

#[test]
fn authenticated_null_initial_job_waits_without_hashing_then_accepts_notification() {
    let worker = tests::fake_worker();
    let mut session = Session::new();
    let mut stats = Stats::default();
    session.register(1, RequestKind::Login).unwrap();
    session.handle(json!({"id":1,"result":{"status":"OK","id":"synthetic-waiting","job":null,"extensions":["keepalive"]},"error":null}), &worker, &mut stats).unwrap();
    assert_eq!(session.token.as_deref(), Some("synthetic-waiting"));
    assert!(session.active.is_none());
    assert!(worker.state.latest.lock().unwrap().is_none());
    assert_eq!(worker.state.hashes.load(Ordering::Relaxed), 0);
    session.handle(json!({"jsonrpc":"2.0","method":"job","params":{"clean_jobs":true,"job":tests::fixture()}}), &worker, &mut stats).unwrap();
    assert!(session.active.is_some());
}

struct BrokenEngine {
    invalid_candidate: bool,
}
impl MinerEngine for BrokenEngine {
    fn name(&self) -> &'static str {
        "offline-broken-engine"
    }
    fn prepare_context(&self, header: [u8; 32], difficulty: U512) -> pow_core::JobContext {
        pow_core::JobContext {
            header,
            difficulty,
            target: U512::MAX / difficulty,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn search_range(
        &self,
        _: &pow_core::JobContext,
        range: Range,
        _: &dyn engine_cpu::CancelCheck,
    ) -> EngineStatus {
        if self.invalid_candidate {
            EngineStatus::Found {
                candidate: Candidate {
                    nonce: range.start,
                    work: range.start.to_big_endian(),
                    hash: U512::zero(),
                },
                hash_count: 1,
                origin: engine_cpu::FoundOrigin::Cpu,
            }
        } else {
            EngineStatus::DeviceLost { hash_count: 1 }
        }
    }
}

#[tokio::test]
async fn invalid_candidate_and_device_loss_are_fatal_worker_errors_cancel_all_devices() {
    for invalid_candidate in [false, true] {
        let engines: Vec<Arc<dyn MinerEngine>> = vec![
            Arc::new(BrokenEngine { invalid_candidate }),
            Arc::new(FastCpuEngine::new(1)),
        ];
        let (mut worker, _shares) = Worker::with_engines(engines, 4, [0; 16]);
        worker.assign(tests::easy_job()).unwrap();
        timeout(Duration::from_secs(10), async {
            while check_worker_failure(&worker).is_ok() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        let (_sender, mut failures) = mpsc::channel(1);
        let (stream, _pool) = tokio::io::duplex(65536);
        let (_shutdown_tx, mut shutdown) = watch::channel(false);
        let config = Config {
            login: "synthetic".into(),
            ..Config::default()
        };
        let mut stats = Stats::default();
        let mut id = 1;
        let error = timeout(
            Duration::from_secs(10),
            session_loop(
                &config,
                &worker,
                &mut failures,
                &mut shutdown,
                &mut stats,
                &mut id,
                stream,
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(
            error.downcast_ref::<WorkerFailure>().is_some(),
            "worker failure must not enter pool failover"
        );
        assert!(worker.state.latest.lock().unwrap().is_none());
        assert_eq!(stats.submitted, 0);
        worker.stop();
        let mut handles = std::mem::take(&mut worker.extra_handles);
        handles.push(worker.handle.take().unwrap());
        timeout(
            Duration::from_secs(10),
            tokio::task::spawn_blocking(move || {
                for handle in handles {
                    handle.join().unwrap();
                }
            }),
        )
        .await
        .unwrap()
        .unwrap();
    }
}

#[tokio::test]
async fn fatal_device_failure_survives_full_queue_and_reconnect_cancellation() {
    let worker = tests::fake_worker();
    let assigned = worker.assign(tests::easy_job()).unwrap();
    let (sender, _receiver) = mpsc::channel(1);
    sender
        .try_send(WorkerEvent::Failed("queued placeholder"))
        .unwrap_or_else(|_| panic!("queue fill"));
    report_worker_failure(
        &sender,
        &worker.state,
        assigned.generation,
        "synthetic device lost",
    );
    assert!(check_worker_failure(&worker)
        .unwrap_err()
        .downcast_ref::<WorkerFailure>()
        .is_some());
    assert!(worker.state.latest.lock().unwrap().is_none());
    assert_ne!(
        worker.state.epoch.load(Ordering::SeqCst),
        assigned.generation
    );
    worker.cancel();
    assert!(check_worker_failure(&worker).is_err());
}
