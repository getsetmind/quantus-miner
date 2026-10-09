use super::*;
use engine_cpu::FastCpuEngine;

pub(super) fn fixture() -> Value {
    json!({"algo":"qpow-poseidon2","difficulty":10000000000u64,"extranonce":"01002807","job_id":"141589","mining_hash":"3b7dfe994ef7f0724a96540c547724153b02c9a0bfc1e166872411d9b2cf8a4a","seq":136419,"target":"000000006df37f675ef6eadf5ab9a2072d44268d97df837e6748956e5c6c2117501e68855669e4b8356cf292464d9e16cc8d4655f8fb96f429b149fc87d74da4"})
}
pub(super) fn easy_job() -> Job {
    let mut v = fixture();
    v["difficulty"] = json!(1);
    v["target"] = json!(hex::encode(U512::MAX.to_big_endian()));
    Job::parse(&v).unwrap()
}
pub(super) fn fake_worker() -> Worker {
    Worker {
        state: Arc::new(WorkerState {
            salt: [0; 16],
            hashes: Arc::new(AtomicU64::new(0)),
            failure: Mutex::new(None),
            epoch: AtomicU64::new(0),
            latest: Mutex::new(None),
            wake: Condvar::new(),
        }),
        handle: None,
        extra_handles: Vec::new(),
    }
}
fn candidate(job: &Job, nonce: U512) -> Candidate {
    Candidate {
        nonce,
        work: nonce.to_big_endian(),
        hash: pow_core::hash_from_nonce(&job.context(), nonce),
    }
}
#[test]
fn parses_observed_login_fixture() {
    let worker = fake_worker();
    let mut state = Session::new();
    let mut stats = Stats::default();
    state.register(1, RequestKind::Login).unwrap();
    state.handle(json!({"id":1,"jsonrpc":"2.0","result":{"status":"OK","id":"synthetic-session","extensions":["keepalive"],"job":fixture()},"error":null}), &worker, &mut stats).unwrap();
    assert!(state.keepalive);
    assert_eq!(state.token.as_deref(), Some("synthetic-session"));
    assert_eq!(state.active.as_ref().unwrap().job.prefix, [1, 0, 40, 7]);
    assert_eq!(
        state.active.as_ref().unwrap().job.difficulty,
        U512::from(10_000_000_000u64)
    );
}
#[test]
fn rejects_malformed_job_fields_and_wrong_target() {
    for field in ["mining_hash", "extranonce", "target"] {
        for bad in ["00", "zz", "", "0x12"] {
            let mut v = fixture();
            v[field] = json!(bad);
            assert!(Job::parse(&v).is_err());
        }
    }
    for bad in [json!(0), json!(-1), json!(1.5), json!("NaN"), Value::Null] {
        let mut v = fixture();
        v["difficulty"] = bad;
        assert!(Job::parse(&v).is_err());
    }
    let mut v = fixture();
    v["target"] = json!(hex::encode(U512::MAX.to_big_endian()));
    assert!(Job::parse(&v).is_err());
    v = fixture();
    v["algo"] = json!("other");
    assert!(Job::parse(&v).is_err());
}
#[test]
fn strict_boundary_and_share_encoding() {
    let mut job = easy_job();
    let nonce = job.nonce_bounds().0;
    let c = candidate(&job, nonce);
    job.target = c.hash;
    assert!(!job.verify(&c));
    job.target = c.hash + U512::one();
    assert!(job.verify(&c));
    let request = protocol::submit_request(12, "synthetic-session", &job, &c).unwrap();
    assert_eq!(request["params"]["nonce"].as_str().unwrap().len(), 128);
    assert_eq!(
        request["params"]["result"],
        hex::encode(c.hash.to_big_endian())
    );
    assert_eq!(
        request["params"]["nonce"].as_str().unwrap().get(..8),
        Some("01002807")
    );
    let mut bad = c.clone();
    bad.hash += U512::one();
    assert!(!job.verify(&bad));
    bad = c.clone();
    bad.work[0] ^= 1;
    assert!(!job.verify(&bad));
    bad = candidate(&job, nonce - U512::one());
    assert!(!job.verify(&bad));
}
#[test]
fn nonce_prefix_carry_is_bounded() {
    let job = easy_job();
    let (start, end) = job.nonce_bounds();
    assert_eq!(&start.to_big_endian()[..4], &job.prefix);
    assert_eq!(&end.to_big_endian()[..4], &job.prefix);
    assert_eq!(protocol::next_nonce(end, end), None);
    let carry = start + U512::from(u64::MAX);
    assert_eq!(
        protocol::next_nonce(carry, end),
        Some(start + (U512::one() << 64))
    );
    let mut max_prefix = job;
    max_prefix.prefix = [255; 4];
    assert_eq!(max_prefix.nonce_bounds().1, U512::MAX);
    assert_eq!(protocol::next_nonce(U512::MAX, U512::MAX), None);
}
#[test]
fn framing_limits_fragmentation_and_multiple_frames() {
    let mut decoder = FrameDecoder::default();
    let mut frames = Vec::new();
    for byte in b"{\"id\":1}\n{\"id\":2}\r\n" {
        if let Some(v) = decoder.push(*byte).unwrap() {
            frames.push(v);
        }
    }
    assert_eq!(frames, vec![json!({"id":1}), json!({"id":2})]);
    assert!(decoder.push(b'\n').is_err());
    for _ in 0..protocol::MAX_FRAME_BYTES {
        assert!(decoder.push(b' ').is_ok());
    }
    assert!(decoder.push(b' ').is_err());
    let mut decoder = FrameDecoder::default();
    decoder.push(b'x').unwrap();
    assert!(decoder.push(b'\n').is_err());
}
#[test]
fn session_updates_disconnect_and_ack_accounting() {
    let worker = fake_worker();
    let mut s = Session::new();
    let mut stats = Stats::default();
    s.token = Some("fake".into());
    s.active = Some(worker.assign(easy_job()).unwrap());
    let old_generation = s.active.as_ref().unwrap().generation;
    let mut newer = fixture();
    newer["seq"] = json!(136420);
    newer["job_id"] = json!("next");
    s.handle(
        json!({"method":"job","params":{"job":newer}}),
        &worker,
        &mut stats,
    )
    .unwrap();
    assert!(s.active.as_ref().unwrap().generation > old_generation);
    let current = s.active.as_ref().unwrap().generation;
    s.handle(json!({"method":"job","params":newer}), &worker, &mut stats)
        .unwrap();
    assert_eq!(s.active.as_ref().unwrap().generation, current);
    assert!(s
        .handle(
            json!({"method":"job","params":fixture()}),
            &worker,
            &mut stats
        )
        .is_err());
    assert!(s
        .handle(
            json!({"method":"mining.set_difficulty","params":[5]}),
            &worker,
            &mut stats
        )
        .is_err());
    s.register(2, RequestKind::Submit).unwrap();
    s.register(3, RequestKind::Submit).unwrap();
    s.register(4, RequestKind::Submit).unwrap();
    s.handle(
        json!({"id":2,"result":{"status":"OK"},"error":null}),
        &worker,
        &mut stats,
    )
    .unwrap();
    s.handle(
        json!({"id":3,"result":false,"error":{"message":"synthetic rejection"}}),
        &worker,
        &mut stats,
    )
    .unwrap();
    assert_eq!((stats.accepted, stats.rejected), (1, 1));
    assert!(s
        .handle(json!({"id":3,"result":true}), &worker, &mut stats)
        .is_err());
    s.disconnect(&worker, &mut stats);
    assert_eq!(stats.unacknowledged, 1);
    assert!(s.token.is_none() && s.active.is_none() && s.pending.is_empty());
    assert_ne!(worker.state.epoch.load(Ordering::SeqCst), current);
    s.register(5, RequestKind::Login).unwrap();
    assert!(s
        .handle(json!({"id":5,"result":false}), &worker, &mut stats)
        .is_err());
    assert!(s.token.is_none());
}
#[test]
fn ack_limits_and_id_overflow() {
    let mut s = Session::new();
    for id in 0..64 {
        s.register(id, RequestKind::Submit).unwrap();
    }
    assert!(s.register(64, RequestKind::Submit).is_err());
    let mut id = u64::MAX;
    assert!(allocate_id(&mut id).is_err());
    assert!(!protocol::response_ok(
        &json!({"result":{"status":"OK"},"error":"bad"})
    ));
}
#[tokio::test]
async fn worker_finds_multiple_unique_shares_and_stops_with_backpressure() {
    let (mut worker, mut rx) = Worker::with_salt(Arc::new(FastCpuEngine::new(1)), 4, [0; 16]);
    let assigned = worker.assign(easy_job()).unwrap();
    let start = protocol::nonce_partition(assigned.job.prefix, [0; 16], assigned.generation).0;
    for offset in 0..20u64 {
        let event = timeout(Duration::from_secs(10), rx.recv())
            .await
            .unwrap()
            .unwrap();
        match event {
            WorkerEvent::Share(share) => {
                let (a, c) = *share;
                assert_eq!(a.generation, assigned.generation);
                assert_eq!(c.nonce, start + U512::from(offset));
                assert!(a.job.verify(&c));
            }
            _ => panic!("unexpected worker failure"),
        }
    }
    assert!(worker.state.hashes.load(Ordering::Relaxed) >= 20);
    worker.cancel();
    worker.assign(easy_job()).unwrap();
    worker.stop();
    let handle = worker.handle.take().unwrap();
    tokio::task::spawn_blocking(move || handle.join().unwrap())
        .await
        .unwrap();
}
#[tokio::test]
async fn already_cancelled_run_is_offline() {
    let (_tx, rx) = watch::channel(true);
    let config = Config {
        login: "synthetic-local-fixture".into(),
        ..Config::default()
    };
    let stats = run(config, Arc::new(FastCpuEngine::new(1)), rx)
        .await
        .unwrap();
    assert_eq!(stats.submitted, 0);
    assert_eq!(stats.reconnects, 0);
}

#[test]
fn sequence_only_updates_do_not_restart_nonce_scan() {
    let worker = fake_worker();
    let mut s = Session::new();
    let mut stats = Stats::default();
    s.token = Some("synthetic".into());
    s.active = Some(worker.assign(Job::parse(&fixture()).unwrap()).unwrap());
    let generation = s.active.as_ref().unwrap().generation;
    let mut update = fixture();
    update["seq"] = json!(136420);
    s.handle(json!({"method":"job","params":update}), &worker, &mut stats)
        .unwrap();
    assert_eq!(s.active.as_ref().unwrap().generation, generation);
    assert_eq!(s.active.as_ref().unwrap().job.sequence, Some(136420));
    assert!(s
        .handle(
            json!({"method":"job","params":fixture()}),
            &worker,
            &mut stats
        )
        .is_err());
}

#[tokio::test]
async fn shutdown_interrupts_blocked_write() {
    // In-memory duplex only: no socket or pool access.
    let (mut writer, _reader) = tokio::io::duplex(1);
    let (tx, mut shutdown) = watch::channel(false);
    let sender = tokio::spawn(async move {
        tokio::task::yield_now().await;
        tx.send(true).unwrap();
    });
    assert!(!timeout(
        Duration::from_secs(1),
        write_request(
            &mut writer,
            json!({"id":1}),
            Duration::from_secs(60),
            &mut shutdown
        )
    )
    .await
    .unwrap()
    .unwrap());
    sender.await.unwrap();
}

#[tokio::test]
async fn idle_worker_stop_cannot_lose_condvar_wakeup() {
    // Repetition exercises stop both before and during predicate/wait entry.
    for _ in 0..100 {
        let (mut worker, _rx) = Worker::new(Arc::new(FastCpuEngine::new(1)), 4);
        tokio::task::yield_now().await;
        worker.stop();
        let handle = worker.handle.take().unwrap();
        timeout(
            Duration::from_secs(1),
            tokio::task::spawn_blocking(move || handle.join().unwrap()),
        )
        .await
        .unwrap()
        .unwrap();
    }
}

#[test]
fn unknown_partial_work_update_fails_closed() {
    let worker = fake_worker();
    let mut s = Session::new();
    let mut stats = Stats::default();
    s.token = Some("synthetic".into());
    s.active = Some(worker.assign(easy_job()).unwrap());
    assert!(s
        .handle(
            json!({"method":"update","params":{"data":{"target":"00"}}}),
            &worker,
            &mut stats
        )
        .is_err());
    assert!(s
        .handle(
            json!({"method":"notice","params":{"message":"synthetic"}}),
            &worker,
            &mut stats
        )
        .is_ok());
}

#[test]
fn fresh_login_does_not_reuse_disconnected_session_or_pending_shares() {
    let worker = fake_worker();
    let mut s = Session::new();
    let mut stats = Stats::default();
    s.token = Some("old-synthetic".into());
    s.active = Some(worker.assign(easy_job()).unwrap());
    let old_generation = s.active.as_ref().unwrap().generation;
    s.register(1, RequestKind::Submit).unwrap();
    s.disconnect(&worker, &mut stats);
    s.register(2, RequestKind::Login).unwrap();
    s.handle(
        json!({"id":2,"result":{"status":"OK","id":"new-synthetic","job":fixture()},"error":null}),
        &worker,
        &mut stats,
    )
    .unwrap();
    assert_eq!(s.token.as_deref(), Some("new-synthetic"));
    assert!(s.active.as_ref().unwrap().generation > old_generation);
    assert!(s.pending.is_empty());
    assert_eq!(stats.unacknowledged, 1);
    assert!(s
        .handle(json!({"id":1,"result":true}), &worker, &mut stats)
        .is_err());
    assert_eq!(stats.accepted, 0);
}

#[test]
fn generation_partitions_are_disjoint_across_reconnect_and_target_changes() {
    let job = easy_job();
    let salt = [23; 16];
    let (start, end) = protocol::nonce_partition(job.prefix, salt, 1);
    let (next_start, next_end) = protocol::nonce_partition(job.prefix, salt, 2);
    let (pool_start, pool_end) = job.nonce_bounds();
    assert!(start >= pool_start && next_end <= pool_end && end < next_start);
    assert_eq!(protocol::next_nonce(end, end), None);
    assert_eq!(protocol::nonce_partition(job.prefix, salt, 1), (start, end));
    assert_ne!(protocol::nonce_partition(job.prefix, [24; 16], 1).0, start);
}

#[test]
fn generation_exhaustion_never_wraps_into_old_nonce_namespace() {
    let worker = fake_worker();
    worker.state.epoch.store(u64::MAX - 1, Ordering::SeqCst);
    assert!(worker.assign(easy_job()).is_err());
    assert_eq!(worker.state.epoch.load(Ordering::SeqCst), u64::MAX - 1);
    worker.cancel();
    assert_eq!(worker.state.epoch.load(Ordering::SeqCst), u64::MAX);
    worker.cancel();
    assert_eq!(worker.state.epoch.load(Ordering::SeqCst), u64::MAX);
    assert!(worker.assign(easy_job()).is_err());
}

#[tokio::test]
async fn offline_mock_pool_verifies_login_multiple_shares_and_acks_end_to_end() {
    use tokio::io::AsyncBufReadExt;
    let (miner_stream, pool_stream) = tokio::io::duplex(64 * 1024);
    let (tx, mut shutdown) = watch::channel(false);
    let config = Config {
        login: "synthetic-local-fixture.worker".into(),
        ..Config::default()
    };
    let (mut worker, mut shares) = Worker::with_salt(Arc::new(FastCpuEngine::new(1)), 4, [0; 16]);
    let pool = tokio::spawn(async move {
        let (read, mut write) = tokio::io::split(pool_stream);
        let mut lines = BufReader::new(read).lines();
        let login: Value =
            serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(login["method"], "login");
        assert_eq!(login["params"]["login"], "synthetic-local-fixture.worker");
        let job = easy_job();
        let mut job_value = fixture();
        job_value["difficulty"] = json!(1);
        job_value["target"] = json!(hex::encode(U512::MAX.to_big_endian()));
        let mut wire = serde_json::to_vec(&json!({"id":login["id"],"result":{"status":"OK","id":"synthetic-session","job":job_value},"error":null})).unwrap();
        wire.push(b'\n');
        write.write_all(&wire).await.unwrap();
        let mut last_nonce = None;
        for _ in 0..2 {
            let submit: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            assert_eq!(submit["method"], "submit");
            assert_eq!(submit["params"]["id"], "synthetic-session");
            assert_eq!(submit["params"]["job_id"], job.job_id);
            let nonce_bytes = hex::decode(submit["params"]["nonce"].as_str().unwrap()).unwrap();
            let nonce = U512::from_big_endian(&nonce_bytes);
            let c = candidate(&job, nonce);
            assert!(job.verify(&c));
            assert_eq!(
                submit["params"]["result"],
                hex::encode(c.hash.to_big_endian())
            );
            if let Some(previous) = last_nonce {
                assert!(nonce > previous);
            }
            last_nonce = Some(nonce);
            let mut ack = serde_json::to_vec(
                &json!({"id":submit["id"],"result":{"status":"OK"},"error":null}),
            )
            .unwrap();
            ack.push(b'\n');
            write.write_all(&ack).await.unwrap();
        }
        // EOF follows both acknowledgements in byte order, so the test does
        // not depend on sleeps or race shutdown against ack processing.
        write.shutdown().await.unwrap();
        // Keep the read half alive until the session consumes all buffered ack
        // bytes, avoiding BrokenPipe masking the deterministic receive path.
        while lines.next_line().await.unwrap_or(None).is_some() {}
    });
    let mut stats = Stats::default();
    let mut id = 1;
    let result = timeout(
        Duration::from_secs(10),
        session_loop(
            &config,
            &worker,
            &mut shares,
            &mut shutdown,
            &mut stats,
            &mut id,
            miner_stream,
        ),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    assert_eq!(stats.accepted, 2);
    assert_eq!(stats.rejected, 0);
    assert!(stats.submitted >= 2);
    assert!(stats.attempted >= stats.submitted);
    tx.send(true).unwrap();
    worker.stop();
    let handle = worker.handle.take().unwrap();
    tokio::task::spawn_blocking(move || handle.join().unwrap())
        .await
        .unwrap();
    timeout(Duration::from_secs(10), pool)
        .await
        .unwrap()
        .unwrap();
}

#[test]
fn keepalive_success_is_request_specific_and_requires_valid_status_without_error() {
    for status in ["KEEPALIVED", "OK"] {
        for error in [None, Some(Value::Null)] {
            let mut reply = json!({"id":7,"result":{"status":status}});
            if let Some(error) = error {
                reply["error"] = error;
            }
            assert!(protocol::keepalive_response_ok(&reply));
        }
    }
    let invalid = [
        json!({"result":{"status":"KEEPALIVED"},"error":{"code":-1,"message":"synthetic rejection"}}),
        json!({"result":{"status":"KEEPALIVED"},"error":false}),
        json!({"result":{"status":"KEEPALIVED"},"error":{}}),
        json!({"result":{"status":"keepalived"},"error":null}),
        json!({"result":{"status":"REJECTED"},"error":null}),
        json!({"result":{"status":true},"error":null}),
        json!({"result":true,"error":null}),
        json!({"result":false,"error":null}),
        json!({"result":"KEEPALIVED","error":null}),
        json!({"result":null,"error":null}),
        json!({"error":null}),
    ];
    for reply in invalid {
        assert!(
            !protocol::keepalive_response_ok(&reply),
            "must reject {reply}"
        );
    }
    assert!(!protocol::response_ok(
        &json!({"result":{"status":"KEEPALIVED"},"error":null})
    ));
    assert!(protocol::response_ok(
        &json!({"result":{"status":"OK"},"error":null})
    ));
}

#[test]
fn keepalive_ack_preserves_session_and_never_increments_share_acceptance() {
    let worker = fake_worker();
    let mut s = Session::new();
    let mut stats = Stats::default();
    s.token = Some("synthetic-session".into());
    s.active = Some(worker.assign(easy_job()).unwrap());
    let generation = s.active.as_ref().unwrap().generation;
    s.register(10, RequestKind::Keepalive).unwrap();
    s.handle(
        json!({"id":10,"jsonrpc":"2.0","result":{"status":"KEEPALIVED"},"error":null}),
        &worker,
        &mut stats,
    )
    .unwrap();
    assert!(s.pending.is_empty());
    assert_eq!(s.active.as_ref().unwrap().generation, generation);
    assert_eq!(stats.accepted, 0);
    assert_eq!(stats.rejected, 0);
    s.register(11, RequestKind::Submit).unwrap();
    s.handle(
        json!({"id":11,"result":{"status":"KEEPALIVED"},"error":null}),
        &worker,
        &mut stats,
    )
    .unwrap();
    assert_eq!(stats.accepted, 0);
    assert_eq!(stats.rejected, 1);
    s.register(12, RequestKind::Keepalive).unwrap();
    assert!(s
        .handle(
            json!({"id":12,"result":{"status":"KEEPALIVED"},"error":{"code":-1}}),
            &worker,
            &mut stats
        )
        .is_err());
    assert!(s
        .handle(
            json!({"id":999,"result":{"status":"KEEPALIVED"},"error":null}),
            &worker,
            &mut stats
        )
        .is_err());
}
