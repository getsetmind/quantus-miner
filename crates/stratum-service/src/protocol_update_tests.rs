//! Offline synthetic protocol fixtures, not observations of Suprnova extensions.
use crate::protocol::{parse_notification, Job, Notification};
use primitive_types::U512;
use serde_json::{json, Value};

fn job_value() -> Value {
    json!({"algo":"qpow-poseidon2", "job_id":"synthetic", "mining_hash":"11".repeat(32),
        "extranonce":"12345678", "difficulty":2, "target":hex::encode((U512::MAX / U512::from(2)).to_big_endian()), "seq":40})
}

#[test]
fn documented_wrapped_and_compatible_direct_jobs_are_validated() {
    let job = job_value();
    for frame in [
        json!({"jsonrpc":"2.0","method":"job","params":{"clean_jobs":true,"job":job}}),
        json!({"method":"job","params":job}),
    ] {
        assert_eq!(
            parse_notification(&frame).unwrap(),
            Notification::Job(Box::new(Job::parse(&job).unwrap()))
        );
    }
}

#[test]
fn full_job_difficulty_updates_change_work_without_changing_identity() {
    let previous = Job::parse(&job_value()).unwrap();
    let mut next = job_value();
    next["difficulty"] = json!(4);
    next["target"] = json!(hex::encode((U512::MAX / U512::from(4)).to_big_endian()));
    next["seq"] = json!(41);
    for frame in [
        json!({"method":"job","params":{"job":next,"clean_jobs":true}}),
        json!({"method":"job","params":next}),
    ] {
        let Notification::Job(next) = parse_notification(&frame).unwrap() else {
            panic!("expected job")
        };
        let next = previous.reconcile_update(*next).unwrap();
        assert_eq!(previous.job_id, next.job_id);
        assert_eq!(previous.header, next.header);
        assert_eq!(previous.prefix, next.prefix);
        assert!(!previous.same_work(&next));
    }
}

#[test]
fn sequence_only_updates_and_missing_sequence_preserve_work() {
    let previous = Job::parse(&job_value()).unwrap();
    let mut next = previous.clone();
    next.sequence = Some(41);
    let next = previous.reconcile_update(next).unwrap();
    assert!(previous.same_work(&next));
    assert_eq!(next.sequence, Some(41));
    let mut absent = next.clone();
    absent.sequence = None;
    assert_eq!(next.reconcile_update(absent).unwrap().sequence, Some(41));
}

#[test]
fn missing_sequence_on_changed_work_cannot_erase_high_water() {
    let previous = Job::parse(&job_value()).unwrap();
    let mut changed = previous.clone();
    changed.job_id = "changed".into();
    changed.sequence = None;
    let changed = previous.reconcile_update(changed).unwrap();
    assert_eq!(changed.sequence, Some(40));
    let mut stale = changed.clone();
    stale.sequence = Some(39);
    assert!(changed.reconcile_update(stale).is_err());
}

#[test]
fn unknown_notifications_always_fail_closed_even_without_recognized_fields() {
    for method in [
        "mining.set_difficulty",
        "set_target",
        "set_extranonce",
        "update",
        "mining.notify",
        "new_work",
        "future_extension",
        "JOB",
    ] {
        for params in [
            json!({}),
            json!({"message":"text"}),
            json!({"deep":{"unknown_pow_limit":"ff"}}),
            json!([4]),
        ] {
            assert!(
                parse_notification(&json!({"method":method,"params":params})).is_err(),
                "accepted {method}: {params}"
            );
        }
    }
}

#[test]
fn malformed_job_envelopes_and_unknown_work_fields_fail_closed() {
    let job = job_value();
    for frame in [
        json!({"method":"job","params":{"job":job,"clean_jobs":false}}),
        json!({"method":"job","params":{"job":job,"clean_jobs":"true"}}),
        json!({"method":"job","params":{"job":job,"future_target":"00"}}),
        json!({"method":"job","params":{"job":null}}),
        json!({"method":"job","params":[]}),
        json!({"id":1,"method":"job","params":job}),
        json!({"jsonrpc":"1.0","method":"job","params":job}),
        json!({"method":"job","params":job,"result":{}}),
    ] {
        assert!(parse_notification(&frame).is_err(), "accepted {frame}");
    }
    let mut unknown = job_value();
    unknown["future_work"] = json!({"limit":"ff"});
    assert!(Job::parse(&unknown).is_err());
    let mut mismatch = job_value();
    mismatch["difficulty"] = json!(3);
    assert!(parse_notification(&json!({"method":"job","params":mismatch})).is_err());
}

#[test]
fn local_notice_compatibility_has_no_structured_work_payload() {
    assert_eq!(
        parse_notification(&json!({"method":"notice","params":{"message":"synthetic"}})).unwrap(),
        Notification::Notice
    );
    for params in [
        json!({"message":"synthetic","deep":{"target":"00"}}),
        json!({"message":{"target":"00"}}),
        json!({"message":"x".repeat(4097)}),
        json!({"message":""}),
        json!({"target":"00"}),
    ] {
        assert!(parse_notification(&json!({"method":"notice","params":params})).is_err());
    }
}
