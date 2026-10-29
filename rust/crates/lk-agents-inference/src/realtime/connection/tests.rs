use super::*;
use lk_agents_abi::errors::ApiErrorKind;

fn at(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}
fn options() -> ApiConnectOptions {
    ApiConnectOptions {
        connect_timeout: Some(1.0),
        total_timeout: None,
        max_retry: 1,
        ..Default::default()
    }
}

#[test]
fn realtime_connection_generation_and_connect_timeout_keep_their_identity() {
    let mut connection = RealtimeConnection::new(&options(), at(0));
    assert_eq!(connection.start(at(0)), [ConnectionEffect::Connect { generation: 1 }]);
    let effects = connection.tick(at(1000));
    assert!(effects.contains(&ConnectionEffect::Close { generation: 1 }));
    assert!(effects.iter().any(|effect| matches!(effect, ConnectionEffect::Retry { error }
        if error.kind == ApiErrorKind::Timeout && error.phase == Some(TimeoutPhase::Connect))));
    assert_eq!(connection.next_deadline(), Some(at(1100)));
    connection.released(1, at(1000));
    assert_eq!(connection.tick(at(1100)), [ConnectionEffect::Connect { generation: 2 }]);
    assert!(!connection.connected(1, at(1100)));
    assert!(connection.connected(2, at(1100)));
    assert!(connection.failed(1, at(1200), ApiError::connection(None)).is_empty());
}

#[test]
fn realtime_cancel_retains_cleanup_deadlines_and_the_original_generation() {
    let mut connection = RealtimeConnection::new(&options(), at(0));
    connection.start(at(0));
    connection.tick(at(1000));
    connection.tick(at(1100));
    connection.close(at(1200));
    assert_eq!(connection.next_deadline(), Some(at(6000)));
    let effects = connection.tick(at(6000));
    assert!(effects.contains(&ConnectionEffect::CleanupIncomplete { generation: 1 }));
    assert!(!effects.contains(&ConnectionEffect::CleanupIncomplete { generation: 2 }));
    connection.released(2, at(6100));
    assert!(connection.tick(at(6200)).is_empty());
}

#[test]
fn realtime_total_deadline_survives_a_healthy_reconnection() {
    let options = ApiConnectOptions { total_timeout: Some(2.0), ..options() };
    let mut connection = RealtimeConnection::new(&options, at(0));
    connection.start(at(0));
    connection.connected(1, at(100));
    let effects = connection.failed(1, at(500), ApiError::connection(None));
    assert!(effects.contains(&ConnectionEffect::Connect { generation: 2 }));
    connection.connected(2, at(600));
    assert_eq!(connection.next_deadline(), Some(at(2000)));
    assert!(connection.tick(at(2000)).iter().any(|effect| matches!(effect,
        ConnectionEffect::Failed { error } if error.phase == Some(TimeoutPhase::Total))));
    assert!(connection.start(at(2100)).is_empty());
}

#[test]
fn realtime_old_acquisition_cleanup_progresses_after_a_new_socket_connects() {
    let mut connection = RealtimeConnection::new(&options(), at(0));
    connection.start(at(0));
    connection.tick(at(1000));
    connection.tick(at(1100));
    assert!(connection.connected(2, at(1200)));
    assert_eq!(connection.next_deadline(), Some(at(6000)));
    assert!(
        connection.tick(at(6000)).contains(&ConnectionEffect::CleanupIncomplete { generation: 1 })
    );
    assert!(connection.next_deadline().is_none());
}

#[test]
fn realtime_old_acquisition_cleanup_survives_another_reconnection_window() {
    let mut connection = RealtimeConnection::new(&options(), at(0));
    connection.start(at(0));
    connection.tick(at(1000));
    connection.tick(at(1100));
    connection.connected(2, at(1200));
    connection.failed(2, at(1300), ApiError::connection(None));
    connection.connected(3, at(1400));
    assert!(
        connection.tick(at(6000)).contains(&ConnectionEffect::CleanupIncomplete { generation: 1 })
    );
}
