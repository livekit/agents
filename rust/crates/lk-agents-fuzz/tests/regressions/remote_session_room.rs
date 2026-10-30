//! The remote session's room transport (`RoomTransport`), which no DST seed carries: the console
//! seeds run the protocol over a socket.
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::rtc::events::{RoomEventBus, byte_streams};
use lk_agents_runtime::voice::remote_session::room::RoomTransport;
use lk_agents_runtime::voice::remote_session::{SessionTransport, TOPIC_SESSION_MESSAGES, proto};
use lk_agents_test_support::rtc::{ConnectionState, FakeRtc, StreamInfo};
use prost::Message as _;
use proto::agent_session_message::Message;
use std::sync::Arc;

fn ping(id: &str) -> proto::AgentSessionMessage {
    proto::AgentSessionMessage {
        message: Some(Message::Request(proto::SessionRequest {
            request_id: id.to_owned(),
            request: Some(proto::session_request::Request::Ping(proto::session_request::Ping {})),
        })),
    }
}

fn request_id(message: proto::AgentSessionMessage) -> String {
    match message.message {
        Some(Message::Request(request)) => request.request_id,
        _ => panic!("not a request"),
    }
}

// Catches: messages read out of order or lost to a malformed one, a sent message not on the
// topic as one closed stream, a send on a disconnected room not raised, or a close leaving
// the topic registered or a receive waiting.
#[tokio::test]
async fn the_room_carries_one_message_per_stream() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("call");
    let transport =
        Arc::new(RoomTransport::new(room.clone(), clock, Arc::new(TokioSpawner::current())));
    transport.start().await.unwrap();
    transport.start().await.expect("idempotent");
    let info = || StreamInfo { topic: TOPIC_SESSION_MESSAGES.into(), ..StreamInfo::default() };
    let first = ping("req_1").encode_to_vec();
    let (head, tail) = first.split_at(3);
    room.send_byte_stream("cli", info(), [head.to_vec(), tail.to_vec()]).unwrap();
    room.send_byte_stream("cli", info(), [vec![0xff, 0xff]]).unwrap();
    room.send_byte_stream("cli", info(), [ping("req_2").encode_to_vec()]).unwrap();
    assert_eq!(request_id(transport.recv().await.unwrap().unwrap()), "req_1");
    assert_eq!(request_id(transport.recv().await.unwrap().unwrap()), "req_2");

    transport.send_message(&ping("req_3")).await.unwrap();
    let sent = room.local_participant().byte_streams();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].opts.topic, TOPIC_SESSION_MESSAGES);
    assert!(sent[0].opts.name.starts_with("AS_") && sent[0].closed);
    let decoded = proto::AgentSessionMessage::decode(sent[0].chunks.concat().as_slice()).unwrap();
    assert_eq!(request_id(decoded), "req_3");

    room.set_connection_state(ConnectionState::Disconnected);
    assert!(transport.send_message(&ping("req_4")).await.is_err());

    let waiting = {
        let transport = Arc::clone(&transport);
        tokio::spawn(async move { transport.recv().await })
    };
    tokio::task::yield_now().await;
    transport.close().await;
    assert!(waiting.await.unwrap().unwrap().is_none());
    byte_streams(room.as_ref(), TOPIC_SESSION_MESSAGES).expect("the topic is free again");
}
