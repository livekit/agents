use lk_agents_a2a::{Message, rpc};
use serde_json::{Value, json};

fn request(part: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [part]}},
    }))
    .unwrap()
}

#[test]
fn malformed_declared_parts_are_refused_before_upstream_lossy_decoding() {
    for part in [
        json!({"text": "visible", "raw": "c2VjcmV0"}),
        json!({"text": "visible", "url": "opaque-app-ref"}),
        json!({"text": "visible", "filename": 5}),
        json!({"url": "opaque-app-ref", "mediaType": ["image/png"]}),
        json!({"text": "visible", "metadata": "ignored-by-upstream"}),
        json!({"data": ["not-a-Struct"]}),
        json!({"raw": "c2VjcmV0", "mediaType": "image/png"}),
    ] {
        let refusal = rpc::decode(&request(part.clone())).expect_err(&part.to_string());
        assert_eq!(refusal.error.code, a2a::error_code::INVALID_PARAMS);
    }
}

#[test]
fn an_opaque_application_reference_survives_without_underlying_data_access() {
    let part = json!({
        "url": "application-only reference that requires its own authorization",
        "mediaType": "application/pdf", "filename": "return.pdf",
        "metadata": {"application": "kept"},
    });
    let decoded = rpc::decode(&request(part.clone())).unwrap();
    let rpc::Call::Send { message, .. } = decoded.call else { panic!("send expected") };
    let incoming = Message::from_a2a(&message, 1.0, None).unwrap();
    assert_eq!(serde_json::to_value(incoming.to_a2a()).unwrap()["parts"][0], part);
}

#[test]
fn direct_native_receipt_cannot_bypass_the_inline_upload_boundary() {
    let message = serde_json::from_value::<a2a::Message>(json!({
        "messageId": "m1", "role": "ROLE_USER",
        "parts": [{"raw": "dXBsb2FkZWQgYnl0ZXM=", "mediaType": "image/png"}],
    }))
    .unwrap();
    assert!(Message::from_a2a(&message, 1.0, None).is_err());
}
