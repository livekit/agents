use super::*;
use crate::testkit::NoImages;

fn opts() -> LlmOptions {
    LlmOptions {
        model: "openai/gpt-4o".into(),
        provider: None,
        base_url: "https://gw/v1".into(),
        auth: Auth::Gateway { api_key: "k".into(), api_secret: "s".into() },
        inference_class: None,
        strict_tools: true,
        body: Default::default(),
        label: "llm".into(),
        extra_kwargs: Default::default(),
    }
}

fn header<'a>(request: &'a HttpRequest, name: &str) -> Option<&'a str> {
    request.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

// Catches: job headers sent when there is no job, the bearer missing, or a wrong path.
#[test]
fn with_no_job_the_request_carries_no_livekit_job_headers() {
    let request = build_request(
        &opts(),
        None,
        &NoImages,
        &ChatContext::default(),
        &[],
        ChatArgs::default(),
        "t",
    )
    .unwrap();
    let job: Vec<&str> = request
        .headers
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| ["X-LiveKit-Room-ID", "X-LiveKit-Job-ID", "X-LiveKit-Agent-ID"].contains(k))
        .collect();
    assert!(job.is_empty(), "got {job:?}");
    assert_eq!(header(&request, "Authorization"), Some("Bearer t"));
    assert!(request.url.ends_with("/v1/chat/completions"), "{}", request.url);
}

// Catches: the host's identification headers (the job's `X-LiveKit-*` identity, read where
// `chat()` runs) not reaching the request, or the native priority/User-Agent leaking in beside
// them. The pinned provider and the bearer are still ours.
#[test]
fn the_hosts_identification_headers_ride_on_the_request() {
    let opts = LlmOptions {
        provider: Some("openai".into()),
        inference_class: Some("low".into()),
        ..opts()
    };
    let ident: Headers = [
        ("User-Agent", "LiveKit Agents/1 (python 3)"),
        ("X-LiveKit-Room-ID", "RM_1"),
        ("X-LiveKit-Job-ID", "AJ_1"),
        ("X-LiveKit-Agent-ID", "PA_1"),
        ("X-LiveKit-Session-ID", "inference_abc"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let request = build_request(
        &opts,
        Some(&ident),
        &NoImages,
        &ChatContext::default(),
        &[],
        ChatArgs::default(),
        "t",
    )
    .unwrap();
    assert_eq!(header(&request, "X-LiveKit-Room-ID"), Some("RM_1"));
    assert_eq!(header(&request, "X-LiveKit-Job-ID"), Some("AJ_1"));
    assert_eq!(header(&request, "X-LiveKit-Agent-ID"), Some("PA_1"));
    assert_eq!(header(&request, "X-LiveKit-Session-ID"), None);
    assert_eq!(header(&request, "User-Agent"), Some("LiveKit Agents/1 (python 3)"));
    assert_eq!(
        header(&request, "X-LiveKit-Inference-Priority"),
        None,
        "the host's map already decided it"
    );
    assert_eq!(header(&request, "X-LiveKit-Inference-Provider"), Some("openai"));
}

// Catches: a LiveKit header or gateway token reaching a provider, `extra_body` sent as a
// field instead of merged, `extra_query` dropped, or a strict schema a provider refuses.
#[test]
fn a_provider_key_request_is_plain_chat_completions() {
    let opts = LlmOptions {
        base_url: "https://api.example.com/v1".into(),
        auth: Auth::ProviderKey("sk".into()),
        provider: Some("openai".into()),
        strict_tools: false,
        extra_kwargs: [
            ("extra_body".to_owned(), json!({"provider": {"order": ["a"]}})),
            ("extra_query".to_owned(), json!({"api-version": "1"})),
            ("extra_headers".to_owned(), json!({"X-Title": "app"})),
        ]
        .into_iter()
        .collect(),
        ..opts()
    };
    let ident: Headers = [("X-LiveKit-Job-ID".to_owned(), "AJ_1".to_owned())].into_iter().collect();
    let tool = ToolSpec {
        provider_tool: None,
        name: "lookup".into(),
        description: String::new(),
        json_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        flags: Default::default(),
    };
    let request = build_request(
        &opts,
        Some(&ident),
        &NoImages,
        &ChatContext::default(),
        &[tool],
        ChatArgs::default(),
        &opts.auth.bearer().unwrap(),
    )
    .unwrap();
    assert_eq!(request.url, "https://api.example.com/v1/chat/completions?api-version=1");
    assert_eq!(header(&request, "Authorization"), Some("Bearer sk"));
    assert_eq!(header(&request, "X-Title"), Some("app"));
    assert!(request.headers.iter().all(|(k, _)| !k.starts_with("X-LiveKit")));
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["provider"], json!({"order": ["a"]}));
    assert!(body.get("extra_body").is_none() && body.get("extra_query").is_none());
    assert!(body["tools"][0]["function"].get("strict").is_none());
}

// Catches: a compressed or msgpack body sent under a JSON content type (or the reverse),
// or a body that does not decode back to the request.
#[test]
fn an_encoded_body_names_its_encoding_and_round_trips() {
    let opts = LlmOptions { body: BodyEncoding { msgpack: true, gzip: true }, ..opts() };
    let request = build_request(
        &opts,
        None,
        &NoImages,
        &ChatContext::default(),
        &[],
        ChatArgs::default(),
        "t",
    )
    .unwrap();
    assert_eq!(header(&request, "Content-Type"), Some("application/vnd.msgpack"));
    assert_eq!(header(&request, "Content-Encoding"), Some("gzip"));
    let mut packed = Vec::new();
    std::io::Read::read_to_end(
        &mut flate2::read::GzDecoder::new(request.body.as_slice()),
        &mut packed,
    )
    .unwrap();
    let body: Value = rmp_serde::from_slice(&packed).unwrap();
    assert_eq!(body["model"], json!("openai/gpt-4o"));
    assert_eq!(body["stream"], json!(true));
}

#[test]
fn gateway_chat_completions_ignores_all_server_side_provider_declarations() {
    for family in [None, Some("openai"), Some("xai"), Some("gemini"), Some("mistral")] {
        let declaration = ToolSpec {
            name: "provider_id".into(),
            description: String::new(),
            json_schema: Value::Null,
            flags: Default::default(),
            provider_tool: Some(lk_agents_abi::llm::types::ProviderToolDeclaration {
                provider: family.map(str::to_owned),
                payload: json!({"type": "web_search"}),
            }),
        };
        assert!(tool_schemas(&[declaration], true).unwrap().is_empty());
    }
}

// Catches: a tool schema reaching the gateway un-strict, or a bad schema failing silently.
#[test]
fn tool_schemas_are_built_in_the_strict_openai_form() {
    let tool = lk_agents_abi::ToolSpec {
        provider_tool: None,
        name: "lookup".into(),
        description: "look it up".into(),
        json_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        flags: Default::default(),
    };
    let built = tool_schemas(std::slice::from_ref(&tool), true).expect("a usable schema");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0]["type"], json!("function"));
    assert_eq!(built[0]["function"]["name"], json!("lookup"));
    assert_eq!(built[0]["function"]["strict"], json!(true));
    assert_eq!(built[0]["function"]["description"], json!("look it up"));
    assert_eq!(built[0]["function"]["parameters"]["additionalProperties"], json!(false));
    assert_eq!(built[0]["function"]["parameters"]["required"], json!(["q"]));
    assert!(tool_schemas(&[], true).unwrap().is_empty());
}
