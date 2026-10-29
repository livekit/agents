//! `ChatContext.to_dict` (`chat_context.py:629`): the item filters over each item's generated
//! `model_dump(mode="json", exclude_none=True)`.

use std::marker::PhantomData;

use lk_agents_abi::llm::chat_context::ItemKind;
use lk_agents_abi::llm::to_dict::ChatContextDict;
use lk_agents_abi::{ChatItem, DumpCx, ModelDump, ToDictError, ToDictHost, ToDictOptions};
use serde_json::{Map, Value};

/// `{"items": [...]}` and nothing else: `from_dict` and the session report read that envelope.
pub fn to_dict(
    items: &[ChatItem],
    opts: &ToDictOptions,
    host: &ToDictHost<'_>,
) -> Result<Map<String, Value>, ToDictError> {
    let mut dumped = Vec::with_capacity(items.len());
    for item in items {
        if let Some(v) = item_to_dict(item, opts, host)? {
            dumped.push(v);
        }
    }
    Ok(Map::from_iter([("items".to_owned(), Value::Array(dumped))]))
}

/// The item filters (`chat_context.py:642`), then the item's dump.
pub fn item_to_dict(
    item: &ChatItem,
    opts: &ToDictOptions,
    host: &ToDictHost<'_>,
) -> Result<Option<Value>, ToDictError> {
    if opts.exclude_function_call && item.kind().is_function_call_pair() {
        return Ok(None);
    }
    if opts.exclude_config_update && item.kind() == ItemKind::AgentConfigUpdate {
        return Ok(None);
    }
    item.model_dump(&DumpCx { opts, host }, "ChatItem")
}

/// How an assistant message reads without the speech core's expressive markup.
pub trait AssistantMarkup {
    fn strip_expr_markup(text: &str) -> String;
}

/// The host's `ChatContext.to_dict`, stripping assistant markup with `M`.
pub struct ChatDict<M>(PhantomData<M>);

impl<M: AssistantMarkup> ChatContextDict for ChatDict<M> {
    #[allow(clippy::too_many_arguments)]
    fn chat_ctx_to_dict(
        items: Vec<ChatItem>,
        exclude_image: bool,
        exclude_audio: bool,
        exclude_timestamp: bool,
        exclude_function_call: bool,
        exclude_metrics: bool,
        exclude_config_update: bool,
        strip_markup: bool,
    ) -> Result<Map<String, Value>, ToDictError> {
        let opts = ToDictOptions {
            exclude_image,
            exclude_audio,
            exclude_timestamp,
            exclude_function_call,
            exclude_metrics,
            exclude_config_update,
        };
        let strip = M::strip_expr_markup;
        let host = ToDictHost {
            media: None,
            strip_expr_markup: strip_markup.then_some(&strip as &dyn Fn(&str) -> String),
        };
        to_dict(&items, &opts, &host)
    }
}
