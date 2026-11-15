//! Stable conversation ordering shared by session ledgers and persisted-history reconciliation.

use super::ChatItem;

/// `item` committed to `items` as the record commits it: in place of its earlier telling, else
/// where the record places a new item.
pub fn commit_by_instant(items: &mut Vec<ChatItem>, mut item: ChatItem) {
    match items.iter_mut().find(|held| held.id() == item.id()) {
        Some(held) => *held = item,
        None => {
            let at = insertion_index(items.iter(), &mut item);
            items.insert(at, item);
        }
    }
}

/// Where `item` joins `items`: after the last item stamped at or before it (ties go after), never
/// between a call and its output. A tool output sits with its call: right after its call's group
/// (the call and the calls and outputs right after it), at the group's latest instant, ahead of
/// what was said, heard or changed while its tool ran. A provider reads a tool's result right
/// after its call, and a host orders the history by instant.
pub fn insertion_index<'a>(
    items: impl Iterator<Item = &'a ChatItem>,
    item: &mut ChatItem,
) -> usize {
    let items: Vec<&ChatItem> = items.collect();
    let group_end = |start: usize| {
        let group = items[start..].iter().take_while(|held| {
            matches!(held, ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_))
        });
        start + group.count()
    };
    if let ChatItem::FunctionCallOutput(output) = item
        && let Some(call) = items.iter().position(
            |held| matches!(held, ChatItem::FunctionCall(call) if call.call_id == output.call_id),
        )
    {
        let end = group_end(call);
        if let Some(last) = items[call..end].iter().map(|held| held.created_at()).max() {
            output.created_at = last;
        }
        return end;
    }
    let at = item.created_at();
    let by_instant = items.iter().rposition(|held| held.created_at() <= at).map_or(0, |i| i + 1);
    let split =
        by_instant.checked_sub(1).and_then(|before| items.get(before)).is_some_and(|before| {
            matches!(before, ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_))
        }) && matches!(items.get(by_instant), Some(ChatItem::FunctionCallOutput(_)));
    if split { group_end(by_instant) } else { by_instant }
}
