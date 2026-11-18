//! Three-way reconciliation of atomic persisted rows, preserving the ledger's local order.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::ChatItem;
use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;
use lk_agents_abi::storage::voice::VoiceHistory;

use super::conversational;

#[derive(Debug)]
pub(crate) struct Reconciled {
    pub items: Vec<ChatItem>,
    pub rebased: Vec<String>,
}

pub(crate) fn reconcile(
    stored: &[ChatItem],
    local: VoiceHistory,
    own_writes: &BTreeMap<String, u64>,
) -> Result<Reconciled, Vec<String>> {
    let rows: BTreeMap<_, _> = stored.iter().map(|item| (item.id(), item)).collect();
    let baseline: BTreeMap<_, _> =
        local.stored.iter().map(|item| (item.id(), encode_chat_item(item))).collect();
    let ours = |id: &str| own_writes.get(id).is_some_and(|written| *written > local.revision.1);
    let (mut instructions, pending): (Vec<_>, Vec<_>) =
        local.items.into_iter().partition(|item| !conversational(item));
    let mut conflicts = Vec::new();
    let mut merged = Vec::new();
    for item in pending {
        let proposed = encode_chat_item(&item);
        let before = baseline.get(item.id());
        match rows.get(item.id()) {
            Some(row) => {
                let current = encode_chat_item(row);
                if proposed == current {
                    merged.push(item);
                } else if before == Some(&proposed) {
                    // A stale unchanged member is a read, not a new instruction to overwrite.
                    merged.push((*row).clone());
                } else if before == Some(&current) || ours(item.id()) {
                    merged.push(item);
                } else {
                    conflicts.push(item.id().to_owned());
                }
            }
            None if before == Some(&proposed) => {
                // The foreign writer retired this unchanged row; do not resurrect it.
            }
            None if before.is_none() => merged.push(item),
            None => conflicts.push(item.id().to_owned()),
        }
    }
    if !conflicts.is_empty() {
        return Err(conflicts);
    }
    let mut rebased: BTreeSet<String> = stored
        .iter()
        .filter(|row| baseline.get(row.id()) != Some(&encode_chat_item(row)) && !ours(row.id()))
        .map(|row| row.id().to_owned())
        .collect();
    rebased
        .extend(baseline.keys().filter(|id| !rows.contains_key(**id)).map(|id| (*id).to_owned()));
    if !merged.iter().any(|item| rows.contains_key(item.id())) {
        let additions = std::mem::replace(&mut merged, stored.to_vec());
        for item in additions {
            lk_agents_abi::llm::chat_context::ordering::commit_by_instant(&mut merged, item);
        }
    }
    // A narrow model view can omit stored history. It cannot silently delete those rows.
    for (index, row) in stored.iter().enumerate() {
        if !merged.iter().any(|item| item.id() == row.id()) {
            let next = stored[index + 1..]
                .iter()
                .find_map(|neighbor| merged.iter().position(|item| item.id() == neighbor.id()));
            let previous = stored[..index].iter().rev().find_map(|neighbor| {
                merged.iter().position(|item| item.id() == neighbor.id()).map(|at| at + 1)
            });
            if let Some(at) = next.or(previous) {
                merged.insert(at, row.clone());
            } else {
                lk_agents_abi::llm::chat_context::ordering::commit_by_instant(
                    &mut merged,
                    row.clone(),
                );
            }
        }
    }
    instructions.extend(merged);
    Ok(Reconciled { items: instructions, rebased: rebased.into_iter().collect() })
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
