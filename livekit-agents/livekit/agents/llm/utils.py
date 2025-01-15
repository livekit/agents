from __future__ import annotations

from dataclasses import dataclass

from .chat_context import ChatContext


def _compute_lcs(old_ids: list[str], new_ids: list[str]) -> list[str]:
    """
    Standard dynamic-programming LCS to get the common subsequence
    of IDs (in order) that appear in both old_ids and new_ids.
    """
    n, m = len(old_ids), len(new_ids)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if old_ids[i - 1] == new_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual LCS sequence
    lcs_ids = []
    i, j = n, m
    while i > 0 and j > 0:
        if old_ids[i - 1] == new_ids[j - 1]:
            lcs_ids.append(old_ids[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs_ids))


@dataclass
class DiffOps:
    to_remove: list[str]
    to_create: list[
        tuple[str | None, str]
    ]  # (previous_item_id, id), if previous_item_id is None, add to the root


def compute_chat_ctx_diff(old_ctx: ChatContext, new_ctx: ChatContext) -> DiffOps:
    """Computes the minimal list of create/remove operations to transform old_ctx into new_ctx."""
    # TODO(theomonnom): Make ChatMessage hashable and also add update ops

    old_ids = [m.id for m in old_ctx.messages]
    new_ids = [m.id for m in new_ctx.messages]
    lcs_ids = set(_compute_lcs(old_ids, new_ids))

    to_remove = [msg.id for msg in old_ctx.messages if msg.id not in lcs_ids]
    to_create: list[tuple[str | None, str]] = []

    last_id_in_sequence: str | None = None
    for new_msg in new_ctx.messages:
        if new_msg.id in lcs_ids:
            last_id_in_sequence = new_msg.id
        else:
            if last_id_in_sequence is None:
                prev_id = None  # root
            else:
                prev_id = last_id_in_sequence

            to_create.append((prev_id, new_msg.id))
            last_id_in_sequence = new_msg.id

    return DiffOps(to_remove=to_remove, to_create=to_create)
