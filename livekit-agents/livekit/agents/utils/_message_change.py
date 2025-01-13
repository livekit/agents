from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union


T = TypeVar("T")


def compute_lcs(old_ids: list[str], new_ids: list[str]) -> list[str]:
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


def sync_chat_ctx(old_ctx: ChatContext, new_ctx: ChatContext) -> List[dict]:
    old_ids = [m.id for m in old_ctx.messages]
    new_ids = [m.id for m in new_ctx.messages]

    # 1. Find the set of message IDs we will keep
    lcs_ids = set(compute_lcs(old_ids, new_ids))

    requests = []

    # 2. Remove messages from old that are NOT in LCS
    #    (this ensures that anything not in the final conversation is removed)
    for old_msg in old_ctx.messages:
        if old_msg.id not in lcs_ids:
            requests.append(
                {
                    "action": "remove",
                    "id": old_msg.id,
                }
            )

    # 3. Create the missing messages from new in the correct order
    #    We keep track of the "last message ID" that ends up in the final new sequence.
    last_id_in_new_sequence: Optional[str] = None

    for i, new_msg in enumerate(new_ctx.messages):
        if new_msg.id in lcs_ids:
            # This message is already kept (it's in LCS), so just update
            # our 'last_id_in_new_sequence' pointer to it,
            # meaning "this message is logically next in the final conversation"
            last_id_in_new_sequence = new_msg.id
        else:
            # This message is not in LCS: we need to create it
            if last_id_in_new_sequence is None:
                # Insert at the very beginning
                prev_item = "root"
            else:
                # Insert after the last item that we have in the final conversation
                prev_item = last_id_in_new_sequence

            requests.append(
                {
                    "action": "create",
                    "id": new_msg.id,
                    "role": new_msg.role,
                    "previous_item_id": prev_item,  # could be "root" or last_id
                }
            )

            # Now this newly created message becomes the last in the final conversation
            last_id_in_new_sequence = new_msg.id

    return requests
