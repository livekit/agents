import random
import tracemalloc

import pytest

from livekit.agents.llm import ChatContext, utils

pytestmark = pytest.mark.unit


def _dp_lcs_length(old_ids: list[str], new_ids: list[str]) -> int:
    n, m = len(old_ids), len(new_ids)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if old_ids[i - 1] == new_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _is_subsequence(sub: list[str], seq: list[str]) -> bool:
    it = iter(seq)
    return all(item in it for item in sub)


@pytest.mark.parametrize("seed", range(300))
def test_compute_lcs_returns_a_longest_common_subsequence(seed: int) -> None:
    rng = random.Random(seed)
    pool = [f"id_{i}" for i in range(rng.randint(0, 40))]
    old_ids = rng.sample(pool, rng.randint(0, len(pool)))
    new_ids = rng.sample(pool, rng.randint(0, len(pool)))
    if seed % 3 == 0 and old_ids:
        # repeated IDs take the dynamic-programming path
        new_ids += rng.choices(old_ids, k=3)

    lcs_ids = utils._compute_lcs(old_ids, new_ids)

    assert len(lcs_ids) == _dp_lcs_length(old_ids, new_ids)
    assert _is_subsequence(lcs_ids, old_ids)
    assert _is_subsequence(lcs_ids, new_ids)


def test_compute_chat_ctx_diff_on_a_long_context_does_not_build_an_n_by_m_table() -> None:
    n = 2000
    old = ChatContext()
    for i in range(n):
        old.add_message(role="user", content=f"message {i}", id=f"item_{i}")
    new = old.copy()
    new.add_message(role="system", content="appended", id="appended")

    tracemalloc.start()
    try:
        diff = utils.compute_chat_ctx_diff(old, new)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert diff.to_create == [(f"item_{n - 1}", "appended")]
    assert diff.to_remove == []
    assert diff.to_update == []
    # the O(n*m) table for 2000 x 2001 items takes about 32 MB
    assert peak < 1_000_000
