import pytest
from livekit.agents.llm import ChatMessage
from livekit.agents.utils._message_change import (
    _check_order_preserved,
    _compute_list_changes,
    _find_longest_increasing_subsequence,
    compute_changes,
)


@pytest.mark.parametrize(
    "indices,expected_seq,desc",
    [
        # Basic cases
        ([0, 1, 2], [0, 1, 2], "Already sorted"),
        ([2, 1, 0], [2], "Must keep first (2)"),
        ([2, 0, 1], [2], "Must keep first (2)"),
        ([2, 1, 0, 3], [2, 3], "Keep first and what can follow"),
        ([3, 0, 1, 2], [3], "Only first when nothing can follow"),
        ([2, 1, 0, 3, 4], [2, 3, 4], "Keep first and increasing suffix"),
        ([4, 1, 2, 3], [4], "Only first when better sequence exists"),
        ([0, 1, 4, 2], [0, 1, 4], "Keep longest increasing with first"),
        # Edge cases
        ([], [], "Empty list"),
        ([0], [0], "Single element"),
        ([1], [1], "Single element not zero"),
        ([2, 1], [2], "Two elements, keep first"),
    ],
)
def test_find_longest_increasing_subsequence(indices, expected_seq, desc):
    """Test the LIS algorithm with various cases"""
    result = _find_longest_increasing_subsequence(indices)
    result_seq = [indices[i] for i in result] if result else []

    # Verify sequence is increasing
    if result_seq:
        assert all(
            result_seq[i] < result_seq[i + 1] for i in range(len(result_seq) - 1)
        ), f"Not increasing in {desc}"

    # Verify first element is included
    if result:
        assert result[0] == 0, f"First index not included in {desc}"

    # Verify sequence matches expected
    assert (
        result_seq == expected_seq
    ), f"Wrong sequence in {desc}: expected {expected_seq}, got {result_seq}"


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([], True),
        ([0], True),
        ([0, 1, 2], True),
        ([0, 2, 1], False),
        ([1, 1, 2], False),
    ],
)
def test_check_order_preserved(indices, expected):
    assert _check_order_preserved(indices) is expected


@pytest.mark.parametrize(
    "old,new,expected_delete,expected_add",
    [
        # Empty lists
        ([], [], [], []),
        (["a"], [], ["a"], []),
        ([], ["a"], [], [(None, "a")]),
        # Simple append/delete
        (["a", "b", "c"], ["a", "b", "c", "d"], [], [("c", "d")]),
        (["a", "b", "c", "d"], ["a", "b", "c"], ["d"], []),
        # Delete first item
        (["a", "b", "c", "d"], ["b", "c", "d", "e"], ["a"], [("d", "e")]),
        (["x", "y", "b", "c"], ["b", "c", "d"], ["x", "y"], [("c", "d")]),
        # New first item - must delete all
        (
            ["a", "b", "c", "d"],
            ["e", "a", "b", "c"],
            ["a", "b", "c", "d"],
            [(None, "e"), ("e", "a"), ("a", "b"), ("b", "c")],
        ),
        # First item exists but order changes
        (["a", "b", "c", "d"], ["b", "a", "c", "d"], ["a"], [("b", "a")]),
        (["x", "y", "b", "c"], ["b", "d", "c"], ["x", "y"], [("b", "d")]),
        # Complex reordering
        (
            ["a", "b", "c", "d"],
            ["a", "b", "d", "e", "c"],
            ["d"],
            [("b", "d"), ("d", "e")],
        ),
        (
            ["a", "b", "c", "d"],
            ["a", "d", "c", "b"],
            ["c", "d"],
            [("a", "d"), ("d", "c")],
        ),
    ],
)
def test_compute_list_changes(old, new, expected_delete, expected_add):
    changes = _compute_list_changes(old, new)
    assert changes.to_delete == expected_delete
    assert changes.to_add == expected_add


@pytest.mark.parametrize(
    "old_ids,new_ids",
    [
        (["a", "b", "c", "d"], ["b", "c", "d", "e"]),
        (["a", "b", "c", "d"], ["e", "a", "b", "c"]),
        (["a", "b", "c", "d"], ["a", "b", "d", "e", "c"]),
    ],
)
def test_compute_changes(old_ids, new_ids):
    """Test computing changes with ChatMessage objects"""

    def create_msg(id: str) -> ChatMessage:
        return ChatMessage(role="test", id=id)

    old = [create_msg(id) for id in old_ids]
    new = [create_msg(id) for id in new_ids]

    changes = compute_changes(old, new, lambda msg: msg.id)

    # Apply changes and verify result
    result = [msg for msg in old if msg not in changes.to_delete]

    for prev, msg in changes.to_add:
        if prev is None:
            result.append(msg)
        else:
            idx = result.index(prev) + 1
            result.insert(idx, msg)

    assert [msg.id for msg in result] == new_ids


@pytest.mark.parametrize(
    "old,new",
    [
        (["a", "b", "c", "d"], ["b", "c", "d", "e"]),
        (["a", "b", "c", "d"], ["e", "a", "b", "c"]),
        (["a", "b", "c", "d"], ["a", "b", "d", "e", "c"]),
        (["a", "b", "c", "d"], ["b", "a", "c", "d"]),
        (["x", "y", "b", "c"], ["b", "d", "c"]),
        (["a", "b", "c", "d"], ["a", "d", "c", "b"]),
    ],
)
def test_changes_maintain_list_integrity(old, new):
    """Test that applying changes maintains list integrity"""

    def apply_changes(old: list[str], changes):
        # Apply deletions
        result = [x for x in old if x not in changes.to_delete]

        # Apply insertions
        for prev, item in changes.to_add:
            if prev is None:
                result.append(item)
            else:
                idx = result.index(prev) + 1
                result.insert(idx, item)
        return result

    changes = _compute_list_changes(old, new)
    result = apply_changes(old, changes)
    assert result == new, f"Failed to transform {old} into {new}, got {result}"
