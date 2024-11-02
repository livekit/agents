from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class MessageChange(Generic[T]):
    """Represents changes needed to transform one list into another

    The changes must be applied in order:
    1. First apply all deletions
    2. Then apply all insertions with their previous_item_id
    """

    to_delete: list[T]
    """Items to delete from old list"""
    to_add: list[tuple[T | None, T]]
    """Items to add as (previous_item, new_item) pairs"""


def compute_changes(
    old_list: list[T], new_list: list[T], key_fnc: Callable[[T], str]
) -> MessageChange[T]:
    """Compute minimum changes needed to transform old list into new list"""
    # Convert to lists of ids
    old_ids = [key_fnc(msg) for msg in old_list]
    new_ids = [key_fnc(msg) for msg in new_list]

    # Create lookup maps
    old_msgs = {key_fnc(msg): msg for msg in old_list}
    new_msgs = {key_fnc(msg): msg for msg in new_list}

    # Compute changes using ids
    changes = _compute_list_changes(old_ids, new_ids)

    # Convert back to items
    return MessageChange(
        to_delete=[old_msgs[id] for id in changes.to_delete],
        to_add=[
            (
                None if prev is None else old_msgs.get(prev) or new_msgs[prev],
                new_msgs[new],
            )
            for prev, new in changes.to_add
        ],
    )


def _compute_list_changes(old_list: list[T], new_list: list[T]) -> MessageChange[T]:
    """Compute minimum changes needed to transform old_list into new_list

    Rules:
    - Delete first, then insert
    - Can't insert at start if list not empty (must delete all first)
    - Each insert needs previous item except for first item in new list
    - If an item changes position relative to others, it must be deleted and reinserted
    - If first item in new list exists in old list, must delete all items before it

    Examples:
    old [a b c d] new [b c d e] -> delete a, insert (d,e)
    old [a b c d] new [e a b c d] -> delete all, insert (None,e),(e,a),(a,b),(b,c),(c,d)
    old [a b c d] new [a b d e c] -> delete d, insert (b,d),(d,e)
    old [a b c d] new [a d c b] -> delete c,d, insert (a,d),(d,c)
    """
    if not new_list:
        return MessageChange(to_delete=old_list, to_add=[])

    # Find first item's position in old list
    try:
        first_idx = old_list.index(new_list[0])
    except ValueError:
        # Special case: if first item is new, delete everything
        prev_item = None
        to_add = []
        for x in new_list:
            to_add.append((prev_item, x))
            prev_item = x
        return MessageChange(to_delete=old_list, to_add=to_add)

    # Delete all items before first_idx
    to_delete = old_list[:first_idx]
    remaining_old = old_list[first_idx:]

    # Get positions of remaining items in new list
    indices = []
    items = []
    new_positions = {x: i for i, x in enumerate(new_list)}
    for x in remaining_old:
        if x in new_positions:
            indices.append(new_positions[x])
            items.append(x)

    # Try fast path first - check if remaining order is preserved
    if _check_order_preserved(indices):
        kept_indices = list(range(len(indices)))
    else:
        # Order changed, need to find kept items using LIS
        # First item must be kept since we've already handled items before it
        kept_indices = _find_longest_increasing_subsequence(indices)

    # Convert kept indices back to items
    kept_items = {items[i] for i in kept_indices}

    # Add items that need to be deleted from remaining list
    to_delete.extend(x for x in remaining_old if x not in kept_items)

    # Compute items to add by following new list order
    to_add = []
    prev_item = None
    for x in new_list:
        if x not in kept_items:
            to_add.append((prev_item, x))
        prev_item = x

    return MessageChange(to_delete=to_delete, to_add=to_add)


def _check_order_preserved(indices: list[int]) -> bool:
    """Check if indices form an increasing sequence"""
    if not indices:
        return True

    # Check if indices form an increasing sequence
    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            return False

    return True


def _find_longest_increasing_subsequence(indices: list[int]) -> list[int]:
    """Find indices of the longest increasing subsequence

    Args:
        indices: List of indices to find LIS from

    Returns:
        List of indices into the input list that form the LIS
        For example, indices = [0, 4, 1, 2] -> [0, 2, 3]
    """
    if not indices:
        return []

    # Must include first index, find LIS starting from it
    first_val = indices[0]
    dp = [1] * len(indices)
    prev = [-1] * len(indices)
    best_len = 1  # At minimum we keep the first index
    best_end = 0  # Start with first index

    # Start from second element
    for i in range(1, len(indices)):
        # Only consider sequences starting from first index
        if indices[i] > first_val:
            dp[i] = 2
            prev[i] = 0
            if dp[i] > best_len:
                best_len = dp[i]
                best_end = i

            # Try extending existing sequences
            for j in range(1, i):
                if indices[j] < indices[i] and prev[j] != -1 and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j
                    if dp[i] > best_len:
                        best_len = dp[i]
                        best_end = i

    # Reconstruct sequence
    result = []
    while best_end != -1:
        result.append(best_end)
        best_end = prev[best_end]
    result.reverse()
    return result


if __name__ == "__main__":

    def test_lis():
        test_cases = [
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
        ]

        for indices, expected_seq, desc in test_cases:
            print(f"\nTest: {desc}")
            print(f"Input indices: {indices}")
            print(f"Expected sequence: {expected_seq}")

            result = _find_longest_increasing_subsequence(indices)
            result_seq = [indices[i] for i in result] if result else []
            print(f"Result indices: {result}")
            print(f"Result sequence: {result_seq}")

            # Verify sequence is increasing
            assert all(
                result_seq[i] < result_seq[i + 1] for i in range(len(result_seq) - 1)
            ), "Sequence not increasing"

            # Verify first element is included
            if result:
                assert result[0] == 0, "First index not included"

            # Verify sequence matches expected
            assert (
                result_seq == expected_seq
            ), f"Wrong sequence: expected {expected_seq}, got {result_seq}"

            print("✓ Verified")

    print("Testing longest increasing subsequence algorithm...")
    test_lis()
    print("\nAll LIS tests passed!")

    def print_changes(old: list[str], new: list[str]):
        print(f"\nTransforming {old} -> {new}")
        changes = compute_list_changes(old, new)
        print(f"Delete: {changes.to_delete}")
        print(f"Add: {[(prev, new) for prev, new in changes.to_add]}")

        # Verify changes by applying them
        result = old.copy()
        # Apply deletions
        for x in changes.to_delete:
            result.remove(x)
        # Apply insertions
        for prev, item in changes.to_add:
            if prev is None:
                result.append(item)
            else:
                idx = result.index(prev) + 1
                result.insert(idx, item)
        print(f"Result: {result}")
        assert result == new, f"Failed! Expected {new}, got {result}"
        print("✓ Verified")

    # Test cases focusing on first item handling
    test_cases = [
        # First item exists in old list
        (["a", "b", "c", "d"], ["b", "c", "d", "e"]),  # delete a, insert (d,e)
        (["x", "y", "b", "c"], ["b", "c", "d"]),  # delete x,y, insert (c,d)
        (["x", "b", "y", "c"], ["b", "c", "d"]),  # delete x,y, insert (c,d)
        # First item exists but order changes
        (["x", "y", "b", "c"], ["b", "d", "c"]),  # delete x,y,c insert (b,d),(d,c)
        (["a", "b", "c", "d"], ["b", "a", "c", "d"]),  # delete a, insert (b,a)
        # Original test cases
        (["a", "b", "c"], ["a", "b", "c", "d"]),
        (["a", "b", "c", "d"], ["a", "b", "c"]),
        (["a", "b", "c", "d"], ["e", "a", "b", "c"]),
        (["a", "b", "c", "d"], ["a", "b", "d", "e", "c"]),
        (["a", "b", "c", "d"], ["a", "d", "c", "b"]),
    ]

    print("Testing message change algorithm...")
    for old, new in test_cases:
        print_changes(old, new)

    print("\nAll tests passed!")
