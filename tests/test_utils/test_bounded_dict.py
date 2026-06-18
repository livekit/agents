from dataclasses import dataclass

import pytest

from livekit.agents.utils.bounded_dict import BoundedDict


@dataclass
class _Item:
    name: str
    value: int = 0


class TestInit:
    def test_defaults(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict()
        assert bd.maxsize is None
        assert len(bd) == 0

    def test_maxsize_stored(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=5)
        assert bd.maxsize == 5

    @pytest.mark.parametrize("bad", [0, -1, -100], ids=["zero", "neg_one", "neg_hundred"])
    def test_invalid_maxsize_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="maxsize must be greater than 0"):
            BoundedDict(maxsize=bad)


class TestSetItem:
    def test_basic_insert(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=3)
        bd["a"] = 1
        bd["b"] = 2
        assert dict(bd) == {"a": 1, "b": 2}

    def test_evicts_oldest_on_overflow(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=3)
        bd["a"] = 1
        bd["b"] = 2
        bd["c"] = 3
        bd["d"] = 4
        assert "a" not in bd
        assert list(bd.keys()) == ["b", "c", "d"]

    def test_repeated_overflow(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=2)
        for i in range(10):
            bd[str(i)] = i
        assert list(bd.keys()) == ["8", "9"]
        assert list(bd.values()) == [8, 9]

    def test_overwrite_existing_key_no_eviction(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=2)
        bd["a"] = 1
        bd["b"] = 2
        bd["a"] = 10
        assert len(bd) == 2
        assert bd["a"] == 10

    def test_no_maxsize_unlimited(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict()
        for i in range(100):
            bd[str(i)] = i
        assert len(bd) == 100

    def test_maxsize_one(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=1)
        bd["a"] = 1
        bd["b"] = 2
        assert len(bd) == 1
        assert bd["b"] == 2
        assert "a" not in bd


class TestUpdateValue:
    def test_updates_existing_fields(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        bd["x"] = _Item(name="original", value=0)
        result = bd.update_value("x", name="updated", value=42)
        assert result is not None
        assert result.name == "updated"
        assert result.value == 42
        assert bd["x"].name == "updated"

    def test_missing_key_returns_none(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        result = bd.update_value("missing", name="test")
        assert result is None

    def test_none_field_value_skipped(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        bd["x"] = _Item(name="keep", value=10)
        bd.update_value("x", name=None, value=99)
        assert bd["x"].name == "keep"
        assert bd["x"].value == 99

    def test_nonexistent_field_warns(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        bd["x"] = _Item(name="test")
        result = bd.update_value("x", bogus_field="val")
        assert result is not None
        assert not hasattr(result, "bogus_field")

    def test_mutates_in_place(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        item = _Item(name="orig")
        bd["x"] = item
        bd.update_value("x", name="changed")
        assert item.name == "changed"


class TestSetOrUpdate:
    def test_creates_new_entry(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        result = bd.set_or_update("k", factory=lambda: _Item(name="new"), value=5)
        assert result.name == "new"
        assert result.value == 5
        assert "k" in bd

    def test_updates_existing_entry(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        bd["k"] = _Item(name="old", value=0)
        result = bd.set_or_update("k", factory=lambda: _Item(name="unused"), value=99)
        assert result.name == "old"
        assert result.value == 99

    def test_factory_not_called_when_key_exists(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict()
        bd["k"] = _Item(name="existing")
        called = False

        def factory() -> _Item:
            nonlocal called
            called = True
            return _Item(name="should_not_appear")

        bd.set_or_update("k", factory=factory, name="updated")
        assert not called

    def test_respects_maxsize(self) -> None:
        bd: BoundedDict[str, _Item] = BoundedDict(maxsize=2)
        bd.set_or_update("a", factory=lambda: _Item(name="a"))
        bd.set_or_update("b", factory=lambda: _Item(name="b"))
        bd.set_or_update("c", factory=lambda: _Item(name="c"))
        assert "a" not in bd
        assert list(bd.keys()) == ["b", "c"]


class TestPopIf:
    def test_pop_no_predicate_fifo(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict()
        bd["a"] = 1
        bd["b"] = 2
        bd["c"] = 3
        key, value = bd.pop_if()
        assert key == "a"
        assert value == 1
        assert len(bd) == 2

    def test_pop_no_predicate_empty(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict()
        key, value = bd.pop_if()
        assert key is None
        assert value is None

    def test_pop_with_predicate_finds_last_match(self) -> None:
        """pop_if iterates in reverse order, returning the most recently added match."""
        bd: BoundedDict[str, int] = BoundedDict()
        bd["a"] = 10
        bd["b"] = 20
        bd["c"] = 30
        key, value = bd.pop_if(predicate=lambda v: v >= 20)
        assert key == "c"
        assert value == 30
        assert list(bd.keys()) == ["a", "b"]

    def test_pop_with_predicate_no_match(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict()
        bd["a"] = 1
        bd["b"] = 2
        key, value = bd.pop_if(predicate=lambda v: v > 100)
        assert key is None
        assert value is None
        assert len(bd) == 2

    def test_pop_with_predicate_single_match(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict()
        bd["x"] = 5
        bd["y"] = 50
        bd["z"] = 3
        key, value = bd.pop_if(predicate=lambda v: v > 10)
        assert key == "y"
        assert value == 50
        assert "y" not in bd
        assert len(bd) == 2


class TestOrderPreservation:
    def test_insertion_order(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=5)
        for k in "abcde":
            bd[k] = ord(k)
        assert list(bd.keys()) == ["a", "b", "c", "d", "e"]

    def test_eviction_preserves_remaining_order(self) -> None:
        bd: BoundedDict[str, int] = BoundedDict(maxsize=3)
        for k in "abcde":
            bd[k] = ord(k)
        assert list(bd.keys()) == ["c", "d", "e"]
