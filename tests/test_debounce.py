import asyncio

import pytest

from livekit.agents.utils.aio.debounce import Debounced, debounced


class TestDebounce:
    """Test cases for the Debounce class."""

    async def test_basic_debounce_execution(self) -> None:
        """Test that debounced function executes after delay."""
        result = []

        async def test_func() -> str:
            result.append("executed")
            return "success"

        debouncer = Debounced(test_func, 0.1)
        task = debouncer.schedule()

        # Should not execute immediately
        assert len(result) == 0
        assert debouncer.is_running()

        # Wait for execution
        value = await task
        assert value == "success"
        assert len(result) == 1
        assert result[0] == "executed"
        assert not debouncer.is_running()

    async def test_debounce_cancellation(self) -> None:
        """Test that rapid calls cancel previous executions."""
        execution_count = 0

        async def test_func() -> int:
            nonlocal execution_count
            execution_count += 1
            return execution_count

        debouncer = Debounced(test_func, 0.1)

        # Schedule multiple times rapidly
        task1 = debouncer.schedule()
        await asyncio.sleep(0.05)  # Wait less than delay
        task2 = debouncer.schedule()
        await asyncio.sleep(0.05)  # Wait less than delay
        task3 = debouncer.schedule()

        # First two tasks should be cancelled
        await asyncio.gather(task1, task2, task3, return_exceptions=True)

        assert task1.cancelled()
        assert task2.cancelled()

        # Only the last task should complete
        result = await task3
        assert result == 1
        assert execution_count == 1

    async def test_manual_cancellation(self) -> None:
        """Test manual cancellation of debounced task."""
        executed = False

        async def test_func() -> str:
            nonlocal executed
            executed = True
            return "done"

        debouncer = Debounced(test_func, 0.1)
        task = debouncer.schedule()

        assert debouncer.is_running()
        debouncer.cancel()

        assert not debouncer.is_running()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled()
        assert not executed

    async def test_call_operator(self) -> None:
        """Test that __call__ method works as alias for schedule."""
        result = []

        async def test_func() -> str:
            result.append("called")
            return "result"

        debouncer = Debounced(test_func, 0.05)
        task = debouncer()  # Using __call__

        value = await task
        assert value == "result"
        assert len(result) == 1

    async def test_is_running_status(self) -> None:
        """Test is_running method returns correct status."""

        async def test_func() -> str:
            await asyncio.sleep(0.05)
            return "done"

        debouncer = Debounced(test_func, 0.02)

        # Initially not running
        assert not debouncer.is_running()

        # Running after schedule
        task = debouncer.schedule()
        assert debouncer.is_running()

        # Not running after completion
        await task
        assert not debouncer.is_running()

    async def test_exception_handling(self) -> None:
        """Test that exceptions in debounced functions are properly propagated."""

        async def failing_func() -> None:
            raise ValueError("Test error")

        debouncer = Debounced(failing_func, 0.05)
        task = debouncer.schedule()

        with pytest.raises(ValueError, match="Test error"):
            await task

    async def test_multiple_debounce_instances(self) -> None:
        """Test that multiple Debounce instances work independently."""
        results: dict[str, list[str]] = {"func1": [], "func2": []}

        async def func1() -> str:
            results["func1"].append("executed")
            return "func1_result"

        async def func2() -> str:
            results["func2"].append("executed")
            return "func2_result"

        debouncer1 = Debounced(func1, 0.05)
        debouncer2 = Debounced(func2, 0.05)

        task1 = debouncer1.schedule()
        task2 = debouncer2.schedule()

        result1, result2 = await asyncio.gather(task1, task2)

        assert result1 == "func1_result"
        assert result2 == "func2_result"
        assert len(results["func1"]) == 1
        assert len(results["func2"]) == 1

    async def test_zero_delay(self) -> None:
        """Test debounce with zero delay."""
        executed = []

        async def test_func() -> str:
            executed.append("done")
            return "immediate"

        debouncer = Debounced(test_func, 0.0)
        task = debouncer.schedule()

        result = await task
        assert result == "immediate"
        assert len(executed) == 1


class TestDebounceDecorator:
    """Test cases for the @debounce decorator."""

    async def test_decorator_basic_usage(self) -> None:
        """Test basic usage of @debounce decorator."""
        execution_count = 0

        @debounced(0.1)
        async def decorated_func() -> str:
            nonlocal execution_count
            execution_count += 1
            return f"executed_{execution_count}"

        # The decorator returns a Debounce instance
        assert isinstance(decorated_func, Debounced)

        # Schedule execution
        task = decorated_func.schedule()
        result = await task

        assert result == "executed_1"
        assert execution_count == 1

    async def test_decorator_with_cancellation(self) -> None:
        """Test that decorator properly handles cancellation."""
        execution_count = 0

        @debounced(0.1)
        async def decorated_func() -> int:
            nonlocal execution_count
            execution_count += 1
            return execution_count

        # Multiple rapid calls
        task1 = decorated_func.schedule()
        await asyncio.sleep(0.05)
        task2 = decorated_func.schedule()
        await asyncio.sleep(0.05)
        task3 = decorated_func.schedule()

        # Wait for final execution
        result = await task3

        assert result == 1
        assert execution_count == 1
        assert task1.cancelled()
        assert task2.cancelled()

    async def test_decorator_call_operator(self) -> None:
        """Test that decorated function can be called directly."""
        results: list[str] = []

        @debounced(0.05)
        async def decorated_func() -> str:
            results.append("executed")
            return "success"

        # Call using () operator
        task = decorated_func()
        result = await task

        assert result == "success"
        assert len(results) == 1

    async def test_decorator_with_parameters(self) -> None:
        """Test decorator with different delay parameters."""

        @debounced(0.01)  # Very short delay
        async def fast_func() -> str:
            return "fast"

        @debounced(0.1)  # Longer delay
        async def slow_func() -> str:
            return "slow"

        # Both should work with their respective delays
        fast_task = fast_func.schedule()
        slow_task = slow_func.schedule()

        fast_result, slow_result = await asyncio.gather(fast_task, slow_task)

        assert fast_result == "fast"
        assert slow_result == "slow"

    async def test_decorator_maintains_function_signature(self) -> None:
        """Test that decorator preserves the original function's behavior."""

        @debounced(0.05)
        async def original_func() -> str:
            """Original docstring."""
            return "original_result"

        # The Debounce should wrap the original function
        task = original_func.schedule()
        result = await task
        assert result == "original_result"
