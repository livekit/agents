---
name: docstring
description: Generate Google-style docstrings for Python classes and functions
---

Add or update docstrings for Python code. This skill follows Google-style docstrings compatible with pdoc3.

**Note:** For pdoc3 to properly render Google-style docstrings, ensure the module includes `__docformat__ = "google"` at the top, or run pdoc with `--docformat google`.

## Usage

```bash
/docstring ClassName
/docstring path/to/file.py
```

## Instructions

1. Find the target class or function
2. If multiple matches exist, ask for clarification
3. Skip already-documented code unless explicitly asked to update
4. Generate comprehensive docstrings

## Format

```python
def function_name(arg1: str, arg2: int = 0) -> bool:
    """Short one-line description.

    Longer description if needed, explaining the purpose
    and any important behavior.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2. Defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is empty.

    Example:
        >>> function_name("hello", 42)
        True
    """
```

## Class Docstrings

```python
class MyClass:
    """Short description of the class.

    Longer description explaining the purpose and usage.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.
    """
```

## Guidelines

- Keep the first line under 80 characters
- Use imperative mood ("Return" not "Returns")
- Document all public methods and attributes
- Include type hints in signatures, not docstrings
- Add Examples section for complex functionality
