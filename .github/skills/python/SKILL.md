---
name: python
description: Coding conventions and best practices for Python based on the Google Python Style Guide, emphasizing readability, explicitness, consistent naming, clear docstrings, safe exception handling, small focused functions, and static typing for writing maintainable production-quality Python code.
---

# Skill: Python (Google Style Guide)

## Overview

This skill implements conventions from the Google Python Style Guide.
The goal is to produce Python code that is readable, maintainable, and consistent across large codebases.

Reference: https://google.github.io/styleguide/pyguide.html

---

# 1. General Principles

- Prefer readability and maintainability over cleverness.
- Be consistent with the surrounding code.
- Follow automated tooling where possible.

Recommended tools:

- pylint
- Black or Pyink formatter
- type checkers (pytype, mypy)

---

# 2. Imports

## Import Rules

Imports should be grouped in the following order:

1. Standard library
2. Third-party libraries
3. Local application imports

Example:

```python
import os
import sys

import requests

from myproject import utils
```

Avoid wildcard imports:

```python
from module import *
```

---

# 3. Exceptions

### Creating Exceptions

Custom exceptions must:

- inherit from an existing exception
- end with `Error`

Example:

```python
class DataValidationError(Exception):
    pass
```

### Avoid Catch-All Exceptions

Avoid:

```python
except:
```

or

```python
except Exception:
```

Unless:

- re-raising the exception
- isolating failures such as thread boundaries.

---

# 4. Mutable Global State

Avoid mutable global variables when possible.

Prefer:

- constants
- dependency injection
- passing objects explicitly.

---

# 5. Functions

## Function Length

Prefer small, focused functions.

Guideline:

- reconsider functions longer than ~40 lines.

Large functions make debugging and modification harder.

---

# 6. Lambda Functions

Use lambda functions only for simple expressions.

Good:

```python
sorted(items, key=lambda x: x.id)
```

Bad:

```python
lambda x: complicated_logic(x)
```

If logic becomes complex, use a named function.

---

# 7. Conditional Expressions

Python ternary expressions are allowed for simple cases:

```python
value = "yes" if condition else "no"
```

Avoid long or complex ternary expressions.

---

# 8. Default Argument Values

Default parameters are allowed:

```python
def foo(value, retries=3):
    pass
```

Be careful with mutable defaults:

Bad:

```python
def add_item(item, items=[]):
    items.append(item)
```

Correct:

```python
def add_item(item, items=None):
    if items is None:
        items = []
```

---

# 9. Truth Value Testing

Prefer implicit boolean checks:

Good:

```python
if items:
```

Avoid:

```python
if len(items) > 0:
```

Check for `None` explicitly:

```python
if value is None:
```

---

# 10. Type Annotations

Use Python type hints for readability and static analysis.

Example:

```python
def get_user(id: int) -> str:
    return "user"
```

Benefits:

- improved readability
- earlier error detection
- better IDE support. ([Google GitHub][1])

---

# 11. Line Length

Maximum line length:

```
80 characters
```

Exceptions include:

- long URLs
- import lines
- long constants.

Avoid using `\` for line continuation.

Prefer parentheses:

```python
result = (
    some_function_with_long_name(
        arg1,
        arg2,
    )
)
```

---

# 12. Semicolons

Do not use semicolons.

Incorrect:

```python
x = 1; y = 2
```

Correct:

```python
x = 1
y = 2
```

---

# 13. Naming Conventions

Follow these naming patterns:

| Entity     | Style              |
| ---------- | ------------------ |
| Modules    | `lower_with_under` |
| Packages   | `lower_with_under` |
| Classes    | `CapWords`         |
| Exceptions | `CapWords`         |
| Functions  | `lower_with_under` |
| Methods    | `lower_with_under` |
| Variables  | `lower_with_under` |
| Constants  | `CAPS_WITH_UNDER`  |

Examples:

```python
class UserService:
    pass

def fetch_user():
    pass

MAX_RETRIES = 3
```

Names should be descriptive and avoid abbreviations. ([Google GitHub][1])

---

# 14. File Naming

Rules:

- must end with `.py`
- no dashes (`-`)

Correct:

```
user_service.py
```

Incorrect:

```
user-service.py
```

---

# 15. Module Structure

Every module should begin with a module docstring.

Example:

```python
"""Example module.

Provides utilities for working with user accounts.
"""
```

---

# 16. Docstrings

Use triple quotes:

```python
"""Summary line.

Detailed explanation.
"""
```

### Function Docstring Example

```python
def fetch_rows(table, keys):
    """Fetch rows from a Bigtable.

    Args:
        table: Bigtable instance.
        keys: List of keys.

    Returns:
        Mapping of keys to rows.
    """
```

Sections may include:

- Args
- Returns
- Raises
- Attributes

Docstrings explain _what_ code does, not how.

---

# 17. Comments

Write clear comments explaining intent.

Good:

```python
# Binary search to locate value.
```

Bad:

```python
# increment i
i += 1
```

Comments should:

- be grammatically correct
- start with `# ` (space after hash)
- describe purpose, not obvious behavior. ([Google GitHub][1])

---

# 18. Main Entry Point

Executable modules should use:

```python
def main():
    pass

if __name__ == "__main__":
    main()
```

This prevents execution when imported.

---

# 19. Consistency

When style rules conflict:

1. follow the local file
2. follow the project style
3. otherwise use the Google guide.

Consistency across a codebase is more important than personal preference.

---

# Key Principles

1. Prefer readability.
2. Avoid surprising behavior.
3. Write small, focused functions.
4. Use clear naming conventions.
5. Document public APIs.
6. Use static typing where possible.

---

# Summary

The Google Python Style Guide emphasizes:

- strong readability
- consistent naming
- clear documentation
- small maintainable functions
- linting and static analysis
