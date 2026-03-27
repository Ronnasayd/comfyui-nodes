# FAQ

**Audience:** Everyone

---

## Why don't my custom nodes appear in ComfyUI?

- Ensure the `my_custom_nodes` folder is placed under `ComfyUI/custom_nodes/`.
- Restart ComfyUI after adding or updating nodes.
- Check for Python errors in the ComfyUI console.

## How do I install development dependencies?

- Run:
    ```bash
    pip install -e .[dev]
    pre-commit install
    ```

## How do I run the tests?

- From the project root, run:
    ```bash
    pytest
    ```

## What Python version is required?

- Python 3.10 or higher is recommended for compatibility.

## How do I publish my custom nodes to the registry?

- Create an account at [registry.comfy.org](https://registry.comfy.org)
- Set up your API key and follow the publishing instructions in [setup.md](setup.md).

## Where can I find more documentation?

- See the [SUMMARY.md](SUMMARY.md) for a full documentation index.
- Feature-specific docs are in [docs/features/](features/).

## Who maintains this project?

- See the [pyproject.toml](../pyproject.toml) for author and contact information.

---

For more troubleshooting, see [setup.md](setup.md) and [usage.md](usage.md).
