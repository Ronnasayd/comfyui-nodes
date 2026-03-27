# Installation and Execution

**Audience:** Developers

---

## Prerequisites

- Python 3.10+
- [ComfyUI](https://comfyui.org/) installed
- (Optional) [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) for easy extension management

## Repository Cloning

Clone this repository into your ComfyUI custom nodes directory:

```bash
git clone https://github.com/Ronnasayd/my_custom_nodes.git ComfyUI/custom_nodes/my_custom_nodes
```

Or install via ComfyUI-Manager by searching for "My-custom-nodes".

## Environment Configuration

No special environment variables are required for basic usage. For development:

- Install dev dependencies:
    ```bash
    pip install -e .[dev]
    pre-commit install
    ```
- Run pre-commit hooks and linters automatically on commit.

## Local Execution

1. Start ComfyUI as usual (see [ComfyUI docs](https://docs.comfy.org/get_started)).
2. The custom nodes will appear in the node palette.
3. Use them in your workflows.

## Tests

Run all unit tests with:

```bash
pytest
```

## Deploy (optional)

To publish your custom nodes to the ComfyUI registry:

1. Create an account at [registry.comfy.org](https://registry.comfy.org)
2. Set up your API key
3. Follow the registry publishing instructions

## Common Issues

- If nodes do not appear, ensure the folder is under `ComfyUI/custom_nodes/` and restart ComfyUI.
- For dependency or Python errors, check your Python version and virtual environment.
- See [faq.md](faq.md) for more troubleshooting tips.
