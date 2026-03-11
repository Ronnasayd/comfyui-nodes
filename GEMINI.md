# GEMINI.md - Project Context

## Project Overview

**my_custom_nodes** is a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed to extend its functionality with specialized image and video processing capabilities. The project was initialized using the [ComfyUI Cookiecutter Template](https://github.com/Comfy-Org/cookiecutter-comfy-extension).

### Key Features & Nodes

- **Image Processing**:
    - `AspectRatioCrop`: Crops images to match a target aspect ratio.
    - `PixelatedBorderNode`: Adds stylized pixelated borders to images.
- **Video Segmentation & Temporal Consistency**:
    - `VideoSegmentPrepare` & `VideoSegmentSave`: Infrastructure for processing long videos in smaller segments.
    - `LatentPrependCache`, `LatentExtendFrames`, `ConditioningExtendFrames`: Advanced nodes for maintaining temporal consistency between video segments using latent caching.
- **Debugging**:
    - `LatentShapeDebug` & `ConditioningShapeDebug`: Utility nodes for inspecting tensor shapes within workflows.

### Architecture

- **Entry Point**: The root `__init__.py` exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` required by ComfyUI.
- **Source Code**: Implementation logic resides in `src/my_custom_nodes/`.
    - `nodes.py`: Central registration point for all nodes.
    - `video_segment_extender.py`: Contains complex logic for video segment handling and caching.
- **Tests**: Comprehensive test suite in `tests/` using Pytest and Torch.

---

## Building and Running

### Development Setup

1. **Clone into ComfyUI**:
    ```bash
    cd ComfyUI/custom_nodes
    git clone <repository_url>
    ```
2. **Install Dependencies**:
    ```bash
    pip install -e .[dev]
    ```
3. **Install Pre-commit Hooks**:
    ```bash
    pre-commit install
    ```

### Running Tests

Execute the full test suite using Pytest:

```bash
pytest tests/
```

### Linting and Type Checking

The project uses `ruff` for linting and `mypy` for static type checking:

```bash
# Linting
ruff check .

# Type Checking
mypy .
```

---

## Development Conventions

### Coding Style

- **Linter**: `ruff` is configured with a line length of 140 characters and specific safety checks (e.g., banning `exec`/`eval`).
- **Type Safety**: `mypy` is used in `strict` mode to ensure high type safety across the codebase (except for tests where some rules are relaxed).
- **Naming**: Follows standard Python (PEP 8) and ComfyUI naming conventions (PascalCase for Node Classes).

### Testing Practices

- **Framework**: `pytest`.
- **Standards**: Every new node or significant logic change should be accompanied by unit tests in `tests/test_my_custom_nodes.py`.
- **CI/CD**: GitHub Actions (`build-pipeline.yml`) automatically run linting and tests on every Pull Request to `main` or `master`.

### ComfyUI Integration

- All exported nodes must be added to `NODE_CLASS_MAPPINGS` in `src/my_custom_nodes/nodes.py`.
- Metadata such as `RETURN_TYPES`, `FUNCTION`, and `CATEGORY` must be explicitly defined in each node class.
- The project follows the `tool.comfy` configuration in `pyproject.toml` for registry publishing.

- @.github/instructions/workspace.instructions.md
- @.github/instructions/copilot.instructions.md

## References:
@.github/instructions/orchestration.instructions.md
