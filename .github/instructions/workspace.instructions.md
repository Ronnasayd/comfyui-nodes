# Copilot Instructions — comfyui-node

## Project Overview

A ComfyUI custom node extension containing image and video processing nodes. Installed as a package under `ComfyUI/custom_nodes/`. Generated from the [cookiecutter-comfy-extension](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template.

## Architecture

### Node Registration Flow

ComfyUI discovers nodes via the top-level [`__init__.py`](../__init__.py), which re-exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from [`src/my_custom_nodes/nodes.py`](../src/my_custom_nodes/nodes.py). Adding a new node requires:

1. Implement the node class in `src/my_custom_nodes/<name>.py`
2. Import and add it to both dicts in `nodes.py`
3. `NODE_CLASS_MAPPINGS` keys must be **globally unique** across all ComfyUI extensions

### Mandatory Node Class Contract

Every node class must define:

```python
@classmethod
def INPUT_TYPES(cls): ...   # returns {"required": {...}, "optional": {...}}
RETURN_TYPES = ("IMAGE",)   # tuple of ComfyUI type strings
FUNCTION = "my_method"      # name of the entry-point method
CATEGORY = "MYNodes"        # UI category (use "MYNodes/SubCat" for sub-menus)
```

Optional: `RETURN_NAMES`, `DESCRIPTION`, `OUTPUT_NODE = True`, `IS_CHANGED`.

### Tensor Format Conventions

- **IMAGE inputs** arrive as `torch.Tensor` shape `[B, H, W, C]`, float32 0–1
- Torchvision expects `[C, H, W]` — convert with `.permute(2, 0, 1).contiguous()`
- Return IMAGE as `[1, H, W, C]` (unsqueeze batch dim)
- **MASK** shape is `[1, H, W]`, float32 0–1 (0 = keep, 1 = inpaint area)

### VIDEO Type Handling

Videos may arrive as `VideoInput` objects (modern ComfyUI) **or** raw tensors `[F, H, W, C]` / `[B, F, H, W, C]`. Always check both:

```python
if hasattr(video_input, "get_components"):
    tensor = video_input.get_components().images
elif isinstance(video_input, torch.Tensor):
    tensor = video_input
```

See [`video_segment_extender.py`](../src/my_custom_nodes/video_segment_extender.py) for the pattern.

### External Dependencies

- `folder_paths` — ComfyUI built-in for output/input/model directories (not installable via pip)
- `ffmpeg` — required binary for video encode/decode; invoked via `subprocess`
- `torch`, `torchvision`, `PIL`, `numpy` — standard image processing stack

## Developer Workflows

### Setup

```bash
pip install -e .[dev]
pre-commit install
```

### Run Tests

```bash
pytest tests/
```

Tests live in `tests/`, and `conftest.py` inserts the project root into `sys.path` so `src.my_custom_nodes` is importable without ComfyUI being present.

### Lint

```bash
ruff check .
ruff format .
```

Pre-commit runs ruff automatically on each commit. Line length is 140, target is Python 3.9+.

### Publish

Push a git tag — the [`publish_node.yml`](workflows/publish_node.yml) action publishes to the ComfyUI Registry using the `REGISTRY_ACCESS_TOKEN` secret.

## Video Segment Pipeline

`VideoSegmentPrepare` + `VideoSegmentSave` implement a **cycle-free** multi-segment video extension pattern: run the ComfyUI queue N times (once per segment). `IS_CHANGED` returns `time.time()` to force re-execution on every queue run so the nodes read fresh state from disk.

## Conventions

- Categories: `"MYNodes"` for image nodes, `"MYNodes/VideoSegment"` for the video pipeline
- Docstrings on node classes populate `DESCRIPTION` via `cleandoc(__doc__)`
- Memory cleanup after video processing: `del video; torch.cuda.empty_cache()`
- Segment files follow the pattern `segment_{index:03d}.mp4` under the project output folder
