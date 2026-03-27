# System Modules

**Audience:** Developers

---

## Overview

The project is organized into modular components for maintainability and extensibility. Each node is implemented as a separate Python module within the main package. Tests and documentation are provided for each feature.

## Module Descriptions

### src/my_custom_nodes/

- **Objective:** Core package containing all custom node implementations for ComfyUI.
- **Main Features:**
    - `aspect_ratio_crop.py`: Crops images to a specified aspect ratio.
    - `pixelated_border_node.py`: Adds pixelated borders to images.
    - `video_segment_extender.py`: Extends video segments or frames.
    - `nodes.py`: Example and base node implementations.
- **Data Flow:** Receives image/video data from ComfyUI, processes it, and outputs results to downstream nodes.
- **Dependencies:** Relies on ComfyUI’s node interface and data structures.
- **Integration:** Nodes are auto-discovered by ComfyUI when placed in the correct directory.
- **Constraints:** Must follow ComfyUI’s node API; see [ComfyUI docs](https://docs.comfy.org/essentials/custom_node_overview).
- **Usage Example:**
    ```python
    # Example: Using Aspect Ratio Crop node in a workflow
    # (actual usage is via ComfyUI UI)
    from my_custom_nodes.aspect_ratio_crop import AspectRatioCropNode
    node = AspectRatioCropNode(aspect_ratio="16:9")
    result = node.process(image)
    ```
- **History:** Created with cookiecutter-comfy-extension; actively extended with new nodes.

### tests/

- **Objective:** Contains all unit tests for custom nodes.
- **Main Features:**
    - `test_my_custom_nodes.py`: Tests for node functionality and edge cases.
    - `conftest.py`: Pytest configuration and fixtures.
- **Data Flow:** Tests import node modules and verify outputs for given inputs.
- **Integration:** Run with `pytest`.
- **Constraints:** Follows Pytest conventions.
- **Usage Example:**
    ```bash
    pytest
    ```
- **History:** Updated as new nodes are added.

### docs/features/

- **Objective:** Feature-specific documentation for advanced node capabilities.
- **Main Features:**
    - Video segment temporal consistency
    - Video latent mask
- **Integration:** Linked from [SUMMARY.md](SUMMARY.md) and referenced in feature discussions.

### docs/agents/

- **Objective:** Agent plans and specifications for advanced automation and workflow features.
- **Main Features:**
    - Plans and specs for video/image processing agents
- **Integration:** Used for planning and documenting agent-based features.

## Module Relationships

- `src/my_custom_nodes/` is the core; all other modules (tests, docs) support or extend it.
- Feature and agent docs provide context and guidance for using and extending nodes.

## Possible Improvements / Technical Debt

- Expand test coverage for new nodes and edge cases.
- Add more advanced feature documentation as new capabilities are developed.
- Consider splitting large nodes into submodules for clarity.

---

For more details, see the [architecture overview](architecture.md) and [usage guide](usage.md).
