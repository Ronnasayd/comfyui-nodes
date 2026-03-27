# My-custom-nodes

> A collection of custom nodes for ComfyUI, enabling advanced workflows and video/image processing extensions.

## Overview

My-custom-nodes is a Python extension for [ComfyUI](https://comfyui.org/) that provides a set of custom nodes for image and video processing, including cropping, border effects, and video segment manipulation. The project is structured for easy extension, testing, and integration with the ComfyUI ecosystem.

## High-Level Architecture

- Modular Python package: `src/my_custom_nodes/`
- Each node is a self-contained module
- Designed for compatibility with ComfyUI and ComfyUI-Manager
- Includes unit tests and feature documentation

## Quick Installation

1. Install [ComfyUI](https://docs.comfy.org/get_started)
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Search for "My-custom-nodes" in ComfyUI-Manager, or clone this repo under `ComfyUI/custom_nodes/`
4. Restart ComfyUI

## How to Use

- Use the new nodes in your ComfyUI workflows
- See [usage guide](usage.md) for examples

## Modules

- [src/my_custom_nodes/](../src/my_custom_nodes/) — core node implementations
- [tests/](../tests/) — unit tests
- [docs/features/](features/) — feature documentation
- [docs/agents/](agents/) — agent plans and specs

## Contribution

- See [contribution guide](contribution.md) for branching, PR, and code standards

## Architectural Decisions

- See [architecture overview](architecture.md) for diagrams and design decisions

## Technologies

- Python 3.10+
- ComfyUI
- Pytest, Ruff, Mypy, Pre-commit

## FAQ / Common Issues

- See [faq](faq.md) for troubleshooting and best practices

---

## Documentation Index

- [SUMMARY.md](SUMMARY.md) — this file
- [architecture.md](architecture.md)
- [setup.md](setup.md)
- [usage.md](usage.md)
- [modules.md](modules.md)
- [contribution.md](contribution.md)
- [models.md](models.md) _(if applicable)_
- [endpoints.md](endpoints.md) _(if applicable)_
- [faq.md](faq.md) _(if applicable)_
- [adr/](adr/) _(if applicable)_
- [techs/](techs/) _(if applicable)_
- [features/](features/) — feature docs
- [agents/](agents/) — agent plans/specs
