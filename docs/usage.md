# How to Use

**Audience:** End users, integrators

---

## Main Flow

1. Open ComfyUI and create a new workflow.
2. Locate the custom nodes from "My-custom-nodes" in the node palette.
3. Drag and connect nodes as needed for your image or video processing pipeline.
4. Configure node parameters in the UI.
5. Run the workflow to process your data.

## Usage Examples

### Example 1: Aspect Ratio Crop

- Add the "Aspect Ratio Crop" node to your workflow.
- Connect an image input node to it.
- Set the desired aspect ratio (e.g., 16:9).
- Connect the output to further processing or save nodes.

### Example 2: Video Segment Extender

- Use the "Video Segment Extender" node to process video frames.
- Connect a video input or frame sequence.
- Configure extension parameters as needed.

### Example 3: Pixelated Border

- Add the "Pixelated Border" node to add stylized borders to images.
- Adjust border size and pixelation level in the node settings.

## Expected Outputs

- Processed images or video frames, depending on the node used.
- Outputs are available for further nodes or saving to disk.

## Advanced Use Cases

- Combine multiple custom nodes for complex workflows (e.g., crop, then add border, then extend video segment).
- Integrate with other ComfyUI nodes for advanced effects.

## Errors and Best Practices

- If a node fails, check the input types and parameters.
- Use unit tests in [tests/](../tests/) to verify node behavior during development.
- Refer to [faq.md](faq.md) for troubleshooting.

---

For more details on each node, see the [modules overview](modules.md) and [feature docs](features/).
