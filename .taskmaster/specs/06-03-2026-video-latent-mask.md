<description>

## Problem Summary

The user wants to create a new ComfyUI node called `VideoLatentMask` to facilitate video-to-video workflows or temporal transitions where initial frames of a segment need to be preserved (masked) while subsequent frames are modified by the model.

Currently, the project has nodes for latent caching and prepending, but lacks a dedicated utility to generate a frame-based mask for 5D video latents `[B, C, F, H, W]`.

The objective is to generate a latent tensor where:
- The first `black_frames` are set to `0.0` (black/masked).
- The remaining frames are set to `1.0` (white/unmasked).

## Relevant Files for Solving the Problem

### Implementation
- **`src/my_custom_nodes/video_segment_extender.py`**: The new node class `VideoLatentMask` will be implemented here alongside other video-related nodes.

### Registration
- **`src/my_custom_nodes/nodes.py`**: The node must be added to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.

### Testing
- **`tests/test_my_custom_nodes.py`**: Unit tests should be added here to verify the generated mask's shape and content.

---

## Relevant Code Snippets for Solving the Problem

### 5D Video Latent Shape Reference
```python
# From src/my_custom_nodes/video_segment_extender.py
# Video latent: [B, C, F, H, W]
# B=Batch, C=Channels, F=Frames, H=Height, W=Width
```

### Latent Dictionary Format
```python
# Nodes return latents as a dictionary
return ({"samples": latent_tensor},)
```

### Typical Node Structure in the Project
```python
class VideoLatentMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "channels": ("INT", {"default": 16, "min": 1, "max": 128}),
                "frames": ("INT", {"default": 16, "min": 1, "max": 256}),
                "height": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 8}),
                "black_frames": ("INT", {"default": 4, "min": 0, "max": 256}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "MYNodes/VideoSegment"

    def generate_mask(self, batch_size, channels, frames, height, width, black_frames):
        # Implementation logic
        pass
```

---

## Proposed Action Plan for Task Implementation

1. **Implement `VideoLatentMask` in `src/my_custom_nodes/video_segment_extender.py`**:
    - Create the class with appropriate `INPUT_TYPES`.
    - Implement `generate_mask` method:
        - Create a tensor of ones with shape `(batch_size, channels, frames, height, width)`.
        - Set the slice `[:, :, :black_frames, :, :]` to `0.0`.
        - Use `torch.zeros` and `torch.ones` or `torch.full`.
        - Ensure the tensor is on the correct device (CPU by default for creation, or follow existing patterns).
    - Add detailed docstrings explaining the purpose and parameters.

2. **Register the Node in `src/my_custom_nodes/nodes.py`**:
    - Import `VideoLatentMask`.
    - Add `"VideoLatentMask": VideoLatentMask` to `NODE_CLASS_MAPPINGS`.
    - Add `"VideoLatentMask": "Video Latent Mask"` to `NODE_DISPLAY_NAME_MAPPINGS`.

3. **Verify and Format**:
    - Run `ruff` to ensure coding standards are met.
    - Ensure type safety with `mypy` if necessary.

---

## Testing Strategy for Validating the Implementation

### Unit Tests (`tests/test_my_custom_nodes.py`)

1. **Test Shape**:
    - Call `generate_mask` with various parameters and assert the returned tensor shape matches `[B, C, F, H, W]`.

2. **Test Content**:
    - Assert that frames `0` to `black_frames - 1` are all `0.0`.
    - Assert that frames `black_frames` to `frames - 1` are all `1.0`.
    - Test edge cases: `black_frames = 0` (all white), `black_frames = frames` (all black).

3. **Integration Check**:
    - Verify the node is present in `NODE_CLASS_MAPPINGS`.

---

## Context Map
```markdown
  ### Files to Modify
  | File | Purpose | Changes Needed |
  |------|---------|----------------|
  | src/my_custom_nodes/video_segment_extender.py | Node Implementation | Add VideoLatentMask class |
  | src/my_custom_nodes/nodes.py | Node Registration | Export the new node |
  
  ### Test Files
  | Test | Coverage |
  |------|----------|
  | tests/test_my_custom_nodes.py | Unit tests for VideoLatentMask | Correctness of tensor generation |
  
  ### Reference Patterns
  | File | Pattern |
  |------|---------|
  | src/my_custom_nodes/video_segment_extender.py | 5D tensor handling in LatentExtendFrames |
  
  ### Risk Assessment
  - [ ] Breaking changes to public API: None, adding a new node.
  - [ ] Performance impact: Minimal, simple tensor allocation.
```

</description>

<!--- THE FOLLOWING TEXT IS UNCHANGEABLE. --->
<!--- DO NOT REWRITE, DO NOT CORRECT, DO NOT ADAPT. --->
<!--- USE IT EXACTLY AS IT IS, CHARACTER BY CHARACTER. --->
<!--- UNCHANGING_TEXT_START --->
<workflow>
- If documentation files or any other type of file are provided, extract relevant links and related files that may assist in implementing the task.
- When creating a task or subtask, add references to relevant files or links that may assist in implementing the task.
- Before each implementation step (tasks or subtasks), check relevant references and links. Perform a thorough review of relevant files and documents until you have a complete understanding of what needs to be done.
- Add relevant code snippets that may assist in implementing the task in markdown format.
- Check all *.md files starting from SUMMARY.md and docs/ to find relevant documentation.
- Create and present a detailed action plan for executing the task implementation.
- Ensure that changes are fully backward compatible and do not affect other system flows.
- At the end of the implementation, show a summary of what was done and save it as a .md file in docs/features/dd-MM-YYYY-<description>/README.md
</workflow>
<!--- UNCHANGING_TEXT_END --->
