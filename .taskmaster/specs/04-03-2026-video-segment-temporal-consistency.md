# Video Segment Temporal Consistency Implementation

## Problem Summary

The current `video_segment_extender.py` implementation lacks proper temporal consistency across video segments. When generating long videos by concatenating multiple segments, there is no mechanism to preserve the temporal flow between segments, resulting in visual discontinuities and jarring transitions.

The module currently:

- Extracts the last frame from previous segments as an IMAGE (2D) to initialize the next segment
- Has placeholder parameters (`cache_window_size`, `overlap_frames`) that are not implemented
- Performs concatenation only at the video file level (via ffmpeg), with no latent-space continuity
- Does not handle the temporal context needed for smooth video-to-video transitions

To ensure temporal consistency, the system must:

1. **Cache latent frames** from the end of each segment to disk
2. **Prepend cached latents** to the next segment's generation process
3. **Handle extended segment lengths** resulting from latent prepending
4. **Trim overlapping frames** during final concatenation to maintain smooth transitions
5. Support multiple video formats and tensor representations (VideoInput objects, raw tensors)

This implementation is critical for high-quality long-form video generation where temporal coherence is essential.

---

## Relevant Files for Solving the Problem

### Primary Implementation File

- **`src/my_custom_nodes/video_segment_extender.py`** (544 lines)
    - Contains `VideoSegmentPrepare`, `VideoSegmentSave`, and `LatentShapeDebug` nodes
    - Has existing infrastructure for segment management, ffmpeg operations, and frame extraction
    - Missing: latent caching logic, latent prepending, and overlap handling

### Related Files

- **`src/my_custom_nodes/nodes.py`** (27 lines)
    - Node registration file
    - No changes needed, but useful for understanding node export pattern

- **`test_latent_debug.py`** (96 lines)
    - Demonstrates latent shape handling for video (5D) and image (4D) tensors
    - Shows overlap extraction pattern: `full_latent[:, :, -overlap_frames:, :, :]`
    - Useful reference for latent manipulation

- **`tests/test_my_custom_nodes.py`** (60 lines)
    - Existing test suite structure
    - New tests for latent caching/loading should be added here

### External Dependencies

- `folder_paths` (ComfyUI built-in): for project directory management
- `torch`: tensor operations for latent manipulation
- `ffmpeg`/`ffprobe`: video processing (already used)
- `numpy`, `PIL`: image/array conversion (already used)

---

## Relevant Code Snippets for Solving the Problem

### Current Latent Pass-Through Pattern

```python
# VideoSegmentPrepare.prepare() - lines 371-414
def prepare(self, ..., latent=None, cache_window_size=4, overlap_frames=8):
    # ... segment logic ...
    return (frame, current_segment, False, "", latent)  # latent is passed through unchanged
```

### Current Video Tensor Handling Pattern

```python
# _save_video_tensor() - lines 261-298
if hasattr(video_input, "get_components"):
    tensor = video_input.get_components().images.detach().cpu()
elif isinstance(video_input, torch.Tensor):
    tensor = video_input.detach().cpu()
else:
    raise TypeError(f"Unsupported VIDEO type: {type(video_input)}")
```

### Latent Shape Detection Pattern (from test_latent_debug.py)

```python
samples = latent.get("samples")  # latent is dict with "samples" key
if len(shape) == 5:
    b, c, f, h, w = shape  # [Batch, Channels, Frames, Height, Width]
elif len(shape) == 4:
    b, c, h, w = shape  # [Batch, Channels, Height, Width] - image latent
```

### Overlap Extraction Reference (from test_latent_debug.py)

```python
# Extract last N frames from a 5D latent tensor
overlap_frames = 8
full_latent = torch.randn(1, 4, 32, 64, 64)  # B=1, C=4, F=32, H=64, W=64
overlap_latent = full_latent[:, :, -overlap_frames:, :, :]  # Last 8 frames
```

### Project Path Management Pattern

```python
# _get_project_path() - lines 87-91
def _get_project_path(project_name: str) -> str:
    base_output = folder_paths.get_output_directory()
    project_path = os.path.join(base_output, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path
```

### IS_CHANGED Pattern (forces re-execution)

```python
# VideoSegmentPrepare - lines 361-363
@classmethod
def IS_CHANGED(cls, **kwargs):
    return hash(tuple(sorted(kwargs.items())))  # Always re-executes on each queue run
```

---

## Proposed Action Plan for Task Implementation

### **Phase 1: Latent Caching Infrastructure**

**Objective:** Implement disk-based persistence for latent frames

1. **Create latent save function** `_save_latent_cache(latent_dict, project_path, segment_index)`
    - Validate latent is dict with "samples" key containing torch.Tensor
    - Extract last N frames: `samples[:, :, -overlap_frames:, :, :]` (for 5D video latents)
    - Save to disk using `torch.save()` at path: `{project_path}/latent_cache_seg_{index:03d}.pt`
    - Include metadata: shape, dtype, device, timestamp
    - Handle edge cases: None latent, image latents (4D), insufficient frames

2. **Create latent load function** `_load_latent_cache(project_path, segment_index)`
    - Load from disk using `torch.load()`
    - Validate loaded data structure and shapes
    - Return latent dict format: `{"samples": tensor}`
    - Return None if cache doesn't exist or is corrupted
    - Log cache hit/miss for debugging

3. **Implement cache cleanup utility** `_cleanup_old_caches(project_path, keep_last_n=2)`
    - Remove outdated cache files to save disk space
    - Keep only the last N segment caches (default: 2)

**Deliverables:**

- Three new helper functions in `video_segment_extender.py`
- Unit tests validating save/load roundtrip
- Error handling for all edge cases

---

### **Phase 2: Latent Prepending Logic**

**Objective:** Modify VideoSegmentPrepare to return cached latents for next segment

1. **Enhance `VideoSegmentPrepare.prepare()` method**
    - After determining `current_segment` index:
        - If `current_segment > 0`: load cached latent from previous segment
        - Return loaded latent in `cached_latent` output (currently returns unchanged input)
    - If cache load fails: log warning and proceed with None (graceful degradation)
    - Ensure `overlap_frames` parameter is respected when loading cache

2. **Update return values**
    - Current: `return (..., latent)` where `latent` is pass-through input
    - New: `return (..., cached_latent)` where `cached_latent` is loaded from previous segment

**Deliverables:**

- Modified `VideoSegmentPrepare.prepare()` with cache loading
- Backward compatibility: works even when no cache exists
- Updated docstring/comments explaining the flow

---

### **Phase 3: Latent Caching on Save**

**Objective:** Modify VideoSegmentSave to cache latents after saving video

1. **Enhance use `VideoSegmentSave.save()` method**
    - After successfully saving video segment to disk
    - Before returning, call `_save_latent_cache(latent, project_path, seg_index)`
    - Only cache if `overlap_frames > 0` (allow users to disable feature)
    - Handle None latent gracefully (some workflows may not provide latents)

2. **Add cache status to logging**
    - Log when cache is successfully saved
    - Log when cache save is skipped (None latent, overlap_frames=0)
    - Log any cache save errors without breaking the workflow

**Deliverables:**

- Modified `VideoSegmentSave.save()` with cache saving
- Non-blocking error handling (cache failure doesn't stop video generation)
- Clear logging for debugging

---

### **Phase 4: Video Trimming for Overlap Handling**

**Objective:** Handle extended segment lengths during concatenation

**Context:** When latents are prepended to segments 1+, the generated videos will have `overlap_frames` extra frames at the beginning. These must be trimmed during final concatenation to avoid temporal duplication.

1. **Implement smart concatenation** `_concat_videos_with_overlap()`
    - New function based on existing `_concat_videos()` and `_crossfade_videos()`
    - For each segment (except first):
        - Use ffmpeg `-ss` to skip the first `overlap_frames / fps` seconds
        - Example: `-ss 0.5 -i segment_001.mp4` (if overlap_frames=8, fps=16)
    - Concatenate trimmed segments using ffmpeg concat demuxer or filter

2. **Update `VideoSegmentSave.save()` final concatenation logic**
    - Replace current `_concat_videos()` / `_crossfade_videos()` calls
    - Use new `_concat_videos_with_overlap()` when `overlap_frames > 0`
    - Fallback to existing behavior when `overlap_frames == 0` (backward compatibility)

3. **Handle crossfade + overlap combination**
    - When both `crossfade_frames > 0` and `overlap_frames > 0`:
        - Trim overlap first, then apply crossfade
        - Adjust crossfade offset calculations to account for trimmed frames

**Deliverables:**

- New `_concat_videos_with_overlap()` function
- Updated concatenation logic in `VideoSegmentSave.save()`
- Support for overlap-only, crossfade-only, and combined modes

---

### **Phase 5: Edge Case Handling & Validation**

**Objective:** Ensure robustness across diverse scenarios

1. **Validate latent compatibility**
    - Check if incoming latent from segment N-1 matches expected shape for segment N
    - Handle dimension mismatches (e.g., resolution changes between segments)
    - Log warnings when incompatibilities detected

2. **Handle first segment special case**
    - No cache to load (current_segment == 0)
    - Should work exactly as before (use initial_image or initial_video)

3. **Handle insufficient overlap frames**
    - If segment has only 5 frames but overlap_frames=8, cache all available frames
    - Log this situation for user awareness

4. **Memory management**
    - Ensure latent tensors are properly moved to CPU before saving
    - Clear GPU cache after loading latents if on CUDA
    - Add `torch.cuda.empty_cache()` calls where appropriate

5. **Concurrent execution safety**
    - Add file locking or unique naming to prevent cache corruption
    - Consider timestamp-based cache file names if multiple projects run simultaneously

**Deliverables:**

- Comprehensive input validation
- Edge case tests
- Clear error messages for users

---

### **Phase 6: Testing & Documentation**

**Objective:** Validate implementation and provide clear usage guidelines

1. **Unit Tests** (add to `tests/test_my_custom_nodes.py`)
    - `test_latent_cache_save_load_roundtrip`: Verify save/load preserves tensor data
    - `test_latent_cache_with_video_latent_5d`: Test with shape [1, 4, 32, 64, 64]
    - `test_latent_cache_with_image_latent_4d`: Test with shape [1, 4, 64, 64] (should skip or handle gracefully)
    - `test_latent_cache_missing_file`: Verify graceful handling of nonexistent cache
    - `test_overlap_extraction`: Verify correct frame slicing
    - `test_prepare_node_with_cache`: Mock cache file and verify `VideoSegmentPrepare` loads it
    - `test_save_node_creates_cache`: Mock latent input and verify `VideoSegmentSave` creates cache file

2. **Integration Test**
    - Simulate multi-segment workflow:
        1. Prepare segment 0 (no cache)
        2. Save segment 0 with latent (creates cache)
        3. Prepare segment 1 (loads cache from segment 0)
        4. Save segment 1 (creates cache)
        5. Prepare segment 2 (loads cache from segment 1)
        6. Final concatenation with overlap trimming
    - Verify video durations match expected lengths after trimming

3. **Documentation**
    - Add comprehensive docstrings to all new functions
    - Update `VideoSegmentPrepare` and `VideoSegmentSave` docstrings with examples
    - Create usage guide in `docs/features/04-03-2026-video-segment-temporal-consistency/README.md`:
        - Explain latent caching concept
        - Provide example workflow diagrams
        - Document parameters: `overlap_frames`, `cache_window_size`
        - Explain performance implications (disk I/O, memory usage)
        - Troubleshooting section

4. **Code Review Checklist**
    - [ ] All new code follows Google Python Style Guide
    - [ ] Type hints added to all new functions
    - [ ] Logging added to all critical operations
    - [ ] Error handling covers all failure modes
    - [ ] No breaking changes to existing workflows
    - [ ] Memory management is efficient (no leaks)
    - [ ] Tests achieve >90% code coverage for new code

**Deliverables:**

- Comprehensive test suite
- Usage documentation
- Code review approval

---

## Testing Strategy for Validating the Implementation

### **Unit Testing Approach**

#### **1. Latent Cache Save/Load Tests**

```python
def test_latent_cache_roundtrip():
    """Verify latent tensors are preserved through save/load cycle"""
    # Create mock latent dict with 5D video tensor
    original_latent = {
        "samples": torch.randn(1, 4, 32, 64, 64)  # B, C, F, H, W
    }
    project_path = tempfile.mkdtemp()

    # Save cache
    _save_latent_cache(original_latent, project_path, 0)

    # Load cache
    loaded_latent = _load_latent_cache(project_path, 0)

    # Verify shapes and values match
    assert loaded_latent is not None
    assert torch.allclose(original_latent["samples"], loaded_latent["samples"])
```

#### **2. Overlap Frame Extraction Tests**

```python
def test_overlap_extraction():
    """Verify correct slicing of last N frames"""
    full_latent = {"samples": torch.randn(1, 4, 32, 64, 64)}
    overlap_frames = 8

    extracted = _extract_overlap_latent(full_latent, overlap_frames)

    assert extracted["samples"].shape == (1, 4, 8, 64, 64)
    # Verify extracted frames match the last 8 frames of original
    assert torch.allclose(
        extracted["samples"],
        full_latent["samples"][:, :, -overlap_frames:, :, :]
    )
```

#### **3. Edge Case Tests**

```python
def test_latent_cache_with_none():
    """Verify graceful handling of None latent"""
    result = _save_latent_cache(None, "/tmp/test", 0)
    assert result is False  # Should return False but not crash

def test_latent_cache_insufficient_frames():
    """Handle case where video has fewer frames than overlap_frames"""
    short_latent = {"samples": torch.randn(1, 4, 5, 64, 64)}  # Only 5 frames
    overlap_frames = 8

    extracted = _extract_overlap_latent(short_latent, overlap_frames)
    # Should return all 5 frames instead of crashing
    assert extracted["samples"].shape[2] == 5
```

---

### **Integration Testing Approach**

#### **1. Multi-Segment Workflow Simulation**

```python
def test_multi_segment_workflow():
    """Simulate complete 3-segment video generation with caching"""
    project_name = "test_project"

    # Segment 0: Generate, save, create cache
    prepare_node = VideoSegmentPrepare()
    save_node = VideoSegmentSave()

    # First segment (no cache)
    img, seg_idx, finished, path, cached_latent = prepare_node.prepare(
        project_name=project_name,
        total_seconds=6,
        segment_seconds=2,
        fps=16.0,
        overlap_frames=8,
        ...
    )
    assert seg_idx == 0
    assert cached_latent is None or cached_latent == initial_latent

    # Simulate VAE decode -> video generation -> VAE encode
    mock_video = create_mock_video_tensor(32, 512, 512)  # 32 frames
    mock_latent = {"samples": torch.randn(1, 4, 32, 64, 64)}

    # Save segment 0
    saved_idx, finished, path = save_node.save(
        video=mock_video,
        project_name=project_name,
        latent=mock_latent,
        overlap_frames=8,
        ...
    )

    # Verify cache file created
    project_path = _get_project_path(project_name)
    cache_path = os.path.join(project_path, "latent_cache_seg_000.pt")
    assert os.path.exists(cache_path)

    # Segment 1: Should load cache from segment 0
    img, seg_idx, finished, path, cached_latent = prepare_node.prepare(
        project_name=project_name,
        overlap_frames=8,
        ...
    )
    assert seg_idx == 1
    assert cached_latent is not None
    assert cached_latent["samples"].shape == (1, 4, 8, 64, 64)  # 8 overlap frames
```

#### **2. Concatenation with Trim Validation**

```python
def test_concatenation_with_overlap_trim():
    """Verify final video duration is correct after trimming overlap"""
    # Create 3 mock segments:
    # - Segment 0: 32 frames, 2.0s @ 16fps
    # - Segment 1: 40 frames (32 + 8 overlap), should trim to 2.0s
    # - Segment 2: 40 frames (32 + 8 overlap), should trim to 2.0s

    project_path = setup_mock_segments_with_overlap()

    # Perform concatenation with overlap handling
    final_video = os.path.join(project_path, "final_video.mp4")
    _concat_videos_with_overlap(
        video_files=[...],
        output_path=final_video,
        overlap_frames=8,
        fps=16.0
    )

    # Verify final duration: should be 6.0s (3 segments × 2.0s each)
    duration = _get_video_duration(final_video)
    assert abs(duration - 6.0) < 0.1  # Allow 0.1s tolerance for encoding
```

---

### **Manual Testing Checklist**

1. **Basic Workflow Test**
    - [ ] Create new workflow in ComfyUI
    - [ ] Add VideoSegmentPrepare node
    - [ ] Connect to SVD/CogVideoX model
    - [ ] Add VideoSegmentSave node
    - [ ] Set `overlap_frames=8`, run 3 segments
    - [ ] Verify final video plays smoothly without temporal jumps

2. **Performance Test**
    - [ ] Generate 10-segment video (20 seconds total)
    - [ ] Monitor disk I/O for cache operations
    - [ ] Verify cache files are cleaned up after completion
    - [ ] Check memory usage doesn't leak between segments

3. **Edge Case Manual Tests**
    - [ ] Test with `overlap_frames=0` (should behave like old system)
    - [ ] Test with `crossfade_frames > 0` + `overlap_frames > 0`
    - [ ] Test with very short segments (1 second = 16 frames, overlap=8)
    - [ ] Test workflow interruption and resume (cache persistence)

4. **Compatibility Test**
    - [ ] Load old workflow without latent connections (should work)
    - [ ] Load workflow created with v0.0.1 (backward compatibility)
    - [ ] Test with different video models (SVD, CogVideoX, AnimateDiff)

---

### **Automated CI/CD Integration**

Update `.github/workflows/build-pipeline.yml` to include:

```yaml
- name: Run Video Segment Tests
  run: |
      pytest tests/test_my_custom_nodes.py::test_latent_cache_roundtrip -v
      pytest tests/test_my_custom_nodes.py::test_multi_segment_workflow -v
```

---

## Context Map

### Files to Modify

| File                                            | Purpose                  | Changes Needed                                                                                                                                |
| ----------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/my_custom_nodes/video_segment_extender.py` | Main implementation file | Add latent caching functions, modify `VideoSegmentPrepare.prepare()` and `VideoSegmentSave.save()`, implement `_concat_videos_with_overlap()` |
| `tests/test_my_custom_nodes.py`                 | Unit tests               | Add test functions for latent caching, overlap extraction, multi-segment workflow                                                             |

### Dependencies (may need updates)

| File                           | Relationship                                                                              |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| `src/my_custom_nodes/nodes.py` | Imports `VideoSegmentPrepare`, `VideoSegmentSave`, `LatentShapeDebug` - no changes needed |
| `__init__.py`                  | Re-exports node mappings from nodes.py - no changes needed                                |

### Test Files

| Test                            | Coverage                                                    |
| ------------------------------- | ----------------------------------------------------------- |
| `tests/test_my_custom_nodes.py` | Will test latent caching, overlap extraction, node behavior |
| `test_latent_debug.py`          | Reference for latent shape handling patterns                |

### Reference Patterns

| File                                            | Pattern                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| `test_latent_debug.py`                          | Shows overlap extraction: `full_latent[:, :, -overlap_frames:, :, :]`    |
| `src/my_custom_nodes/video_segment_extender.py` | Existing video tensor handling: `hasattr(video_input, "get_components")` |
| `src/my_custom_nodes/video_segment_extender.py` | Project path management: `_get_project_path()`, `_list_segments()`       |
| `src/my_custom_nodes/video_segment_extender.py` | Subprocess error handling: `_run_subprocess()`                           |

### Risk Assessment

- [ ] **Breaking changes to public API**: MITIGATED - Changes are backward compatible (latent caching is optional, defaults ensure old behavior)
- [x] **Database migrations needed**: N/A - No database, uses file-based caching
- [ ] **Configuration changes required**: NO - All new parameters have sensible defaults
- [x] **Performance impact**: MODERATE - Disk I/O for latent caching (~50-200MB per cache), but parallelizable and cleanup mitigates
- [x] **Memory impact**: LOW - Latents are moved to CPU before saving, cache cleanup prevents accumulation
- [x] **Disk space impact**: MODERATE - Each cache file ~50-200MB depending on resolution, but cleanup keeps only last 2
- [ ] **Compatibility with existing workflows**: HIGH - Must ensure workflows without latent connections still work
- [x] **ffmpeg version dependencies**: EXISTING - Already depends on ffmpeg, new `-ss` flag is widely supported

---

## Relevant Links (Optional)

### Official Documentation

- **ComfyUI Custom Nodes**: https://docs.comfy.org/essentials/custom_node_overview
- **ComfyUI Latent Format**: https://github.com/comfyanonymous/ComfyUI (search for VAE encode/decode nodes)
- **PyTorch torch.save/load**: https://pytorch.org/docs/stable/torch.html#torch.save
- **FFmpeg concat demuxer**: https://trac.ffmpeg.org/wiki/Concatenate
- **FFmpeg seeking**: https://trac.ffmpeg.org/wiki/Seeking

### Video Model References

- **Stable Video Diffusion (SVD)**: https://github.com/Stability-AI/generative-models - Uses latent frames for temporal consistency
- **CogVideoX**: https://github.com/THUDM/CogVideo - Video generation with temporal attention
- **AnimateDiff**: https://github.com/guoyww/AnimateDiff - Motion modules work on latent frames

### Related Projects

- **ComfyUI-VideoHelperSuite**: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite - Reference for video tensor handling
- **ComfyUI-Advanced-ControlNet**: https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet - Shows latent caching patterns

### Technical References

- **Video Frame Interpolation**: https://arxiv.org/abs/2011.06803 - Academic paper on temporal consistency
- **Latent Diffusion Models**: https://arxiv.org/abs/2112.10752 - Theory behind latent space operations

---

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
