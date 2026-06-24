# Tasks: VideoConcatenate Node

## Summary

| ID   | Task                                                        | Depends On | Status  |
| ---- | ----------------------------------------------------------- | ---------- | ------- |
| T-01 | Add `VideoConcatenate` class to `video_segment_extender.py` | —          | pending |
| T-02 | Register node in `nodes.py`                                 | T-01       | pending |
| T-03 | Verify in ComfyUI + end-to-end test                         | T-02       | pending |

---

## T-01 — Add `VideoConcatenate` class

**Where:** `src/my_custom_nodes/video_segment_extender.py` — append after `VaceControlPrepare` class (line ~1967), before any trailing code.

**What:**

```python
class VideoConcatenate:
    """
    Concatenate all video segments in a directory into one final video.
    Connect the last segment's VIDEO output to `last_video` to enforce
    execution ordering — all SaveVideo nodes must finish before this runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "last_video": ("VIDEO",),
                "video_dir": ("STRING", {"default": "video/Wan2.2_image_to_video"}),
                "output_name": ("STRING", {"default": "final"}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "crossfade_frames": ("INT", {"default": 0, "min": 0, "max": 32}),
                "file_extension": (["mp4", "webm", "avi", "mov"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "concatenate"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/Video"

    def concatenate(
        self,
        last_video,
        video_dir,
        output_name,
        fps,
        crossfade_frames=0,
        file_extension="mp4",
    ):
        base = folder_paths.get_output_directory()
        full_dir = os.path.join(base, video_dir)

        if not os.path.exists(full_dir):
            raise RuntimeError(f"Video directory not found: {full_dir}")

        output_filename = f"{output_name}.{file_extension}"
        output_path = os.path.join(full_dir, output_filename)

        sources = sorted([
            os.path.join(full_dir, f)
            for f in os.listdir(full_dir)
            if f.endswith(f".{file_extension}") and f != output_filename
        ])

        logger.info(f"VideoConcatenate: found {len(sources)} source file(s) in {full_dir}")

        if len(sources) == 0:
            logger.warning(f"VideoConcatenate: no .{file_extension} files found in {full_dir}")
            return ("",)

        if len(sources) == 1:
            shutil.copy2(sources[0], output_path)
            logger.info(f"Single file — copied to {output_path}")
            return (output_path,)

        if crossfade_frames > 0:
            _validate_positive(fps, "fps")
            _crossfade_videos(sources, output_path, crossfade_frames, fps)
        else:
            _concat_videos(sources, output_path)

        logger.info(f"VideoConcatenate: saved final video to {output_path}")
        return (output_path,)
```

**Reuses:** `_concat_videos`, `_crossfade_videos`, `_validate_positive`, `folder_paths`, `os`, `shutil`, `logger` — all already imported/defined in the file.

**Done when:**

- Class exists in file with correct `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `OUTPUT_NODE`, `CATEGORY`
- `concatenate()` handles 0, 1, N files without crashing
- Self-exclusion logic: `output_filename` filtered from `sources`

---

## T-02 — Register in `nodes.py`

**Where:** `src/my_custom_nodes/nodes.py`

**What:** Two changes:

1. Add to import block:

```python
from .video_segment_extender import (
    ...
    VideoConcatenate,   # add this
)
```

2. Add to `NODE_CLASS_MAPPINGS`:

```python
"VideoConcatenate": VideoConcatenate,
```

3. Add to `NODE_DISPLAY_NAME_MAPPINGS`:

```python
"VideoConcatenate": "Video Concatenate",
```

**Done when:** Both dicts contain the entry, import succeeds.

---

## T-03 — Verification

**Manual steps:**

1. Restart ComfyUI
2. Search node panel for "Video Concatenate" — must appear under `MYNodes/Video`
3. In `wan-14B-1.22.json`:
    - Remove the 3 `SaveVideo` nodes (or mute them)
    - Connect `Segment3[616]` VIDEO output → `VideoConcatenate.last_video`
    - Set `video_dir = "video/Wan2.2_image_to_video"`
    - Set `output_name = "final"`, `fps = 16`
4. Run workflow
5. Check `{comfyui_output}/video/Wan2.2_image_to_video/final.mp4` exists and plays all 3 segments in order

**Edge case tests:**

- Empty dir: manually test `video_dir` pointing to empty folder → no crash, returns `""`
- Single file: one `.mp4` in dir → copies to `final.mp4`
- Crossfade: set `crossfade_frames=8` → transitions visible between segments

**Done when:** `final.mp4` plays correctly, ~18s duration, no duplicate frames at boundaries.
