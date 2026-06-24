# Plan: VideoConcatenate Node

## Context

Workflow `wan-14B-1.22.json` has 3 subgraphs, each saving its VIDEO to the same directory via `SaveVideo` (path: `"video/Wan2.2_image_to_video"`). User wants a single node that:

1. Receives the **last VIDEO** output (execution trigger — signals all segments are done)
2. Receives the **directory path** where all segments were saved
3. Lists + sorts all video files in that directory
4. Concatenates them into one final video

Existing utilities to reuse from `video_segment_extender.py`:

- `_concat_videos(files, output)` — simple concat
- `_crossfade_videos(files, output, frames, fps)` — with crossfade

---

## New Node: `VideoConcatenate`

**File**: `src/my_custom_nodes/video_segment_extender.py` — append in the Nodes section.

### Inputs

| Name               | Type                            | Notes                                                                                                |
| ------------------ | ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `last_video`       | VIDEO                           | Trigger — ensures all SaveVideo nodes finished before this runs                                      |
| `video_dir`        | STRING                          | Directory containing segments, relative to ComfyUI output dir (e.g. `"video/Wan2.2_image_to_video"`) |
| `output_name`      | STRING                          | Output filename without extension (e.g. `"final"`)                                                   |
| `fps`              | FLOAT                           | Needed for crossfade duration calculation                                                            |
| `crossfade_frames` | INT (optional)                  | 0 = simple concat, >0 = xfade                                                                        |
| `file_extension`   | `["mp4", "webm", "avi", "mov"]` | Filter extension for segment files                                                                   |

### Outputs

`("STRING",)` → `output_path` — absolute path to the concatenated file. `OUTPUT_NODE = True`.

### Logic

```python
def concatenate(self, last_video, video_dir, output_name, fps, crossfade_frames, file_extension):
    base = folder_paths.get_output_directory()
    full_dir = os.path.join(base, video_dir)

    files = sorted([
        os.path.join(full_dir, f)
        for f in os.listdir(full_dir)
        if f.endswith(f".{file_extension}")
    ])

    output_path = os.path.join(full_dir, f"{output_name}.{file_extension}")

    if crossfade_frames > 0:
        _crossfade_videos(files, output_path, crossfade_frames, fps)
    else:
        _concat_videos(files, output_path)

    return (output_path,)
```

---

## Files to Modify

| File                                            | Change                                        |
| ----------------------------------------------- | --------------------------------------------- |
| `src/my_custom_nodes/video_segment_extender.py` | Add `VideoConcatenate` class in Nodes section |
| `src/my_custom_nodes/nodes.py`                  | Import + register in both mappings            |

---

## Workflow Integration

In `wan-14B-1.22.json`:

- Connect `Segment3[616]` VIDEO output → `VideoConcatenate.last_video`
- Set `video_dir = "video/Wan2.2_image_to_video"`
- Set `output_name = "final"`
- Set `fps = 16`

---

## Verification

1. Restart ComfyUI, search "Video Concatenate"
2. Run workflow, check `output/video/Wan2.2_image_to_video/final.mp4` exists
3. Verify all 3 segments appear in order
4. Test `crossfade_frames=8` for smooth transitions
