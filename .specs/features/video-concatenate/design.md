# Design: VideoConcatenate Node

## Architecture

Single stateless node. No disk state, no project system. Execution model: triggered once after all segments are saved, reads directory, concatenates, returns path.

```
[SaveVideo seg1] ──────────────────────────────────────────────────────┐
[SaveVideo seg2] ──────────────────────────────────────────────────────┤  (parallel, no dep)
[SaveVideo seg3] → last_video ──► [VideoConcatenate]                  │
                                        │                              │
                   video_dir ───────────┤                              │
                   output_name ─────────┤  list+sort files in dir      │
                   fps ─────────────────┤  exclude output_name.ext     │
                   crossfade_frames ────┤  concat or xfade             │
                   file_extension ──────┘                              │
                                        │                              │
                                   output_path (STRING) ◄─────────────┘
```

**Key constraint:** `last_video` input creates an implicit execution dependency in ComfyUI's graph — the node only runs after `SaveVideo seg3` finishes. The VIDEO value itself is ignored.

---

## Component Map

```
video_segment_extender.py
├── [REUSE] _concat_videos(files, output)          ← simple ffmpeg concat
├── [REUSE] _crossfade_videos(files, output, n, fps) ← xfade filter
├── [REUSE] _validate_positive(value, name)
├── [NEW]   VideoConcatenate.INPUT_TYPES()
├── [NEW]   VideoConcatenate.concatenate()
│           ├── resolve full_dir = output_dir / video_dir
│           ├── list + sort files by name
│           ├── exclude output file from list
│           ├── guard: 0 files → warning, return ("",)
│           ├── guard: 1 file → copy to output_path
│           └── dispatch: crossfade_frames > 0 → _crossfade_videos
│                         else               → _concat_videos

nodes.py
├── [NEW] import VideoConcatenate
├── [NEW] NODE_CLASS_MAPPINGS["VideoConcatenate"]
└── [NEW] NODE_DISPLAY_NAME_MAPPINGS["VideoConcatenate"]
```

---

## Data Flow

```
1. ComfyUI executes node (triggered after last SaveVideo)
2. video_dir → joined with folder_paths.get_output_directory()
3. os.listdir(full_dir) filtered by file_extension, sorted
4. output filename = f"{output_name}.{file_extension}"
5. output_path excluded from source list (prevent self-include)
6. if crossfade_frames > 0:
       _crossfade_videos(sources, output_path, crossfade_frames, fps)
   else:
       _concat_videos(sources, output_path)
7. return (output_path,)
```

---

## Key Decisions

| Decision          | Choice                                 | Reason                                                                          |
| ----------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| Trigger mechanism | Accept VIDEO input, ignore value       | ComfyUI graph dependency — no better primitive for "wait for all"               |
| Directory scope   | Relative to ComfyUI output dir         | Matches SaveVideo convention; user sees same path string                        |
| Output location   | Inside `video_dir`                     | Keeps all segment files + final together                                        |
| Self-exclusion    | Filter by exact filename match         | Prevents re-running from including prior output                                 |
| Return type       | STRING (path) only                     | VIDEO construction API is internal to comfy-core; STRING is safe and composable |
| ffmpeg reuse      | `_concat_videos` / `_crossfade_videos` | Already tested, avoids duplication                                              |

---

## File Sort Order

`os.listdir` + `.sort()` = lexicographic order. SaveVideo names segments with auto-incrementing suffixes (e.g. `Wan2.2_image_to_video_00001_.mp4`). Lexicographic sort produces correct temporal order. No custom sort needed.

---

## Edge Cases

| Case                                  | Behavior                                                           |
| ------------------------------------- | ------------------------------------------------------------------ |
| 0 matching files                      | Log warning, return `("",)`                                        |
| 1 matching file                       | `shutil.copy2(file, output_path)`, return path                     |
| Output file already exists            | `_concat_videos` uses `ffmpeg -y` (overwrite)                      |
| `video_dir` doesn't exist             | `os.listdir` raises → propagate as RuntimeError with clear message |
| `crossfade_frames` ≥ segment duration | `_crossfade_videos` will produce ffmpeg error → propagate          |
