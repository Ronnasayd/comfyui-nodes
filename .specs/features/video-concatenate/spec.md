# Feature: VideoConcatenate Node

## Overview

Single ComfyUI custom node that concatenates all video segments saved to a directory into one final video. Triggered by receiving the last segment's VIDEO output (execution ordering), then reads all video files from the specified directory and concatenates them.

**Motivation:** Workflow `wan-14B-1.22.json` chains 3 Wan2.2 subgraph instances, each saving their VIDEO independently via `SaveVideo`. No built-in node concatenates them. This node replaces 3 `SaveVideo` nodes with a single output that handles ordering + concatenation.

---

## Requirements

### Functional

| ID      | Requirement                                                                                           |
| ------- | ----------------------------------------------------------------------------------------------------- |
| REQ-001 | Node accepts a `last_video` (VIDEO) input as execution trigger — signals all prior segments are saved |
| REQ-002 | Node accepts a `video_dir` (STRING) path relative to ComfyUI output directory                         |
| REQ-003 | Node lists and sorts all video files in `video_dir` matching `file_extension`                         |
| REQ-004 | Node concatenates sorted files into a single output file saved inside `video_dir`                     |
| REQ-005 | Node supports optional crossfade transitions between segments                                         |
| REQ-006 | Node must NOT include the output file itself in the concatenation list                                |
| REQ-007 | `output_name` parameter controls the output filename (no extension)                                   |
| REQ-008 | Node is marked `OUTPUT_NODE = True` and returns the absolute output path as STRING                    |
| REQ-009 | Node must be registered in `nodes.py` under `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`    |

### Non-Functional

| ID      | Requirement                                                                             |
| ------- | --------------------------------------------------------------------------------------- |
| REQ-010 | Reuse existing `_concat_videos` and `_crossfade_videos` utilities — no new ffmpeg logic |
| REQ-011 | If directory has 0 or 1 files, handle gracefully (log warning / copy single file)       |
| REQ-012 | `last_video` is passthrough only — not processed, not saved                             |

---

## Inputs

| Name               | Type   | Required | Default                         | Description                                                      |
| ------------------ | ------ | -------- | ------------------------------- | ---------------------------------------------------------------- |
| `last_video`       | VIDEO  | Yes      | —                               | Trigger input; last segment ensures all SaveVideo nodes finished |
| `video_dir`        | STRING | Yes      | `"video/Wan2.2_image_to_video"` | Directory path relative to ComfyUI output dir                    |
| `output_name`      | STRING | Yes      | `"final"`                       | Output filename without extension                                |
| `fps`              | FLOAT  | Yes      | `16.0`                          | FPS; used only when crossfade_frames > 0                         |
| `crossfade_frames` | INT    | No       | `0`                             | Frames for xfade transition; 0 = simple concat                   |
| `file_extension`   | COMBO  | No       | `"mp4"`                         | Extension filter: mp4, webm, avi, mov                            |

## Outputs

| Name          | Type   | Description                                  |
| ------------- | ------ | -------------------------------------------- |
| `output_path` | STRING | Absolute path to the concatenated video file |

---

## Acceptance Criteria

- [ ] Node appears in ComfyUI as "Video Concatenate" under category `MYNodes/Video`
- [ ] Given 3 `.mp4` files in `video_dir`, node produces 1 concatenated `.mp4` at `{video_dir}/{output_name}.mp4`
- [ ] Output file is excluded from input list (no self-inclusion)
- [ ] With `crossfade_frames=8`, transitions are smooth between segments
- [ ] With `crossfade_frames=0`, segments are hard-cut concatenated
- [ ] Node handles 1-file directory without error
- [ ] Node handles empty directory with logged warning and no crash
