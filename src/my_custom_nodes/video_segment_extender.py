import logging
import math
import os
import shutil
import subprocess
import tempfile
from typing import List

import folder_paths
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Environment Validation
# ============================================================


def _ensure_binary_exists(binary: str):
    if shutil.which(binary) is None:
        raise EnvironmentError(f"{binary} not found in PATH.")


_ensure_binary_exists("ffmpeg")
_ensure_binary_exists("ffprobe")


# ============================================================
# Subprocess Utilities
# ============================================================


def _run_subprocess(cmd: List[str], error_message: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(
            f"{error_message}\nCommand: {' '.join(cmd)}\nError: {result.stderr}"
        )
        raise RuntimeError(f"{error_message}\n{result.stderr}")

    return result


def _get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = _run_subprocess(cmd, f"Failed to get duration for {video_path}")

    try:
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise ValueError
        return duration
    except Exception:
        raise RuntimeError(f"Invalid duration returned for {video_path}")


# ============================================================
# Shared Helpers
# ============================================================


def _validate_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _get_project_path(project_name: str) -> str:
    base_output = folder_paths.get_output_directory()
    project_path = os.path.join(base_output, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path


def _list_segments(project_path: str) -> List[str]:
    segments = [
        f
        for f in os.listdir(project_path)
        if f.startswith("segment_") and f.endswith(".mp4")
    ]
    segments.sort()
    return segments


def _count_segments(project_path: str) -> int:
    return len(_list_segments(project_path))


def _extract_last_frame(video_path: str, project_path: str) -> torch.Tensor:
    temp_frame = os.path.join(project_path, "last_frame.png")

    cmd = [
        "ffmpeg",
        "-sseof",
        "-0.1",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-y",
        temp_frame,
    ]

    _run_subprocess(cmd, "Failed extracting last frame")

    if not os.path.exists(temp_frame):
        raise RuntimeError("Last frame not created")

    with Image.open(temp_frame) as img:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr).float()


def _extract_blended_frame(video_path, project_path, blend_count, offset_percent):
    duration = _get_video_duration(video_path)

    offset_percent = max(0.0, min(offset_percent, 99.0))
    start_time = duration * (offset_percent / 100.0)
    start_time = min(start_time, max(0.0, duration - 0.05))

    frames_dir = os.path.join(project_path, "blend_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    frame_pattern = os.path.join(frames_dir, "frame_%03d.png")

    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-vframes",
        str(blend_count),
        "-y",
        frame_pattern,
    ]

    _run_subprocess(cmd, "Failed extracting blended frames")

    frames = []
    for i in range(1, blend_count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        if os.path.exists(frame_path):
            with Image.open(frame_path) as img:
                frames.append(np.array(img.convert("RGB")).astype(np.float32) / 255.0)

    if not frames:
        raise RuntimeError("No blended frames extracted")

    avg_frame = np.mean(frames, axis=0)
    avg_frame = np.expand_dims(avg_frame, 0)
    return torch.from_numpy(avg_frame).float()


# ============================================================
# Video Saving
# ============================================================


def _normalize_video_tensor(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 5:
        tensor = tensor[0]

    if tensor.dim() != 4:
        raise ValueError(f"Unexpected VIDEO format: {tensor.shape}")

    # Support both [F,H,W,3] and [F,3,H,W]
    if tensor.shape[-1] == 3:
        frames = tensor
    elif tensor.shape[1] == 3:
        frames = tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError("Unsupported channel format")

    return (frames.numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _save_video_tensor(video_input, output_path: str, fps: float):
    _validate_positive(fps, "fps")

    if hasattr(video_input, "get_components"):
        tensor = video_input.get_components().images.detach().cpu()
    elif isinstance(video_input, torch.Tensor):
        tensor = video_input.detach().cpu()
    else:
        raise TypeError(f"Unsupported VIDEO type: {type(video_input)}")

    frames = _normalize_video_tensor(tensor)
    h, w = frames.shape[1], frames.shape[2]

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for frame in frames:
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed while saving video")


# ============================================================
# Concatenation (FIXED)
# ============================================================


def _concat_videos(video_files: List[str], output_path: str):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for vf in video_files:
            f.write(f"file '{vf}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        output_path,
    ]

    _run_subprocess(cmd, "Concatenation failed")
    os.remove(list_path)


def _crossfade_videos(video_files, output_path, crossfade_frames, fps):
    _validate_positive(fps, "fps")

    if len(video_files) < 2:
        shutil.copy2(video_files[0], output_path)
        return

    crossfade_duration = crossfade_frames / fps

    inputs = []
    for vf in video_files:
        inputs.extend(["-i", vf])

    filter_parts = []
    offset = 0
    current = "[0:v]"

    for i in range(1, len(video_files)):
        prev_duration = _get_video_duration(video_files[i - 1])
        offset += prev_duration - crossfade_duration
        next_label = f"[v{i}]"

        filter_parts.append(
            f"{current}[{i}:v]"
            f"xfade=transition=fade:duration={crossfade_duration}:offset={offset}"
            f"{next_label}"
        )

        current = next_label

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            current,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
    )

    _run_subprocess(cmd, "Crossfade failed")


# ============================================================
# Nodes
# ============================================================


class VideoSegmentPrepare:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "wan_project"}),
                "total_seconds": ("INT", {"default": 6}),
                "segment_seconds": ("INT", {"default": 2}),
                "fps": ("FLOAT", {"default": 16.0}),
                "frame_blend_count": ("INT", {"default": 4}),
                "frame_offset_percent": ("FLOAT", {"default": 90.0}),
            },
            "optional": {
                "initial_image": ("IMAGE",),
                "initial_video": ("VIDEO",),
                "cache_window_size": ("INT", {"default": 4}),
                "overlap_frames": ("INT", {"default": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "STRING", "LATENT")
    RETURN_NAMES = (
        "next_image",
        "current_segment",
        "finished",
        "final_video_path",
        "cached_latent",
    )

    FUNCTION = "prepare"
    CATEGORY = "MYNodes/VideoSegment"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash(tuple(sorted(kwargs.items())))

    def prepare(
        self,
        project_name,
        total_seconds,
        segment_seconds,
        fps,
        frame_blend_count,
        frame_offset_percent,
        initial_image=None,
        initial_video=None,
        cache_window_size=4,
        overlap_frames=8,
    ):

        _validate_positive(segment_seconds, "segment_seconds")
        _validate_positive(total_seconds, "total_seconds")

        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        if current_segment >= max_segments:
            final_path = os.path.join(project_path, "final_video.mp4")
            return (
                initial_image,
                current_segment,
                True,
                final_path if os.path.exists(final_path) else "",
                None,
            )

        if current_segment == 0:
            if initial_image is not None:
                return (initial_image, 0, False, "", None)

        last_segment = None
        if current_segment > 0:
            segments = _list_segments(project_path)
            last_segment = os.path.join(project_path, segments[-1])

        if last_segment:
            frame = _extract_blended_frame(
                last_segment, project_path, frame_blend_count, frame_offset_percent
            )
            return (frame, current_segment, False, "", None)

        return (torch.zeros((1, 512, 512, 3)), 0, False, "", None)


class VideoSegmentSave:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "project_name": ("STRING", {"default": "wan_project"}),
                "total_seconds": ("INT", {"default": 6}),
                "segment_seconds": ("INT", {"default": 2}),
                "fps": ("FLOAT", {"default": 16.0}),
                "frame_offset_percent": ("FLOAT", {"default": 90.0}),
                "remove_duplicate_frames": ("INT", {"default": 1}),
            },
            "optional": {
                "latent": ("LATENT",),
                "overlap_frames": ("INT", {"default": 8}),
                "crossfade_frames": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("saved_segment", "finished", "final_video_path")

    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/VideoSegment"

    def save(
        self,
        video,
        project_name,
        total_seconds,
        segment_seconds,
        fps,
        frame_offset_percent,
        remove_duplicate_frames,
        latent=None,
        overlap_frames=8,
        crossfade_frames=0,
    ):

        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        segments = _list_segments(project_path)

        seg_index = len(segments)
        seg_path = os.path.join(project_path, f"segment_{seg_index:03d}.mp4")

        if os.path.exists(seg_path):
            raise RuntimeError("Segment overwrite prevented")

        _save_video_tensor(video, seg_path, fps)

        seg_index += 1

        if seg_index >= max_segments:
            final_path = os.path.join(project_path, "final_video.mp4")
            video_files = [
                os.path.join(project_path, f) for f in _list_segments(project_path)
            ]

            if crossfade_frames > 0:
                _crossfade_videos(video_files, final_path, crossfade_frames, fps)
            else:
                _concat_videos(video_files, final_path)

            return (seg_index, True, final_path)

        return (seg_index, False, "")
