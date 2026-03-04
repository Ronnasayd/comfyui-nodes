import logging
import math
import os
import shutil
import subprocess
import time
from typing import List

import folder_paths
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# Subprocess Utilities (Safe Execution)
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
        raise RuntimeError(error_message)

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
        return float(result.stdout.strip())
    except Exception:
        raise RuntimeError(f"Invalid duration returned for {video_path}")


# ============================================================
# Shared Helpers
# ============================================================


def _get_project_path(project_name: str) -> str:
    base_output = folder_paths.get_output_directory()
    project_path = os.path.join(base_output, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path


def _count_segments(project_path: str) -> int:
    return len(
        [
            f
            for f in os.listdir(project_path)
            if f.startswith("segment_") and f.endswith(".mp4")
        ]
    )


def _extract_last_frame(video_path: str, project_path: str) -> torch.Tensor:
    temp_frame = os.path.join(project_path, "last_frame.png")

    cmd = [
        "ffmpeg",
        "-sseof",
        "-0.1",
        "-i",
        video_path,
        "-update",
        "1",
        "-q:v",
        "1",
        "-y",
        temp_frame,
    ]

    _run_subprocess(cmd, "Failed extracting last frame")

    if not os.path.exists(temp_frame):
        raise RuntimeError("Last frame not created")

    img = Image.open(temp_frame).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr).float()


def _extract_blended_frame(
    video_path: str,
    project_path: str,
    blend_count: int,
    offset_percent: float,
) -> torch.Tensor:
    duration = _get_video_duration(video_path)
    start_time = duration * (offset_percent / 100.0)

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
        "-q:v",
        "1",
        "-y",
        frame_pattern,
    ]

    _run_subprocess(cmd, "Failed extracting blended frames")

    frames = []
    for i in range(1, blend_count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            frames.append(np.array(img).astype(np.float32) / 255.0)

    if not frames:
        raise RuntimeError("No blended frames extracted")

    avg_frame = np.mean(frames, axis=0)
    avg_frame = np.expand_dims(avg_frame, 0)
    return torch.from_numpy(avg_frame).float()


def _save_video_tensor(video_input, output_path: str, fps: float) -> None:
    if hasattr(video_input, "get_components"):
        tensor = video_input.get_components().images.detach().cpu()
    elif isinstance(video_input, torch.Tensor):
        tensor = video_input.detach().cpu()
    else:
        raise TypeError(f"Unsupported VIDEO type: {type(video_input)}")

    if tensor.dim() == 5:
        tensor = tensor[0]

    if tensor.dim() != 4:
        raise ValueError(f"Unexpected VIDEO format: {tensor.shape}")

    frames = (tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
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
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    for frame in frames:
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed while saving video")


# ============================================================
# Latent Utilities
# ============================================================


def _extract_latent_overlap(latent_dict: dict, overlap_frames: int) -> dict:
    if not latent_dict or overlap_frames <= 0:
        return latent_dict

    samples = latent_dict.get("samples")
    if samples is None or samples.dim() < 5:
        return latent_dict

    temporal_dim = 2  # padrão Comfy [B, C, F, H, W]

    num_frames = samples.shape[temporal_dim]
    if overlap_frames >= num_frames:
        return latent_dict

    slices = [slice(None)] * samples.dim()
    slices[temporal_dim] = slice(-overlap_frames, None)

    new_samples = samples[tuple(slices)].contiguous()
    return {"samples": new_samples}


# ============================================================
# Video Concatenation (Correct Crossfade)
# ============================================================


def _crossfade_videos(
    video_files: List[str],
    output_path: str,
    crossfade_frames: int,
    fps: float,
):
    if len(video_files) < 2:
        shutil.copy2(video_files[0], output_path)
        return

    crossfade_duration = crossfade_frames / fps

    inputs = []
    for vf in video_files:
        inputs.extend(["-i", vf])

    filter_complex_parts = []
    cumulative_offset = 0
    current_label = "[0:v]"

    for i in range(1, len(video_files)):
        prev_duration = _get_video_duration(video_files[i - 1])
        cumulative_offset += prev_duration - crossfade_duration

        next_label = f"[v{i}]"

        filter_complex_parts.append(
            f"{current_label}[{i}:v]"
            f"xfade=transition=fade:duration={crossfade_duration}:offset={cumulative_offset}"
            f"{next_label}"
        )

        current_label = next_label

    filter_complex = ";".join(filter_complex_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            current_label,
            "-vcodec",
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
    def IS_CHANGED(cls, **_kwargs):
        return time.time()

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
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        cache_dir = os.path.join(project_path, "latent_cache")
        os.makedirs(cache_dir, exist_ok=True)

        cached_latent = None
        if current_segment > 0:
            cache_path = os.path.join(
                cache_dir, f"segment_{current_segment - 1:03d}.pt"
            )
            if os.path.exists(cache_path):
                cached_latent = torch.load(cache_path, map_location="cpu")

        if current_segment >= max_segments:
            final_path = os.path.join(project_path, "final_video.mp4")
            return (
                initial_image or torch.zeros((1, 512, 512, 3)),
                current_segment,
                True,
                final_path if os.path.exists(final_path) else "",
                cached_latent,
            )

        if current_segment == 0:
            if initial_video is not None:
                init_path = os.path.join(project_path, "initial_video.mp4")
                if not os.path.exists(init_path):
                    _save_video_tensor(initial_video, init_path, fps)

                if frame_blend_count > 1:
                    frame = _extract_blended_frame(
                        init_path,
                        project_path,
                        frame_blend_count,
                        frame_offset_percent,
                    )
                else:
                    frame = _extract_last_frame(init_path, project_path)
                return (frame, 0, False, "", cached_latent)

            if initial_image is not None:
                return (initial_image, 0, False, "", cached_latent)

            return (torch.zeros((1, 512, 512, 3)), 0, False, "", cached_latent)

        last_segment = os.path.join(
            project_path, f"segment_{current_segment - 1:03d}.mp4"
        )

        if frame_blend_count > 1:
            frame = _extract_blended_frame(
                last_segment,
                project_path,
                frame_blend_count,
                frame_offset_percent,
            )
        else:
            frame = _extract_last_frame(last_segment, project_path)

        return (frame, current_segment, False, "", cached_latent)


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
        current_segment = _count_segments(project_path)

        seg_path = os.path.join(project_path, f"segment_{current_segment:03d}.mp4")
        _save_video_tensor(video, seg_path, fps)

        if latent is not None:
            cache_dir = os.path.join(project_path, "latent_cache")
            os.makedirs(cache_dir, exist_ok=True)

            cache_path = os.path.join(cache_dir, f"segment_{current_segment:03d}.pt")
            overlap_latent = _extract_latent_overlap(latent, overlap_frames)
            torch.save(overlap_latent, cache_path)

            existing = sorted(os.listdir(cache_dir))
            if len(existing) > 4:
                for old in existing[:-4]:
                    os.remove(os.path.join(cache_dir, old))

        current_segment += 1
        torch.cuda.empty_cache()

        if current_segment >= max_segments:
            final_path = os.path.join(project_path, "final_video.mp4")

            video_files = sorted(
                [
                    os.path.join(project_path, f)
                    for f in os.listdir(project_path)
                    if f.startswith("segment_") and f.endswith(".mp4")
                ]
            )

            if crossfade_frames > 0:
                _crossfade_videos(video_files, final_path, crossfade_frames, fps)
            else:
                shutil.copy2(video_files[0], final_path)

            return (current_segment, True, final_path)

        return (current_segment, False, "")


class LatentShapeDebug:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("shape_info", "latent")

    FUNCTION = "debug_shape"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/VideoSegment"

    def debug_shape(self, latent):
        if latent is None:
            return ("Latent is None", latent)

        samples = latent.get("samples")
        if samples is None:
            return ("No samples key", latent)

        shape = samples.shape
        mem_mb = samples.numel() * samples.element_size() / (1024**2)

        info = (
            f"Shape: {shape}\n"
            f"Dimensions: {len(shape)}\n"
            f"Total elements: {samples.numel():,}\n"
            f"Memory: {mem_mb:.2f} MB"
        )

        logger.info(info.replace("\n", " | "))
        return (info, latent)
