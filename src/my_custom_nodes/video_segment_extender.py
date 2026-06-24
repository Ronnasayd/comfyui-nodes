import logging
import os
import shutil
import subprocess
import tempfile
from typing import List

import folder_paths

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
        logger.error(f"{error_message}\nCommand: {' '.join(cmd)}\nError: {result.stderr}")
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


# ============================================================
# Concatenation
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


def _trim_and_concat_videos(video_files, output_path, trim_frames, fps, tmp_dir):
    _validate_positive(fps, "fps")

    if len(video_files) < 2:
        shutil.copy2(video_files[0], output_path)
        return

    trim_duration = trim_frames / fps
    trimmed = [video_files[0]]

    for i, vf in enumerate(video_files[1:], start=1):
        trimmed_path = os.path.join(tmp_dir, f"trimmed_{i}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(trim_duration),
            "-i",
            vf,
            "-c",
            "copy",
            trimmed_path,
        ]
        _run_subprocess(cmd, f"Trim failed for segment {i}")
        trimmed.append(trimmed_path)

    _concat_videos(trimmed, output_path)


# ============================================================
# Nodes
# ============================================================


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
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 64}),
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
        trim_frames=0,
        file_extension="mp4",
    ):
        base = folder_paths.get_output_directory()
        full_dir = os.path.join(base, video_dir)

        if not os.path.exists(full_dir):
            raise RuntimeError(f"Video directory not found: {full_dir}")

        output_filename = f"{output_name}.{file_extension}"
        output_path = os.path.join(full_dir, output_filename)

        sources = sorted(
            [os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.endswith(f".{file_extension}") and f != output_filename]
        )

        logger.info(f"VideoConcatenate: found {len(sources)} source file(s) in {full_dir}")

        if len(sources) == 0:
            logger.warning(f"VideoConcatenate: no .{file_extension} files found in {full_dir}")
            return ("",)

        if len(sources) == 1:
            shutil.copy2(sources[0], output_path)
            logger.info(f"Single file — copied to {output_path}")
            return (output_path,)

        if trim_frames > 0:
            _validate_positive(fps, "fps")
            with tempfile.TemporaryDirectory() as tmp_dir:
                _trim_and_concat_videos(sources, output_path, trim_frames, fps, tmp_dir)
        else:
            _concat_videos(sources, output_path)

        logger.info(f"VideoConcatenate: saved final video to {output_path}")
        return (output_path,)
