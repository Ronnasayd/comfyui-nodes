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
    """Get the project directory path, creating it if it doesn't exist."""
    base_output = folder_paths.get_output_directory()
    project_path = os.path.join(base_output, project_name)
    os.makedirs(project_path, exist_ok=True)

    logger.debug(
        f"Project path: base={base_output}, "
        f"project={project_name}, "
        f"full_path={project_path}"
    )

    return project_path


def _list_segments(project_path: str) -> List[str]:
    """List all segment video files in the project directory."""
    if not os.path.exists(project_path):
        logger.warning(f"Project path does not exist: {project_path}")
        return []

    try:
        all_files = os.listdir(project_path)
        segments = [
            f for f in all_files if f.startswith("segment_") and f.endswith(".mp4")
        ]
        segments.sort()

        logger.debug(f"Found {len(segments)} segment(s) in {project_path}: {segments}")
        return segments
    except Exception as e:
        logger.error(f"Error listing segments in {project_path}: {e}", exc_info=True)
        return []


def _count_segments(project_path: str) -> int:
    """Count the number of saved video segments."""
    count = len(_list_segments(project_path))
    logger.info(f"Segment count for {os.path.basename(project_path)}: {count}")
    return count


# ============================================================
# Latent Caching Infrastructure
# ============================================================


def _save_latent_cache(
    latent_dict: dict,
    project_path: str,
    segment_index: int,
    overlap_frames: int,
) -> bool:
    """
    Save the last N frames of a latent tensor to disk for temporal consistency.

    Args:
        latent_dict: Latent dictionary with 'samples' key containing torch.Tensor
        project_path: Project directory path
        segment_index: Current segment index for naming the cache file
        overlap_frames: Number of overlapping frames to extract and cache

    Returns:
        True if cache was successfully saved, False otherwise

    The cache file will be saved as:
        {project_path}/latent_cache_seg_{index:03d}.pt

    For 5D video latents [B, C, F, H, W], extracts the last 'overlap_frames' frames.
    For 4D image latents [B, C, H, W], the function logs a warning and returns False.
    """
    # Validate inputs
    if latent_dict is None:
        logger.debug(
            f"Skipping latent cache save for segment {segment_index}: latent is None"
        )
        return False

    if overlap_frames <= 0:
        logger.debug(
            f"Skipping latent cache save for segment {segment_index}: "
            f"overlap_frames={overlap_frames}"
        )
        return False

    if not isinstance(latent_dict, dict):
        logger.warning(
            f"Cannot save latent cache for segment {segment_index}: "
            f"latent is not a dict (type={type(latent_dict)})"
        )
        return False

    samples = latent_dict.get("samples")
    if samples is None:
        logger.warning(
            f"Cannot save latent cache for segment {segment_index}: "
            f"no 'samples' key in latent dict"
        )
        return False

    if not isinstance(samples, torch.Tensor):
        logger.warning(
            f"Cannot save latent cache for segment {segment_index}: "
            f"'samples' is not a tensor (type={type(samples)})"
        )
        return False

    # Extract overlap frames based on tensor dimensionality
    shape = samples.shape
    dims = len(shape)

    if dims == 5:
        # Video latent: [B, C, F, H, W]
        _, _, f, _, _ = shape
        actual_overlap = min(overlap_frames, f)

        if actual_overlap < overlap_frames:
            logger.warning(
                f"Requested overlap_frames={overlap_frames} but segment only has "
                f"{f} frames. Caching all {actual_overlap} available frames."
            )

        # Extract last N frames
        overlap_latent = samples[:, :, -actual_overlap:, :, :].detach().cpu()

        logger.info(
            f"Extracting {actual_overlap} overlap frames from video latent "
            f"for segment {segment_index} (shape: {list(overlap_latent.shape)})"
        )

    elif dims == 4:
        # Image latent: [B, C, H, W] - no temporal dimension
        logger.warning(
            f"Cannot cache overlap frames for segment {segment_index}: "
            f"latent is 4D image format (shape={list(shape)}), no temporal dimension"
        )
        return False

    else:
        logger.warning(
            f"Cannot cache overlap frames for segment {segment_index}: "
            f"unexpected tensor dimensions={dims} (shape={list(shape)})"
        )
        return False

    # Prepare cache data with metadata
    cache_data = {
        "samples": overlap_latent,
        "metadata": {
            "segment_index": segment_index,
            "overlap_frames": actual_overlap,
            "original_shape": list(shape),
            "cached_shape": list(overlap_latent.shape),
            "dtype": str(overlap_latent.dtype),
            "device": str(overlap_latent.device),
        },
    }

    # Save to disk
    cache_path = os.path.join(project_path, f"latent_cache_seg_{segment_index:03d}.pt")

    try:
        torch.save(cache_data, cache_path)
        mem_mb = overlap_latent.numel() * overlap_latent.element_size() / (1024**2)
        logger.info(
            f"Successfully saved latent cache for segment {segment_index} "
            f"({mem_mb:.2f} MB) to {cache_path}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to save latent cache for segment {segment_index}: {e}",
            exc_info=True,
        )
        return False


def _load_latent_cache(project_path: str, segment_index: int) -> dict | None:
    """
    Load cached latent frames from disk.

    Args:
        project_path: Project directory path
        segment_index: Segment index to load cache from

    Returns:
        Latent dict with 'samples' key containing cached frames, or None if not found

    The cache is expected at:
        {project_path}/latent_cache_seg_{index:03d}.pt
    """
    cache_path = os.path.join(project_path, f"latent_cache_seg_{segment_index:03d}.pt")

    if not os.path.exists(cache_path):
        logger.debug(
            f"No latent cache found for segment {segment_index} at {cache_path}"
        )
        return None

    try:
        cache_data = torch.load(cache_path, map_location="cpu")

        if not isinstance(cache_data, dict):
            logger.warning(
                f"Invalid latent cache for segment {segment_index}: "
                f"expected dict, got {type(cache_data)}"
            )
            return None

        samples = cache_data.get("samples")
        if samples is None or not isinstance(samples, torch.Tensor):
            logger.warning(
                f"Invalid latent cache for segment {segment_index}: "
                f"missing or invalid 'samples' tensor"
            )
            return None

        metadata = cache_data.get("metadata", {})
        logger.info(
            f"Loaded latent cache for segment {segment_index}: "
            f"shape={list(samples.shape)}, "
            f"overlap_frames={metadata.get('overlap_frames', 'unknown')}"
        )

        # Return in standard latent dict format
        return {"samples": samples}

    except Exception as e:
        logger.error(
            f"Failed to load latent cache for segment {segment_index}: {e}",
            exc_info=True,
        )
        return None


def _cleanup_old_caches(project_path: str, keep_last_n: int = 2) -> None:
    """
    Remove old latent cache files, keeping only the most recent N.

    Args:
        project_path: Project directory path
        keep_last_n: Number of most recent cache files to keep (default: 2)

    This helps manage disk space by removing obsolete caches that are no longer needed.
    """
    try:
        # Find all cache files
        cache_files = [
            f
            for f in os.listdir(project_path)
            if f.startswith("latent_cache_seg_") and f.endswith(".pt")
        ]

        if len(cache_files) <= keep_last_n:
            return

        # Sort by segment index (filename format: latent_cache_seg_NNN.pt)
        cache_files.sort()

        # Remove oldest caches
        files_to_remove = cache_files[:-keep_last_n]
        for cache_file in files_to_remove:
            cache_path = os.path.join(project_path, cache_file)
            os.remove(cache_path)
            logger.debug(f"Removed old latent cache: {cache_file}")

        if files_to_remove:
            logger.info(
                f"Cleaned up {len(files_to_remove)} old latent cache(s), "
                f"kept last {keep_last_n}"
            )

    except Exception as e:
        logger.warning(f"Failed to cleanup old latent caches: {e}", exc_info=True)


def _extract_last_frame(video_path: str, project_path: str) -> torch.Tensor:
    """Extract the last frame from a video file."""

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise RuntimeError(f"Video file not found: {video_path}")

    temp_frame = os.path.join(project_path, "last_frame.png")

    logger.debug(f"Extracting last frame from {os.path.basename(video_path)}")

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
        logger.error(f"Last frame file not created: {temp_frame}")
        raise RuntimeError("Last frame not created")

    logger.debug(f"Loading last frame from {temp_frame}")
    with Image.open(temp_frame) as img:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr).float()


def _extract_blended_frame(video_path, project_path, blend_count, offset_percent):
    """Extract and blend multiple frames from a video for smooth transitions."""

    # Validate video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise RuntimeError(f"Video file not found: {video_path}")

    # Get video info
    duration = _get_video_duration(video_path)
    file_size = os.path.getsize(video_path)

    logger.info(
        f"Extracting blended frames from {os.path.basename(video_path)}: "
        f"duration={duration:.2f}s, size={file_size / 1024:.1f}KB, "
        f"blend_count={blend_count}, offset={offset_percent}%"
    )

    # Calculate start time for extraction
    offset_percent = max(0.0, min(offset_percent, 99.0))
    start_time = duration * (offset_percent / 100.0)

    # Ensure we don't try to extract beyond the video duration
    # Leave at least 0.1s margin to ensure we can extract frames
    max_start_time = max(0.0, duration - 0.2)
    start_time = min(start_time, max_start_time)

    logger.debug(
        f"Calculated start_time={start_time:.3f}s "
        f"(duration={duration:.3f}s, offset={offset_percent}%)"
    )

    frames_dir = os.path.join(project_path, "blend_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Clean up old frames
    old_frames = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    for f in old_frames:
        os.remove(os.path.join(frames_dir, f))
    logger.debug(f"Cleaned {len(old_frames)} old frames from {frames_dir}")

    frame_pattern = os.path.join(frames_dir, "frame_%03d.png")

    # Move -ss AFTER -i for better compatibility
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-ss",
        str(start_time),
        "-vframes",
        str(blend_count),
        "-q:v",
        "2",  # High quality
        "-y",
        frame_pattern,
    ]

    logger.info(f"Running ffmpeg: {' '.join(cmd)}")

    try:
        result = _run_subprocess(cmd, "Failed extracting blended frames")
        logger.debug(
            f"ffmpeg stdout: {result.stdout[:200] if result.stdout else 'empty'}"
        )
        logger.debug(
            f"ffmpeg stderr: {result.stderr[:200] if result.stderr else 'empty'}"
        )
    except Exception as e:
        logger.error(f"ffmpeg command failed: {e}")
        raise

    # Check what frames were actually created
    try:
        created_files = sorted(
            [
                f
                for f in os.listdir(frames_dir)
                if f.startswith("frame_") and f.endswith(".png")
            ]
        )
    except Exception as e:
        logger.error(f"Error listing frames directory: {e}")
        created_files = []

    logger.info(
        f"Created {len(created_files)} frame files in {frames_dir}: {created_files}"
    )

    frames = []
    # Try to load frames with different naming patterns
    for i in range(1, blend_count + 2):  # Try a few extra indices
        for fmt in [f"frame_{i:03d}.png", f"frame_{i:01d}.png", f"frame_{i:02d}.png"]:
            frame_path = os.path.join(frames_dir, fmt)
            if os.path.exists(frame_path):
                try:
                    logger.debug(f"Loading frame: {fmt}")
                    with Image.open(frame_path) as img:
                        frames.append(
                            np.array(img.convert("RGB")).astype(np.float32) / 255.0
                        )
                except Exception as e:
                    logger.warning(f"Failed to load {fmt}: {e}")
                break

    if not frames:
        error_msg = (
            f"No blended frames extracted! "
            f"video={os.path.basename(video_path)}, "
            f"duration={duration:.2f}s, "
            f"start_time={start_time:.2f}s, "
            f"requested_frames={blend_count}, "
            f"frames_dir={frames_dir}, "
            f"files_created={created_files}"
        )
        logger.error(error_msg)
        raise RuntimeError(
            f"No blended frames extracted from {os.path.basename(video_path)}"
        )

    logger.info(f"Successfully loaded {len(frames)} frames for blending")
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


def _concat_videos_with_overlap(
    video_files: List[str],
    output_path: str,
    overlap_frames: int,
    fps: float,
):
    """
    Concatenate video segments with overlap trimming for temporal consistency.

    When latent frames are prepended to segments (except the first), the generated
    videos will have overlap_frames extra frames at the beginning. This function
    trims those overlapping frames before concatenation to avoid temporal duplication.

    Args:
        video_files: List of video file paths to concatenate
        output_path: Path for the final concatenated video
        overlap_frames: Number of frames to trim from the start of each segment (except first)
        fps: Frames per second of the videos

    Process:
        1. First segment: keep as-is
        2. Subsequent segments: skip first (overlap_frames / fps) seconds using ffmpeg -ss
        3. Concatenate all trimmed segments seamlessly
    """
    _validate_positive(fps, "fps")

    if overlap_frames <= 0:
        logger.warning(
            "overlap_frames <= 0 in _concat_videos_with_overlap, "
            "falling back to standard concatenation"
        )
        _concat_videos(video_files, output_path)
        return

    if len(video_files) < 2:
        # Single video: no overlap to trim
        shutil.copy2(video_files[0], output_path)
        logger.info(f"Single segment: copied to {output_path}")
        return

    # Calculate trim duration in seconds
    trim_duration = overlap_frames / fps
    logger.info(
        f"Concatenating {len(video_files)} segments with {overlap_frames} "
        f"overlap frames ({trim_duration:.3f}s trim per segment)"
    )

    # Create temporary directory for trimmed segments
    with tempfile.TemporaryDirectory() as temp_dir:
        trimmed_files = []

        # Process each segment
        for i, video_file in enumerate(video_files):
            if i == 0:
                # First segment: no trimming needed
                trimmed_files.append(video_file)
                logger.debug(f"Segment {i}: keeping original (no trim)")
            else:
                # Subsequent segments: trim first overlap_frames
                trimmed_path = os.path.join(temp_dir, f"trimmed_{i:03d}.mp4")

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(trim_duration),  # Skip first N seconds
                    "-i",
                    video_file,
                    "-c",
                    "copy",  # Stream copy for speed
                    trimmed_path,
                ]

                try:
                    _run_subprocess(cmd, f"Failed to trim segment {i}")
                    trimmed_files.append(trimmed_path)
                    logger.debug(
                        f"Segment {i}: trimmed {trim_duration:.3f}s from start"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to trim segment {i}, using original: {e}",
                        exc_info=True,
                    )
                    trimmed_files.append(video_file)  # Fallback to original

        # Concatenate trimmed segments using standard concat
        _concat_videos(trimmed_files, output_path)
        logger.info(
            f"Successfully concatenated {len(trimmed_files)} segments with overlap handling"
        )


def _concat_videos_with_overlap_and_crossfade(
    video_files: List[str],
    output_path: str,
    overlap_frames: int,
    crossfade_frames: int,
    fps: float,
):
    """
    Concatenate video segments with both overlap trimming AND crossfade transitions.

    This handles the complex case where:
    1. Videos have overlapping latent frames that need trimming
    2. Crossfade transitions are desired between segments

    The processing order is:
    1. Trim overlap_frames from each segment (except first)
    2. Apply crossfade transitions between trimmed segments

    Args:
        video_files: List of video file paths to concatenate
        output_path: Path for the final concatenated video
        overlap_frames: Number of frames to trim from start of each segment (except first)
        crossfade_frames: Number of frames for crossfade transition effect
        fps: Frames per second of the videos
    """
    _validate_positive(fps, "fps")

    if overlap_frames <= 0 or len(video_files) < 2:
        # No overlap handling needed, just apply crossfade
        _crossfade_videos(video_files, output_path, crossfade_frames, fps)
        return

    # Calculate trim duration
    trim_duration = overlap_frames / fps
    logger.info(
        f"Concatenating with overlap trim ({trim_duration:.3f}s) "
        f"and crossfade ({crossfade_frames} frames)"
    )

    # Create temporary directory for trimmed segments
    with tempfile.TemporaryDirectory() as temp_dir:
        trimmed_files = []

        # Trim overlap from all segments except first
        for i, video_file in enumerate(video_files):
            if i == 0:
                trimmed_files.append(video_file)
            else:
                trimmed_path = os.path.join(temp_dir, f"trimmed_{i:03d}.mp4")

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(trim_duration),
                    "-i",
                    video_file,
                    "-c:v",
                    "libx264",  # Re-encode (required for xfade filter)
                    "-pix_fmt",
                    "yuv420p",
                    trimmed_path,
                ]

                try:
                    _run_subprocess(cmd, f"Failed to trim segment {i}")
                    trimmed_files.append(trimmed_path)
                except Exception as e:
                    logger.error(f"Failed to trim segment {i}: {e}", exc_info=True)
                    trimmed_files.append(video_file)

        # Apply crossfade to trimmed segments
        _crossfade_videos(trimmed_files, output_path, crossfade_frames, fps)
        logger.info("Successfully concatenated with overlap trim and crossfade")


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
                "latent": ("LATENT",),
                "cache_window_size": ("INT", {"default": 4}),
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 32}),
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
        """Force re-execution on every queue run to check for new segments."""
        import time

        # Return current timestamp to force re-execution
        # This ensures the node checks for new segments on each run
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
        latent=None,
        cache_window_size=4,
        overlap_frames=0,
    ):
        """
        Prepare the next video segment by determining the starting frame and cached latent.

        IMPORTANT - How to use cached_latent:

        The cached_latent output contains the last N frames from the previous segment
        for temporal consistency. There are TWO ways to use it:

        OPTION 1 - Simple Mode (RECOMMENDED):
            Set overlap_frames=0 to disable latent caching completely.
            This makes the workflow simpler and avoids concatenation issues.

        OPTION 2 - Advanced Mode with Temporal Consistency:
            Set overlap_frames > 0 (e.g., 8) to enable latent caching.

            DO NOT use standard LatentBatch/Concatenate nodes!

            Instead, you need to:
            a) For SVD (Stable Video Diffusion):
               - Ignore the cached_latent (it's informational only)
               - Use the blended image for temporal smoothness

            b) For other models with latent conditioning:
               - Verify the cached_latent shape matches your model's expectations
               - Use model-specific conditioning nodes to prepend frames

            c) OR set overlap_frames=0 and only use image-based conditioning

        This method:
        1. Determines the current segment index
        2. For segment 0: returns initial_image (if provided) with no cached latent
        3. For segments 1+:
           - Extracts a blended frame from the previous segment as the starting image
           - Loads cached latent frames from the previous segment (if overlap_frames > 0)
           - Returns the cached latent output (may be None)
        4. Returns finished=True when all segments are complete

        Returns:
            (next_image, current_segment, finished, final_video_path, cached_latent)
        """
        _validate_positive(segment_seconds, "segment_seconds")
        _validate_positive(total_seconds, "total_seconds")

        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)

        # Count existing segments with detailed logging
        current_segment = _count_segments(project_path)

        logger.info(
            f"VideoSegmentPrepare: project={project_name}, "
            f"path={project_path}, "
            f"current_segment={current_segment}, "
            f"max_segments={max_segments}"
        )

        # Check if generation is complete
        if current_segment >= max_segments:
            final_path = os.path.join(project_path, "final_video.mp4")
            logger.info(f"All segments completed ({current_segment}/{max_segments})")
            return (
                initial_image,
                current_segment,
                True,
                final_path if os.path.exists(final_path) else "",
                latent,
            )

        # First segment: use initial_image, no cached latent
        if current_segment == 0:
            if initial_image is not None:
                logger.info("Segment 0: Using initial_image, no cached latent")
                return (initial_image, 0, False, "", None)
            else:
                logger.warning("Segment 0: No initial_image provided, using fallback")
                return (torch.zeros((1, 512, 512, 3)), 0, False, "", None)

        # Subsequent segments: extract blended frame and load cached latent
        cached_latent = None
        last_segment = None

        if current_segment > 0:
            segments = _list_segments(project_path)
            last_segment = os.path.join(project_path, segments[-1])

            # Load cached latent from previous segment
            if overlap_frames > 0:
                prev_seg_index = current_segment - 1
                cached_latent = _load_latent_cache(project_path, prev_seg_index)

                if cached_latent is not None:
                    logger.info(
                        f"Segment {current_segment}: Loaded cached latent from "
                        f"segment {prev_seg_index}"
                    )
                else:
                    logger.warning(
                        f"Segment {current_segment}: No cached latent found for "
                        f"segment {prev_seg_index}, proceeding without temporal continuity"
                    )

        # Extract blended frame from previous segment
        if last_segment:
            try:
                frame = _extract_blended_frame(
                    last_segment, project_path, frame_blend_count, frame_offset_percent
                )
                logger.info(
                    f"Segment {current_segment}: Extracted blended frame from "
                    f"{os.path.basename(last_segment)}"
                )
            except Exception as e:
                logger.warning(
                    f"Blended frame extraction failed, trying last frame fallback: {e}"
                )
                try:
                    frame = _extract_last_frame(last_segment, project_path)
                    logger.info(
                        f"Segment {current_segment}: Using last frame from "
                        f"{os.path.basename(last_segment)} (fallback)"
                    )
                except Exception as e2:
                    logger.error(
                        f"Both blended and last frame extraction failed: {e2}",
                        exc_info=True,
                    )
                    # Final fallback: use a black frame
                    logger.warning("Using black frame as final fallback")
                    frame = torch.zeros((1, 512, 512, 3))

            # Warn about cached_latent usage if it's not None
            if cached_latent is not None:
                logger.warning(
                    f"⚠️  cached_latent returned with shape: "
                    f"{cached_latent['samples'].shape}. "
                    f"If you get concatenation errors, set overlap_frames=0 in both "
                    f"VideoSegmentPrepare and VideoSegmentSave to disable latent caching. "
                    f"For most workflows, image-based conditioning (blended frame) "
                    f"is sufficient for temporal smoothness."
                )

            return (frame, current_segment, False, "", cached_latent)

        # Fallback (should rarely happen)
        logger.warning(f"Segment {current_segment}: Using fallback zero tensor")
        return (torch.zeros((1, 512, 512, 3)), 0, False, "", cached_latent)


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
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 32}),
                "crossfade_frames": ("INT", {"default": 0, "min": 0, "max": 16}),
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
        overlap_frames=0,
        crossfade_frames=0,
    ):
        """
        Save video segment and cache latent frames for temporal consistency.

        This method:
        1. Saves the generated video segment to disk
        2. Caches the last N frames of the latent (if provided) for the next segment
        3. Performs final concatenation when all segments are complete
        4. Cleans up old latent caches to manage disk space
        """
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        segments = _list_segments(project_path)

        seg_index = len(segments)
        seg_path = os.path.join(project_path, f"segment_{seg_index:03d}.mp4")

        logger.info(
            f"VideoSegmentSave: project={project_name}, "
            f"seg_index={seg_index}, "
            f"max_segments={max_segments}, "
            f"existing_segments={len(segments)}, "
            f"target_path={seg_path}"
        )

        if os.path.exists(seg_path):
            logger.error(f"Segment file already exists: {seg_path}")
            raise RuntimeError("Segment overwrite prevented")

        # Save video segment
        _save_video_tensor(video, seg_path, fps)

        # Verify the file was actually saved
        if os.path.exists(seg_path):
            file_size = os.path.getsize(seg_path)
            logger.info(
                f"Successfully saved segment {seg_index} to {seg_path} "
                f"({file_size / 1024 / 1024:.2f} MB)"
            )
        else:
            logger.error(
                f"Failed to save segment {seg_index}: file not found after save"
            )
            raise RuntimeError(f"Video segment file not created: {seg_path}")

        # Cache latent frames for next segment (if overlap_frames > 0)
        if overlap_frames > 0 and latent is not None:
            cache_saved = _save_latent_cache(
                latent, project_path, seg_index, overlap_frames
            )

            if cache_saved:
                # Cleanup old caches (keep only last 2)
                _cleanup_old_caches(project_path, keep_last_n=2)

        seg_index += 1

        # Final concatenation when all segments are complete
        if seg_index >= max_segments:
            final_path = os.path.join(project_path, "final_video.mp4")
            video_files = [
                os.path.join(project_path, f) for f in _list_segments(project_path)
            ]

            # Choose concatenation method based on parameters
            if overlap_frames > 0 and crossfade_frames > 0:
                # Both overlap trimming and crossfade
                logger.info(
                    f"Final concatenation: overlap trim ({overlap_frames} frames) "
                    f"+ crossfade ({crossfade_frames} frames)"
                )
                _concat_videos_with_overlap_and_crossfade(
                    video_files, final_path, overlap_frames, crossfade_frames, fps
                )
            elif overlap_frames > 0:
                # Overlap trimming only
                logger.info(
                    f"Final concatenation: overlap trim ({overlap_frames} frames)"
                )
                _concat_videos_with_overlap(
                    video_files, final_path, overlap_frames, fps
                )
            elif crossfade_frames > 0:
                # Crossfade only
                logger.info(
                    f"Final concatenation: crossfade ({crossfade_frames} frames)"
                )
                _crossfade_videos(video_files, final_path, crossfade_frames, fps)
            else:
                # Standard concatenation (backward compatible)
                logger.info("Final concatenation: standard (no overlap or crossfade)")
                _concat_videos(video_files, final_path)

            logger.info(f"Final video created at {final_path}")
            return (seg_index, True, final_path)

        return (seg_index, False, "")


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
            info = "Latent is None"
            logger.info(info)
            return (info, latent)

        if not isinstance(latent, dict):
            info = f"Unexpected latent type: {type(latent)}"
            logger.warning(info)
            return (info, latent)

        samples = latent.get("samples")
        if samples is None:
            info = "No 'samples' key in latent dictionary"
            logger.warning(info)
            return (info, latent)

        if not isinstance(samples, torch.Tensor):
            info = f"'samples' is not a tensor (type={type(samples)})"
            logger.warning(info)
            return (info, latent)

        shape = tuple(samples.shape)
        dims = samples.dim()
        total_elements = samples.numel()
        mem_mb = total_elements * samples.element_size() / (1024**2)

        device = str(samples.device)
        dtype = str(samples.dtype)

        info = (
            f"Shape: {shape}\n"
            f"Dimensions: {dims}\n"
            f"DType: {dtype}\n"
            f"Device: {device}\n"
            f"Total elements: {total_elements:,}\n"
            f"Memory: {mem_mb:.2f} MB"
        )

        logger.info(info.replace("\n", " | "))

        return (info, latent)


class ConditioningShapeDebug:
    """
    Debug conditioning structure and tensor properties.

    Similar to LatentShapeDebug, this node inspects CONDITIONING inputs
    and returns formatted information about their structure, dimensions,
    and metadata.

    Input:
        - conditioning: CONDITIONING data (typically a list of tuples)

    Output:
        - shape_info: String containing detailed conditioning structure info
        - conditioning: The input conditioning (passthrough for chaining)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("shape_info", "conditioning")

    FUNCTION = "debug_shape"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/VideoSegment"

    def debug_shape(self, conditioning):
        if conditioning is None:
            info = "Conditioning is None"
            logger.info(info)
            return (info, conditioning)

        if not isinstance(conditioning, list):
            info = f"Unexpected conditioning type: {type(conditioning)} (expected list)"
            logger.warning(info)
            return (info, conditioning)

        if len(conditioning) == 0:
            info = "Conditioning list is empty"
            logger.warning(info)
            return (info, conditioning)

        # Analyze first conditioning element (typical structure)
        first_cond = conditioning[0]

        if not isinstance(first_cond, (tuple, list)) or len(first_cond) < 2:
            info = f"Unexpected conditioning element structure: {type(first_cond)}"
            logger.warning(info)
            return (info, conditioning)

        cond_tensor = first_cond[0]
        cond_metadata = first_cond[1] if len(first_cond) > 1 else {}

        # Build info string
        info_parts = [f"Conditioning list length: {len(conditioning)}"]

        if isinstance(cond_tensor, torch.Tensor):
            shape = tuple(cond_tensor.shape)
            dims = cond_tensor.dim()
            total_elements = cond_tensor.numel()
            mem_mb = total_elements * cond_tensor.element_size() / (1024**2)
            device = str(cond_tensor.device)
            dtype = str(cond_tensor.dtype)

            info_parts.append(f"First element tensor shape: {shape}")
            info_parts.append(f"Dimensions: {dims}")
            info_parts.append(f"DType: {dtype}")
            info_parts.append(f"Device: {device}")
            info_parts.append(f"Total elements: {total_elements:,}")
            info_parts.append(f"Memory: {mem_mb:.2f} MB")
        else:
            info_parts.append(
                f"First element tensor type: {type(cond_tensor)} (not a torch.Tensor)"
            )

        if isinstance(cond_metadata, dict):
            info_parts.append(f"Metadata keys: {list(cond_metadata.keys())}")
            for key, value in cond_metadata.items():
                if isinstance(value, torch.Tensor):
                    info_parts.append(f"  {key}: Tensor {tuple(value.shape)}")
                else:
                    info_parts.append(f"  {key}: {type(value).__name__}")
        else:
            info_parts.append(f"Metadata type: {type(cond_metadata)}")

        info = "\n".join(info_parts)
        logger.info(info.replace("\n", " | "))

        return (info, conditioning)


class LatentPrependCache:
    """
    Prepend cached latent frames to a new latent for temporal consistency.

    This node correctly concatenates latents in the FRAME dimension (dim=2)
    for 5D video latent tensors [B, C, F, H, W].

    Use this to combine:
    - cached_latent (e.g., 8 frames from previous segment)
    - new_latent (e.g., 20 frames from current generation)

    Result: Combined latent with 28 frames total.

    IMPORTANT: Both latents must have matching B, C, H, W dimensions!
    Only the frame count (F) can differ.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_latent": ("LATENT",),
            },
            "optional": {
                "cached_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("combined_latent",)

    FUNCTION = "prepend_cache"
    CATEGORY = "MYNodes/VideoSegment"

    def prepend_cache(self, new_latent, cached_latent=None):
        """
        Prepend cached latent frames to new latent.

        If cached_latent is None or empty, returns new_latent unchanged.
        """
        # If no cached latent, return new latent as-is
        if cached_latent is None:
            logger.info("No cached_latent provided, returning new_latent unchanged")
            return (new_latent,)

        # Validate new_latent
        if not isinstance(new_latent, dict) or "samples" not in new_latent:
            raise ValueError("new_latent must be a dict with 'samples' key")

        new_samples = new_latent["samples"]
        if not isinstance(new_samples, torch.Tensor):
            raise ValueError("new_latent['samples'] must be a torch.Tensor")

        # Validate cached_latent
        if not isinstance(cached_latent, dict) or "samples" not in cached_latent:
            logger.warning(
                "Invalid cached_latent format, returning new_latent unchanged"
            )
            return (new_latent,)

        cached_samples = cached_latent["samples"]
        if not isinstance(cached_samples, torch.Tensor):
            logger.warning(
                "cached_latent['samples'] is not a tensor, returning new_latent unchanged"
            )
            return (new_latent,)

        # Check dimensions
        if new_samples.dim() != 5:
            raise ValueError(
                f"new_latent must be 5D [B,C,F,H,W], got shape {list(new_samples.shape)}"
            )

        if cached_samples.dim() != 5:
            raise ValueError(
                f"cached_latent must be 5D [B,C,F,H,W], got shape {list(cached_samples.shape)}"
            )

        # Check compatible dimensions
        new_shape = new_samples.shape
        cached_shape = cached_samples.shape

        if (
            new_shape[0] != cached_shape[0]  # Batch
            or new_shape[1] != cached_shape[1]  # Channels
            or new_shape[3] != cached_shape[3]  # Height
            or new_shape[4] != cached_shape[4]
        ):  # Width
            raise ValueError(
                f"Incompatible latent dimensions!\n"
                f"new_latent:    {list(new_shape)} [B,C,F,H,W]\n"
                f"cached_latent: {list(cached_shape)} [B,C,F,H,W]\n"
                f"B, C, H, W must match! Only F (frames) can differ."
            )

        # Concatenate along frame dimension (dim=2)
        try:
            combined_samples = torch.cat([cached_samples, new_samples], dim=2)

            logger.info(
                f"Prepended cached latent: "
                f"cached={list(cached_shape)} + new={list(new_shape)} "
                f"→ combined={list(combined_samples.shape)}"
            )

            # Return combined latent in standard format
            return ({"samples": combined_samples},)

        except Exception as e:
            logger.error(
                f"Failed to concatenate latents: {e}\n"
                f"new_shape={list(new_shape)}, cached_shape={list(cached_shape)}",
                exc_info=True,
            )
            raise


class LatentExtendFrames:
    """
    Extends a video latent to a target number of frames by repeating the last frame.
    Useful for matching frame counts between latent_image and conditioning images.

    Input:
        - latent: Video latent tensor [B, C, F, H, W]
        - target_frames: Desired total number of frames

    Output:
        - extended_latent: Latent with target_frames frames [B, C, target_frames, H, W]

    Example:
        Input: (1, 16, 20, 51, 85) with target_frames=28
        Output: (1, 16, 28, 51, 85) - last frame repeated 8 times
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "target_frames": ("INT", {"default": 28, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("extended_latent",)
    FUNCTION = "extend_frames"
    CATEGORY = "MYNodes/VideoSegment"

    def extend_frames(self, latent, target_frames):
        """Extend latent to target_frames by repeating the last frame."""
        try:
            samples = latent["samples"]

            # Validate 5D tensor [B, C, F, H, W]
            if samples.dim() != 5:
                raise ValueError(
                    f"Expected 5D latent tensor [B, C, F, H, W], got {samples.dim()}D: {list(samples.shape)}"
                )

            _, _, F, _, _ = samples.shape

            logger.info(
                f"LatentExtendFrames: input={list(samples.shape)}, target_frames={target_frames}"
            )

            # If already at target, return as-is
            if F == target_frames:
                logger.info("Already at target frames, returning unchanged")
                return (latent,)

            # If input has more frames than target, truncate
            if F > target_frames:
                logger.warning(
                    f"Input has more frames ({F}) than target ({target_frames}), truncating"
                )
                extended_samples = samples[:, :, :target_frames, :, :]
                logger.info(f"Truncated to: {list(extended_samples.shape)}")
                return ({"samples": extended_samples},)

            # Extend by repeating last frame
            frames_to_add = target_frames - F
            last_frame = samples[:, :, -1:, :, :]  # [B, C, 1, H, W]

            # Repeat last frame
            repeated_frames = last_frame.repeat(
                1, 1, frames_to_add, 1, 1
            )  # [B, C, frames_to_add, H, W]

            # Concatenate original + repeated
            extended_samples = torch.cat([samples, repeated_frames], dim=2)

            logger.info(
                f"Extended latent: {list(samples.shape)} + {frames_to_add} repeated frames → {list(extended_samples.shape)}"
            )

            return ({"samples": extended_samples},)

        except Exception as e:
            logger.error(f"Failed to extend frames: {e}", exc_info=True)
            raise


class ConditioningExtendFrames:
    """
    Extends video conditioning to a target number of frames by repeating the last frame.
    Useful for matching frame counts between latent_image and conditioning.

    Input:
        - conditioning: CONDITIONING with embedded frame dimension
        - target_frames: Desired total number of frames

    Output:
        - extended_conditioning: CONDITIONING with target_frames frames

    Example:
        If conditioning has 20 frames internally and target_frames=28,
        the last frame is repeated 8 times.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "target_frames": ("INT", {"default": 28, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("extended_conditioning",)
    FUNCTION = "extend_frames"
    CATEGORY = "MYNodes/VideoSegment"

    def extend_frames(self, conditioning, target_frames):
        """Extend conditioning to target_frames by repeating the last frame."""
        try:
            # CONDITIONING is a list of tuples: [(cond_tensor, metadata_dict), ...]
            extended_cond = []

            for cond_item in conditioning:
                # Each item is (tensor, dict)
                cond_tensor = cond_item[0]
                cond_dict = cond_item[1].copy() if len(cond_item) > 1 else {}

                logger.info(
                    f"ConditioningExtendFrames: input tensor shape={list(cond_tensor.shape)}, "
                    f"target_frames={target_frames}"
                )

                # Try to find frame dimension in the tensor
                # Common patterns:
                # - [B, F, ...] - batch, frames, others
                # - [F, ...] - frames first
                # We'll look for a dimension that could reasonably be frames (< 200)

                original_shape = cond_tensor.shape

                # Check if tensor has a temporal dimension we can extend
                # For video models, conditioning often has shape like [B, F, D] or [B, F, H, W]
                if cond_tensor.dim() >= 2:
                    # Assume frame dimension is at index 1 (after batch)
                    # This matches CogVideoX/Wan conditioning format: [B, F, ...]
                    current_frames = cond_tensor.shape[1]

                    logger.info(f"Detected {current_frames} frames in dimension 1")

                    if current_frames == target_frames:
                        logger.info("Already at target frames, returning unchanged")
                        extended_cond.append(cond_item)
                        continue

                    if current_frames > target_frames:
                        logger.warning(
                            f"Conditioning has more frames ({current_frames}) than target ({target_frames}), truncating"
                        )
                        extended_tensor = cond_tensor[:, :target_frames, ...]
                    else:
                        # Extend by repeating last frame
                        frames_to_add = target_frames - current_frames
                        last_frame = cond_tensor[:, -1:, ...]  # Keep dimension

                        # Repeat last frame
                        repeated_frames = last_frame.repeat(
                            1, frames_to_add, *([1] * (cond_tensor.dim() - 2))
                        )

                        # Concatenate
                        extended_tensor = torch.cat(
                            [cond_tensor, repeated_frames], dim=1
                        )

                    logger.info(
                        f"Extended conditioning: {list(original_shape)} → {list(extended_tensor.shape)}"
                    )

                    # Create new conditioning item
                    extended_cond.append((extended_tensor, cond_dict))
                else:
                    # Tensor doesn't have frame dimension, keep as-is
                    logger.warning(
                        f"Conditioning tensor has only {cond_tensor.dim()} dimensions, "
                        "cannot extend frames. Returning unchanged."
                    )
                    extended_cond.append(cond_item)

            return (extended_cond,)

        except Exception as e:
            logger.error(f"Failed to extend conditioning frames: {e}", exc_info=True)
            raise


class VideoLatentMask:
    """
    Generate a frame-wise mask for 5D video latents [B, C, F, H, W].
    The first 'black_frames' are set to 0.0, and the rest are set to 1.0.
    This is useful for preserving the first few frames of a video segment
    while allowing subsequent frames to be modified by the model.
    """

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
        """Generate a 5D latent mask [B, C, F, H, W]."""

        # Create a tensor of ones [B, C, F, H, W]
        # Using ones() as the default 'unmasked' state
        mask = torch.ones((batch_size, channels, frames, height, width), dtype=torch.float32)

        # Set the first black_frames to 0.0 (masked)
        if black_frames > 0:
            actual_black = min(black_frames, frames)
            mask[:, :, :actual_black, :, :] = 0.0

            logger.info(
                f"Generated VideoLatentMask: shape={list(mask.shape)}, "
                f"black_frames={actual_black}/{frames}"
            )
        else:
            logger.info(f"Generated VideoLatentMask (all white): shape={list(mask.shape)}")

        return ({"samples": mask},)
