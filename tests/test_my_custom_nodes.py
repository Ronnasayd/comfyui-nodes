#!/usr/bin/env python

"""Tests for `my_custom_nodes` package."""

import os
import tempfile

import pytest
import torch

from src.my_custom_nodes.nodes import AspectRatioCrop, PixelatedBorderNode, VideoLatentMask
from src.my_custom_nodes.video_segment_extender import (
    VideoSegmentPrepare,
    VideoSegmentSave,
    _cleanup_old_caches,
    _load_latent_cache,
    _save_latent_cache,
)


@pytest.fixture
def example_node():
    """Fixture para criar uma instância do PixelatedBorderNode."""
    return PixelatedBorderNode()


def test_pixelated_border_node_initialization(example_node):
    """Testa se o nó pode ser instanciado."""
    assert isinstance(example_node, PixelatedBorderNode)


def test_return_types():
    """Testa os metadados do nó."""
    assert PixelatedBorderNode.RETURN_TYPES == ("IMAGE", "MASK")
    assert PixelatedBorderNode.FUNCTION == "process_image"
    assert PixelatedBorderNode.CATEGORY == "MYNodes"


# Testes para AspectRatioCrop


@pytest.fixture
def aspect_ratio_crop_node():
    """Fixture para criar uma instância do AspectRatioCrop."""
    return AspectRatioCrop()


def test_aspect_ratio_crop_initialization(aspect_ratio_crop_node):
    """Testa se o nó AspectRatioCrop pode ser instanciado."""
    assert isinstance(aspect_ratio_crop_node, AspectRatioCrop)


def test_aspect_ratio_crop_metadata():
    """Testa os metadados do nó AspectRatioCrop."""
    assert AspectRatioCrop.RETURN_TYPES == ("IMAGE",)
    assert AspectRatioCrop.RETURN_NAMES == ("cropped_image",)
    assert AspectRatioCrop.FUNCTION == "crop_to_aspect_ratio"
    assert AspectRatioCrop.CATEGORY == "MYNodes"


def test_aspect_ratio_crop_input_types():
    """Testa se os tipos de entrada são configurados corretamente."""
    input_types = AspectRatioCrop.INPUT_TYPES()
    assert "required" in input_types
    assert "base_image" in input_types["required"]
    assert "padded_image" in input_types["required"]
    assert input_types["required"]["base_image"] == ("IMAGE",)
    assert input_types["required"]["padded_image"] == ("IMAGE",)


# ============================================================
# Tests for Latent Caching Infrastructure
# ============================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def video_latent_5d():
    """Create a 5D video latent tensor [B, C, F, H, W]."""
    return {"samples": torch.randn(1, 4, 32, 64, 64)}


@pytest.fixture
def image_latent_4d():
    """Create a 4D image latent tensor [B, C, H, W]."""
    return {"samples": torch.randn(1, 4, 64, 64)}


def test_latent_cache_save_load_roundtrip_5d(temp_project_dir, video_latent_5d):
    """Test that 5D video latents are preserved through save/load cycle."""
    overlap_frames = 8
    segment_index = 0

    # Save cache
    result = _save_latent_cache(
        video_latent_5d, temp_project_dir, segment_index, overlap_frames
    )
    assert result is True

    # Verify cache file exists
    cache_path = os.path.join(temp_project_dir, "latent_cache_seg_000.pt")
    assert os.path.exists(cache_path)

    # Load cache
    loaded_latent = _load_latent_cache(temp_project_dir, segment_index)
    assert loaded_latent is not None
    assert "samples" in loaded_latent

    # Verify shape matches expected overlap extraction
    expected_shape = (1, 4, 8, 64, 64)  # Last 8 frames
    assert loaded_latent["samples"].shape == expected_shape

    # Verify values match (last 8 frames of original)
    original_overlap = video_latent_5d["samples"][:, :, -overlap_frames:, :, :]
    assert torch.allclose(loaded_latent["samples"], original_overlap, rtol=1e-5)


def test_latent_cache_save_with_4d_image_latent(temp_project_dir, image_latent_4d):
    """Test that 4D image latents are rejected (no temporal dimension)."""
    result = _save_latent_cache(image_latent_4d, temp_project_dir, 0, 8)
    assert result is False  # Should fail gracefully

    # Verify no cache file created
    cache_path = os.path.join(temp_project_dir, "latent_cache_seg_000.pt")
    assert not os.path.exists(cache_path)


def test_latent_cache_save_with_none_latent(temp_project_dir):
    """Test that None latent is handled gracefully."""
    result = _save_latent_cache(None, temp_project_dir, 0, 8)
    assert result is False

    cache_path = os.path.join(temp_project_dir, "latent_cache_seg_000.pt")
    assert not os.path.exists(cache_path)


def test_latent_cache_save_with_zero_overlap_frames(temp_project_dir, video_latent_5d):
    """Test that overlap_frames=0 skips caching."""
    result = _save_latent_cache(video_latent_5d, temp_project_dir, 0, 0)
    assert result is False

    cache_path = os.path.join(temp_project_dir, "latent_cache_seg_000.pt")
    assert not os.path.exists(cache_path)


def test_latent_cache_insufficient_frames(temp_project_dir):
    """Test handling when video has fewer frames than overlap_frames."""
    short_latent = {"samples": torch.randn(1, 4, 5, 64, 64)}  # Only 5 frames
    overlap_frames = 8

    result = _save_latent_cache(short_latent, temp_project_dir, 0, overlap_frames)
    assert result is True  # Should succeed but cache all 5 frames

    loaded_latent = _load_latent_cache(temp_project_dir, 0)
    assert loaded_latent is not None
    assert loaded_latent["samples"].shape[2] == 5  # All 5 frames cached


def test_latent_cache_load_nonexistent(temp_project_dir):
    """Test that loading non-existent cache returns None."""
    loaded_latent = _load_latent_cache(temp_project_dir, 999)
    assert loaded_latent is None


def test_latent_cache_load_corrupted(temp_project_dir):
    """Test that corrupted cache file is handled gracefully."""
    cache_path = os.path.join(temp_project_dir, "latent_cache_seg_000.pt")

    # Create corrupted cache file
    with open(cache_path, "w") as f:
        f.write("corrupted data")

    loaded_latent = _load_latent_cache(temp_project_dir, 0)
    assert loaded_latent is None  # Should return None on load failure


def test_cleanup_old_caches(temp_project_dir, video_latent_5d):
    """Test that old cache files are cleaned up correctly."""
    # Create 5 cache files
    for i in range(5):
        _save_latent_cache(video_latent_5d, temp_project_dir, i, 8)

    # Verify all 5 exist
    for i in range(5):
        cache_path = os.path.join(temp_project_dir, f"latent_cache_seg_{i:03d}.pt")
        assert os.path.exists(cache_path)

    # Cleanup, keeping last 2
    _cleanup_old_caches(temp_project_dir, keep_last_n=2)

    # First 3 should be removed
    for i in range(3):
        cache_path = os.path.join(temp_project_dir, f"latent_cache_seg_{i:03d}.pt")
        assert not os.path.exists(cache_path)

    # Last 2 should remain
    for i in range(3, 5):
        cache_path = os.path.join(temp_project_dir, f"latent_cache_seg_{i:03d}.pt")
        assert os.path.exists(cache_path)


def test_cleanup_with_fewer_caches_than_keep_limit(temp_project_dir, video_latent_5d):
    """Test cleanup when cache count is less than keep_last_n."""
    # Create only 2 cache files
    for i in range(2):
        _save_latent_cache(video_latent_5d, temp_project_dir, i, 8)

    # Cleanup with keep_last_n=5 (more than we have)
    _cleanup_old_caches(temp_project_dir, keep_last_n=5)

    # Both should still exist
    for i in range(2):
        cache_path = os.path.join(temp_project_dir, f"latent_cache_seg_{i:03d}.pt")
        assert os.path.exists(cache_path)


# ============================================================
# Integration Tests for VideoSegmentPrepare and VideoSegmentSave
# ============================================================


def test_video_segment_prepare_initialization():
    """Test VideoSegmentPrepare node can be instantiated."""
    node = VideoSegmentPrepare()
    assert isinstance(node, VideoSegmentPrepare)
    assert node.CATEGORY == "MYNodes/VideoSegment"
    assert node.FUNCTION == "prepare"


def test_video_segment_save_initialization():
    """Test VideoSegmentSave node can be instantiated."""
    node = VideoSegmentSave()
    assert isinstance(node, VideoSegmentSave)
    assert node.CATEGORY == "MYNodes/VideoSegment"
    assert node.FUNCTION == "save"


def test_video_segment_nodes_metadata():
    """Test that VideoSegment nodes have correct metadata."""
    # VideoSegmentPrepare
    assert VideoSegmentPrepare.RETURN_TYPES == (
        "IMAGE",
        "INT",
        "BOOLEAN",
        "STRING",
        "LATENT",
    )
    assert VideoSegmentPrepare.RETURN_NAMES == (
        "next_image",
        "current_segment",
        "finished",
        "final_video_path",
        "cached_latent",
    )

    # VideoSegmentSave
    assert VideoSegmentSave.RETURN_TYPES == ("INT", "BOOLEAN", "STRING")
    assert VideoSegmentSave.RETURN_NAMES == (
        "saved_segment",
        "finished",
        "final_video_path",
    )
    assert VideoSegmentSave.OUTPUT_NODE is True


def test_overlap_frame_extraction():
    """Test extraction of overlap frames from 5D latent tensor."""
    full_latent = torch.randn(1, 4, 32, 64, 64)  # 32 frames
    overlap_frames = 8

    # Extract last N frames (same logic as _save_latent_cache)
    overlap_latent = full_latent[:, :, -overlap_frames:, :, :]

    assert overlap_latent.shape == (1, 4, 8, 64, 64)

    # Verify extracted frames match the last 8 frames of original
    assert torch.allclose(overlap_latent, full_latent[:, :, -8:, :, :])


# ============================================================
# Tests for VideoLatentMask Node
# ============================================================


@pytest.fixture
def video_latent_mask_node():
    """Fixture para criar uma instância do VideoLatentMask."""
    return VideoLatentMask()


def test_video_latent_mask_initialization(video_latent_mask_node):
    """Testa se o nó VideoLatentMask pode ser instanciado."""
    assert isinstance(video_latent_mask_node, VideoLatentMask)


def test_video_latent_mask_metadata():
    """Testa os metadados do nó VideoLatentMask."""
    assert VideoLatentMask.RETURN_TYPES == ("LATENT",)
    assert VideoLatentMask.RETURN_NAMES == ("mask",)
    assert VideoLatentMask.FUNCTION == "generate_mask"
    assert VideoLatentMask.CATEGORY == "MYNodes/VideoSegment"


def test_video_latent_mask_shape(video_latent_mask_node):
    """Testa se o formato do tensor gerado está correto [B, C, F, H, W]."""
    B, C, F, H, W = 1, 16, 16, 64, 64
    result = video_latent_mask_node.generate_mask(B, C, F, H, W, 4)

    assert isinstance(result, tuple)
    assert isinstance(result[0], dict)
    assert "samples" in result[0]

    mask = result[0]["samples"]
    assert mask.shape == (B, C, F, H, W)
    assert mask.dtype == torch.float32


def test_video_latent_mask_content(video_latent_mask_node):
    """Testa se o conteúdo da máscara está correto (preto vs branco)."""
    B, C, F, H, W = 1, 4, 10, 8, 8
    black_frames = 4
    result = video_latent_mask_node.generate_mask(B, C, F, H, W, black_frames)
    mask = result[0]["samples"]

    # Verifica frames pretos (0.0)
    for f in range(black_frames):
        assert torch.all(mask[:, :, f, :, :] == 0.0)

    # Verifica frames brancos (1.0)
    for f in range(black_frames, F):
        assert torch.all(mask[:, :, f, :, :] == 1.0)


def test_video_latent_mask_edge_cases(video_latent_mask_node):
    """Testa casos extremos: black_frames=0 e black_frames=total_frames."""
    B, C, F, H, W = 1, 4, 10, 8, 8

    # Caso 1: black_frames = 0 (toda branca)
    result_white = video_latent_mask_node.generate_mask(B, C, F, H, W, 0)
    mask_white = result_white[0]["samples"]
    assert torch.all(mask_white == 1.0)

    # Caso 2: black_frames = F (toda preta)
    result_black = video_latent_mask_node.generate_mask(B, C, F, H, W, F)
    mask_black = result_black[0]["samples"]
    assert torch.all(mask_black == 0.0)

    # Caso 3: black_frames > F (deve limitar ao total de frames e ser toda preta)
    result_over = video_latent_mask_node.generate_mask(B, C, F, H, W, F + 5)
    mask_over = result_over[0]["samples"]
    assert torch.all(mask_over == 0.0)


def test_video_latent_mask_registration():
    """Verifica se o nó está registrado corretamente no NODE_CLASS_MAPPINGS."""
    from src.my_custom_nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert "VideoLatentMask" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["VideoLatentMask"] == VideoLatentMask
    assert "VideoLatentMask" in NODE_DISPLAY_NAME_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS["VideoLatentMask"] == "Video Latent Mask"
