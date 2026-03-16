#!/usr/bin/env python

"""Tests for `my_custom_nodes` package."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.my_custom_nodes.nodes import AspectRatioCrop, PixelatedBorderNode, VideoLatentMask
from src.my_custom_nodes.video_segment_extender import (
    VideoSegmentPrepare,
    VideoSegmentSave,
    WanImagesToVideo,
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
        "IMAGE",
        "IMAGE",
    )
    assert VideoSegmentPrepare.RETURN_NAMES == (
        "next_image",
        "current_segment",
        "finished",
        "final_video_path",
        "cached_latent",
        "last_segment_video",
        "context_images",
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
    assert VideoLatentMask.RETURN_TYPES == ("MASK",)
    assert VideoLatentMask.RETURN_NAMES == ("mask",)
    assert VideoLatentMask.FUNCTION == "generate_mask"
    assert VideoLatentMask.CATEGORY == "MYNodes/VideoSegment"


def test_video_latent_mask_shape(video_latent_mask_node):
    """Testa se o formato do tensor gerado está correto [B, C, F, H, W]."""
    B, C, F, H, W = 1, 16, 16, 64, 64
    result = video_latent_mask_node.generate_mask(B, C, F, H, W, 4)

    assert isinstance(result, tuple)
    mask = result[0]
    assert mask.shape == (B, F, H, W)  # ComfyUI MASK is [B, F, H, W] or [F, H, W]
    assert mask.dtype == torch.float32


# ============================================================
# Tests for WanImagesToVideo Node
# ============================================================


@pytest.fixture
def wan_images_node():
    return WanImagesToVideo()


def test_wan_images_to_video_metadata(wan_images_node):
    assert WanImagesToVideo.RETURN_TYPES == ("LATENT", "LATENT", "MASK")
    assert WanImagesToVideo.RETURN_NAMES == ("latent", "concat_latent", "concat_mask")
    assert WanImagesToVideo.CATEGORY == "MYNodes/Wan"


@patch("comfy.utils.common_upscale")
@patch("comfy.node_helpers.conditioning_set_values")
def test_wan_images_to_video_encode(mock_set_values, mock_upscale, wan_images_node):
    # Setup mocks
    vae = MagicMock()
    vae.device = torch.device("cpu")
    
    # Mock VAE output [1, 16, T, H, W]
    # For length=16, Wan latent T should be 5 ((16-1)//4 + 1)
    vae.encode.return_value = torch.randn(1, 16, 5, 32, 32)
    
    # Mock upscale to return the same images but moved dim
    def side_effect(img, w, h, method, center):
        return img
    mock_upscale.side_effect = side_effect
    
    # Mock conditioning patching to return the input (for simplicity)
    mock_set_values.side_effect = lambda cond, values: cond

    # Inputs
    positive = MagicMock()
    negative = MagicMock()
    images = torch.randn(4, 64, 64, 3) # 4 images
    
    result = wan_images_node.encode(
        positive, negative, vae, images, 
        length=16, width=128, height=128, batch_size=1
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    pos_out, neg_out, latent = result
    
    # Verify latent shape [B, 16, T, H/8, W/8]
    # T = (16-1)//4 + 1 = 4 (actually wait, 15//4 is 3, +1 is 4. My mock above used 5, let's fix check)
    # The code uses: (length - 1) // 4 + 1
    # 16-1 = 15. 15 // 4 = 3. 3 + 1 = 4.
    assert latent["samples"].shape == (1, 16, 4, 16, 16) # 128/8 = 16
    
    # Verify node_helpers.conditioning_set_values was called
    assert mock_set_values.call_count >= 2 # Once for pos, once for neg
    
    # Verify the mask passed to set_values
    # Call args: (conditioning, values_dict)
    # values_dict should contain 'concat_mask'
    call_args = mock_set_values.call_args_list[0]
    values = call_args[0][1]
    
    assert "concat_mask" in values
    mask = values["concat_mask"]
    
    # 4 images -> 1 fixed latent frame?
    # (4-1)//4 + 1 = 1. So first frame should be 0.0
    assert torch.all(mask[:, :, :1, :, :] == 0.0)
    assert torch.all(mask[:, :, 1:, :, :] == 1.0)
