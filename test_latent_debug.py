#!/usr/bin/env python3
"""
Script de teste para validar o LatentShapeDebug node.
Simula o comportamento do node sem depender do ComfyUI.
"""

import torch


class LatentShapeDebugTest:
    """Versão standalone do node para teste."""

    def debug_shape(self, latent):
        """Extrai e formata informações sobre as dimensões do latent."""
        if latent is None:
            return ("Latent is None", latent)

        samples = latent.get("samples")
        if samples is None:
            return ("No 'samples' key in latent dict", latent)

        shape = samples.shape
        shape_str = f"[{', '.join(str(d) for d in shape)}]"

        # Tentar identificar o formato
        if len(shape) == 5:
            b, c, f, h, w = shape
            info = (
                f"Shape: {shape_str}\n"
                f"Format: [B, C, F, H, W]\n"
                f"Batch: {b}, Channels: {c}, Frames: {f}, Height: {h}, Width: {w}\n"
                f"Total elements: {samples.numel():,}\n"
                f"Memory (approx): {samples.numel() * samples.element_size() / (1024**2):.2f} MB"
            )
        elif len(shape) == 4:
            b, c, h, w = shape
            info = (
                f"Shape: {shape_str}\n"
                f"Format: [B, C, H, W] (Image latent)\n"
                f"Batch: {b}, Channels: {c}, Height: {h}, Width: {w}\n"
                f"Total elements: {samples.numel():,}\n"
                f"Memory (approx): {samples.numel() * samples.element_size() / (1024**2):.2f} MB"
            )
        else:
            info = (
                f"Shape: {shape_str}\n"
                f"Dimensions: {len(shape)}\n"
                f"Total elements: {samples.numel():,}\n"
                f"Memory (approx): {samples.numel() * samples.element_size() / (1024**2):.2f} MB"
            )

        print(f"LatentShapeDebug: {info.replace(chr(10), ' | ')}")
        return (info, latent)


def test_video_latent():
    """Testa com latent de vídeo [B, C, F, H, W]."""
    print("\n=== Teste 1: Video Latent 5D ===")
    video_latent = {
        "samples": torch.randn(1, 4, 32, 64, 64)  # B=1, C=4, F=32, H=64, W=64
    }
    node = LatentShapeDebugTest()
    shape_info, _ = node.debug_shape(video_latent)
    print(shape_info)


def test_image_latent():
    """Testa com latent de imagem [B, C, H, W]."""
    print("\n=== Teste 2: Image Latent 4D ===")
    image_latent = {"samples": torch.randn(1, 4, 64, 64)}  # B=1, C=4, H=64, W=64
    node = LatentShapeDebugTest()
    shape_info, _ = node.debug_shape(image_latent)
    print(shape_info)


def test_overlap_extraction():
    """Testa extração de overlap de frames."""
    print("\n=== Teste 3: Overlap Extraction ===")
    full_latent = torch.randn(1, 4, 32, 64, 64)  # 32 frames
    print(f"Original shape: {list(full_latent.shape)}")

    overlap_frames = 8
    overlap_latent = full_latent[:, :, -overlap_frames:, :, :]
    print(f"Overlap shape ({overlap_frames} frames): {list(overlap_latent.shape)}")

    node = LatentShapeDebugTest()
    shape_info, _ = node.debug_shape({"samples": overlap_latent})
    print(shape_info)


if __name__ == "__main__":
    print("🔍 Testando LatentShapeDebug Node\n")
    test_video_latent()
    test_image_latent()
    test_overlap_extraction()
    print("\n✅ Todos os testes concluídos!")
