#!/usr/bin/env python

"""Tests for `my_custom_nodes` package."""

import pytest

from src.my_custom_nodes.nodes import AspectRatioCrop, PixelatedBorderNode


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
