#!/usr/bin/env python

"""Tests for `my_custom_nodes` package."""

import pytest

from src.my_custom_nodes.nodes import PixelatedBorderNode


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
