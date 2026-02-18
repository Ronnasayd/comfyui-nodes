#!/usr/bin/env python

"""Tests for `my_custom_nodes` package."""

import pytest
from src.my_custom_nodes.nodes import MyNode

@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return MyNode()

def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, MyNode)

def test_return_types():
    """Test the node's metadata."""
    assert MyNode.RETURN_TYPES == ("IMAGE",)
    assert MyNode.FUNCTION == "test"
    assert MyNode.CATEGORY == "Example"
