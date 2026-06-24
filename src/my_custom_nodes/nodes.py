from .video_segment_extender import VideoConcatenate
from .aspect_ratio_crop import AspectRatioCrop
from .pixelated_border_node import PixelatedBorderNode

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoConcatenate": VideoConcatenate,
    "AspectRatioCrop": AspectRatioCrop,
    "PixelatedBorderNode": PixelatedBorderNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatenate": "Video Concatenate",
    "AspectRatioCrop": "Aspect Ratio Crop",
    "PixelatedBorderNode": "Pixelated Border",
}
