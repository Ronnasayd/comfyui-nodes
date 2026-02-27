from pixelated_border_node import PixelatedBorderNode
from video_segment_extender import WanSegmentStitcher

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PixelatedBorderNode": PixelatedBorderNode,
    "WanSegmentStitcher": WanSegmentStitcher,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelatedBorderNode": "Pixelated Border Node",
    "WanSegmentStitcher": "Video Segment Extender",
}
