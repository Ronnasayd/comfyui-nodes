from .pixelated_border_node import PixelatedBorderNode
from .video_segment_extender import (
    VideoSegmentExtender,
    VideoSegmentPrepare,
    VideoSegmentSave,
)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PixelatedBorderNode": PixelatedBorderNode,
    "VideoSegmentPrepare": VideoSegmentPrepare,
    "VideoSegmentSave": VideoSegmentSave,
    # Mantido para compatibilidade com workflows antigos
    "VideoSegmentExtender": VideoSegmentExtender,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelatedBorderNode": "Pixelated Border Node",
    "VideoSegmentPrepare": "Video Segment Prepare",
    "VideoSegmentSave": "Video Segment Save",
    "VideoSegmentExtender": "Video Segment Extender (deprecated)",
}
