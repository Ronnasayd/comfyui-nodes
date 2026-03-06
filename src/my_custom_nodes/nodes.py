from .aspect_ratio_crop import AspectRatioCrop
from .pixelated_border_node import PixelatedBorderNode
from .video_segment_extender import (
    ConditioningExtendFrames,
    ConditioningShapeDebug,
    LatentExtendFrames,
    LatentPrependCache,
    LatentShapeDebug,
    VideoLatentMask,
    VideoSegmentPrepare,
    VideoSegmentSave,
)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AspectRatioCrop": AspectRatioCrop,
    "PixelatedBorderNode": PixelatedBorderNode,
    "VideoSegmentPrepare": VideoSegmentPrepare,
    "VideoSegmentSave": VideoSegmentSave,
    "LatentShapeDebug": LatentShapeDebug,
    "ConditioningShapeDebug": ConditioningShapeDebug,
    "LatentPrependCache": LatentPrependCache,
    "LatentExtendFrames": LatentExtendFrames,
    "ConditioningExtendFrames": ConditioningExtendFrames,
    "VideoLatentMask": VideoLatentMask,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioCrop": "Aspect Ratio Crop",
    "PixelatedBorderNode": "Pixelated Border Node",
    "VideoSegmentPrepare": "Video Segment Prepare",
    "VideoSegmentSave": "Video Segment Save",
    "LatentShapeDebug": "Latent Shape Debug",
    "ConditioningShapeDebug": "Conditioning Shape Debug",
    "LatentPrependCache": "Latent Prepend Cache",
    "LatentExtendFrames": "Latent Extend Frames",
    "ConditioningExtendFrames": "Conditioning Extend Frames",
    "VideoLatentMask": "Video Latent Mask",
}
