from inspect import cleandoc
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

to_pil = transforms.ToPILImage()
to_tensor = transforms.PILToTensor()
import torchvision.transforms.functional as TF
import numpy as np


class MyNode:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "optional": {
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 0,  # Minimum value
                        "max": 768,  # Maximum value
                        "step": 64,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "percent_left": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "percent_right": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "pixel_height_number": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "pixel_width_left": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "pixel_width_right": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
            },
            "required": {
                "image": ("IMAGE", {"tooltip": "This is an image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image","mask")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "process_image"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "MYNodes"

    def process_image(
        self,
        image,
        height,
        percent_left,
        percent_right,
        pixel_height_number,
        pixel_width_left,
        pixel_width_right,
        alpha,
    ):
        WIDTH = height * 1.77
        if isinstance(image, torch.Tensor):
            image = image[0]  # shape: (h, w, c)
            # Convert to (c, h, w) for torchvision
            image = image.permute(2, 0, 1).contiguous()
            [C, H, W] = image.shape
            image = TF.to_pil_image(image)
        width = int(height * (W / H))
        canvas = TF.resize(image, (height, width))

        cwidth, cheight = canvas.size
        paddingTotal = int(WIDTH - cwidth)
        paddingLeft = int(paddingTotal * alpha)
        paddingRight = int(paddingTotal * (1 - alpha))

        cropLeft = int(height * (cwidth / cheight) / percent_left)
        boxLeft = (0, 0, cropLeft, height)  # Example coordinates
        roiLeft = canvas.crop(boxLeft)

        cropRight = int(height * (cwidth / cheight) / percent_right)
        boxRight = (cwidth - cropRight, 0, cwidth, height)  # Example coordinates
        roiRight = canvas.crop(boxRight)

        canvas = ImageOps.expand(
            canvas, border=(paddingLeft, 0, paddingRight, 0), fill="black"
        )
        ncanvas = np.array(canvas)

        pixelSizeXLeft = int(roiLeft.width / pixel_width_left)
        pixelSizeYLeft = int(roiLeft.height / pixel_height_number)

        smallLeft = roiLeft.resize(
            (roiLeft.width // pixelSizeXLeft, roiLeft.height // pixelSizeYLeft),
            Image.BILINEAR,
        )
        pixaletedRoiLeft = smallLeft.resize(
            (int(paddingLeft + cropLeft), height), Image.NEAREST
        )
        widthPixaletedRoiLeft, heightPixaletedRoiLeft = pixaletedRoiLeft.size

        pixaletedRoiLeft = np.array(pixaletedRoiLeft)
        ncanvas[0:height, 0 : int(paddingLeft + cropLeft), :] = pixaletedRoiLeft

        pixelSizeXRight = int(roiRight.width / pixel_width_right)
        pixelSizeYRight = int(roiRight.height / pixel_height_number)

        smallRight = roiRight.resize(
            (roiRight.width // pixelSizeXRight, roiRight.height // pixelSizeYRight),
            Image.BILINEAR,
        )
        pixaletedRoiRight = smallRight.resize(
            (int(paddingRight + cropRight), height), Image.NEAREST
        )
        widthPixaletedRoiRight, heightPixaletedRoiRight = pixaletedRoiRight.size

        pixaletedRoiRight = np.array(pixaletedRoiRight)
        ncanvas[
            0:height,
            paddingLeft
            + cwidth
            - cropRight : paddingLeft
            + cwidth
            + widthPixaletedRoiRight,
            :,
        ] = pixaletedRoiRight
        result = Image.fromarray(ncanvas)

        resized_tensor = TF.to_tensor(result)
        resized_tensor = resized_tensor.permute(1, 2, 0).contiguous()
        resized_tensor = resized_tensor.unsqueeze(0)


        H, W, C = ncanvas.shape
        mask = torch.ones((1, H, W), dtype=resized_tensor.dtype, device=resized_tensor.device)
        mask[:,0:height, paddingLeft + cropLeft : paddingLeft + cropLeft + cwidth -2*cropRight ] = 0

        return (resized_tensor, mask)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"MyNode": MyNode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"MyNode": "MyNode Node"}
