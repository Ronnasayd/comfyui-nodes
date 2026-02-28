from inspect import cleandoc


class AspectRatioCrop:
    """
    Corta uma imagem quadrada para ter o mesmo aspect ratio de uma imagem base.

    Recebe uma imagem base e uma imagem quadrada (geralmente com padding aplicado),
    e retorna a imagem quadrada cortada do centro para manter o mesmo aspect ratio
    da imagem base.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Retorna a configuração dos campos de entrada.

        Returns:
            dict: Configuração com duas imagens obrigatórias
        """
        return {
            "required": {
                "base_image": ("IMAGE",),
                "padded_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop_to_aspect_ratio"
    CATEGORY = "MYNodes"
    DESCRIPTION = cleandoc(__doc__)

    def crop_to_aspect_ratio(self, base_image, padded_image):
        """
        Corta a imagem quadrada para ter o mesmo aspect ratio da imagem base.

        Args:
            base_image: Tensor [B, H, W, C] - imagem de referência para o aspect ratio
            padded_image: Tensor [B, H, W, C] - imagem quadrada a ser cortada

        Returns:
            tuple: (cropped_image,) - imagem cortada com mesmo aspect ratio da base
        """
        # Obter dimensões da imagem base (formato ComfyUI: [B, H, W, C])
        base_height = base_image.shape[1]
        base_width = base_image.shape[2]

        # Calcular aspect ratio da imagem base
        aspect_ratio = base_width / base_height

        # Obter dimensões da imagem quadrada
        padded_height = padded_image.shape[1]
        padded_width = padded_image.shape[2]

        # Calcular novas dimensões mantendo o aspect ratio da base
        if aspect_ratio > 1:
            # Imagem base é mais larga que alta (landscape)
            # Manter largura da imagem quadrada, ajustar altura
            new_width = padded_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Imagem base é mais alta que larga (portrait) ou quadrada
            # Manter altura da imagem quadrada, ajustar largura
            new_height = padded_height
            new_width = int(new_height * aspect_ratio)

        # Garantir que as dimensões não excedam a imagem quadrada
        if new_height > padded_height:
            new_height = padded_height
            new_width = int(new_height * aspect_ratio)
        if new_width > padded_width:
            new_width = padded_width
            new_height = int(new_width / aspect_ratio)

        # Calcular posição de corte (centro da imagem)
        start_y = (padded_height - new_height) // 2
        start_x = (padded_width - new_width) // 2

        # Aplicar crop para todos os frames/batches
        cropped_image = padded_image[
            :,
            start_y : start_y + new_height,
            start_x : start_x + new_width,
            :,
        ]

        return (cropped_image,)
