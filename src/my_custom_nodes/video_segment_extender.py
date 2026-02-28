import math
import os
import subprocess
import time

import folder_paths
import numpy as np
import torch
from PIL import Image

# ============================================================
# Helpers compartilhados
# ============================================================


def _get_project_path(project_name: str) -> str:
    base_output = folder_paths.get_output_directory()
    project_path = os.path.join(base_output, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path


def _count_segments(project_path: str) -> int:
    return len(
        [
            f
            for f in os.listdir(project_path)
            if f.startswith("segment_") and f.endswith(".mp4")
        ]
    )


def _extract_last_frame(video_path: str, project_path: str) -> torch.Tensor:
    """Extrai o último frame de um vídeo via ffmpeg e retorna tensor IMAGE."""
    temp_frame_path = os.path.join(project_path, "last_frame.png")

    cmd = [
        "ffmpeg",
        "-sseof",
        "-0.1",
        "-i",
        video_path,
        "-update",
        "1",
        "-q:v",
        "1",
        "-y",
        temp_frame_path,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )

    if not os.path.exists(temp_frame_path):
        raise RuntimeError(f"Falha ao extrair último frame de: {video_path}")

    img = Image.open(temp_frame_path).convert("RGB")
    np_img = np.array(img).astype(np.float32) / 255.0
    np_img = np.expand_dims(np_img, 0)
    return torch.from_numpy(np_img).float()


def _save_video_tensor(video_input, output_path: str, fps: int) -> None:
    """Salva VIDEO (tensor ou VideoInput) como arquivo .mp4 via ffmpeg pipe."""

    # ComfyUI moderno: VIDEO é um objeto VideoInput (VideoFromComponents / VideoFromFile)
    # com método get_components() que retorna VideoComponents.images [F, H, W, C]
    if hasattr(video_input, "get_components"):
        components = video_input.get_components()
        tensor = components.images.detach().cpu()
        # Se fps não foi especificado ou queremos preservar o original, é possível usar
        # components.frame_rate, mas respeitamos o fps passado pelo usuário.
    elif isinstance(video_input, torch.Tensor):
        tensor = video_input.detach().cpu()
    else:
        raise TypeError(f"Tipo de VIDEO não suportado: {type(video_input)}")

    if tensor.dim() == 5:
        tensor = tensor[0]  # [B, F, H, W, C] → [F, H, W, C]

    if tensor.dim() != 4:
        raise ValueError(f"Formato de VIDEO inesperado: {tensor.shape}")

    frames = (tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    height, width = frames.shape[1], frames.shape[2]

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

    if not os.path.exists(output_path):
        raise RuntimeError(f"Falha ao salvar vídeo com ffmpeg: {output_path}")


def _concat_videos(folder: str, output_path: str, include_initial: bool = True) -> None:
    """Concatena todos os segment_*.mp4 em um único vídeo final.

    Args:
        folder: Pasta do projeto contendo os segmentos
        output_path: Caminho do arquivo final
        include_initial: Se True, inclui initial_video.mp4 no início (se existir)
    """
    list_file = os.path.join(folder, "concat_list.txt")

    with open(list_file, "w", encoding="utf-8") as f:
        # Inclui vídeo inicial se existir
        if include_initial:
            initial_video = os.path.join(folder, "initial_video.mp4")
            if os.path.exists(initial_video):
                f.write(f"file '{initial_video}'\n")

        # Adiciona os segmentos gerados
        for file in sorted(os.listdir(folder)):
            if file.startswith("segment_") and file.endswith(".mp4"):
                f.write(f"file '{os.path.join(folder, file)}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )

    if not os.path.exists(output_path):
        raise RuntimeError("Falha ao concatenar vídeos")


# ============================================================
# Node 1 — VideoSegmentPrepare
# ============================================================


class VideoSegmentPrepare:
    """
    VideoSegmentPrepare
    -------------------
    Primeiro node do pipeline de extensão de vídeo (sem ciclos no grafo).

    Responsabilidade:
        Determinar qual imagem usar para gerar o PRÓXIMO segmento:
        - Segmento 0  → retorna a initial_image fornecida OU último frame do initial_video.
        - Segmentos N → extrai e retorna o último frame do segmento anterior.

    Inputs:
        - initial_image (opcional): Imagem inicial para começar a geração
        - initial_video (opcional): Vídeo inicial - extrai último frame e concatena no final
        - Se ambos fornecidos, initial_video tem prioridade
        - Se nenhum fornecido, usa imagem preta como fallback

    Fluxo sem ciclo (executar a fila N vezes):
        [VideoSegmentPrepare] ──next_image──► [Wan / gerador de vídeo]
                                                        │
                                                (vídeo gerado)
                                                        ▼
                                           [VideoSegmentSave]

    Outputs:
        next_image       : IMAGE   — frame para alimentar o gerador
        current_segment  : INT     — índice do segmento que SERÁ gerado (0-based)
        finished         : BOOLEAN — True se todos os segmentos já existem
        final_video_path : STRING  — caminho do vídeo final (quando pronto)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "wan_project"}),
                "total_seconds": ("INT", {"default": 6, "min": 2, "max": 600}),
                "segment_seconds": ("INT", {"default": 2, "min": 1, "max": 10}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "initial_image": ("IMAGE",),
                "initial_video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("next_image", "current_segment", "finished", "final_video_path")

    FUNCTION = "prepare"
    CATEGORY = "MYNodes/VideoSegment"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        """Força re-execução a cada enfileiramento para ler estado atualizado do disco."""
        return time.time()

    def prepare(
        self,
        project_name,
        total_seconds,
        segment_seconds,
        fps,
        initial_image=None,
        initial_video=None,
    ):
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        # Determinar a imagem inicial a ser usada
        start_image = None

        # Se recebeu vídeo inicial, processar uma única vez (segmento 0)
        if initial_video is not None and current_segment == 0:
            initial_video_path = os.path.join(project_path, "initial_video.mp4")
            # Salvar vídeo inicial apenas se ainda não existir
            if not os.path.exists(initial_video_path):
                _save_video_tensor(initial_video, initial_video_path, fps)
            # Extrair último frame do vídeo inicial
            start_image = _extract_last_frame(initial_video_path, project_path)
        elif initial_image is not None:
            start_image = initial_image
        else:
            # Fallback: criar imagem preta se nenhum input for fornecido
            start_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        # Todos os segmentos já gerados → retorna caminho do vídeo final
        if current_segment >= max_segments:
            final_video = os.path.join(project_path, "final_video.mp4")
            final_path = final_video if os.path.exists(final_video) else ""
            return (start_image, int(current_segment), True, str(final_path))

        # Primeiro segmento → usa imagem inicial (de video ou image)
        if current_segment == 0:
            return (start_image, 0, False, "")

        # Segmentos seguintes → extrai último frame do segmento anterior
        last_segment_path = os.path.join(
            project_path, f"segment_{current_segment - 1:03d}.mp4"
        )
        frame = _extract_last_frame(last_segment_path, project_path)
        return (frame, int(current_segment), False, "")


# ============================================================
# Node 2 — VideoSegmentSave
# ============================================================


class VideoSegmentSave:
    """
    VideoSegmentSave
    ----------------
    Segundo node do pipeline de extensão de vídeo (sem ciclos no grafo).

    Responsabilidade:
        Receber o vídeo gerado pelo modelo, salvá-lo como próximo segmento
        e, quando todos os segmentos estiverem prontos, concatená-los.

        Se houver initial_video.mp4 no projeto, ele será incluído no início
        da concatenação final: [initial_video + segment_000 + segment_001 + ...]

    Outputs:
        saved_segment    : INT     — índice do segmento salvo
        finished         : BOOLEAN — True se todos os segmentos foram gerados
        final_video_path : STRING  — caminho do vídeo final (quando pronto)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "project_name": ("STRING", {"default": "wan_project"}),
                "total_seconds": ("INT", {"default": 6, "min": 2, "max": 600}),
                "segment_seconds": ("INT", {"default": 2, "min": 1, "max": 10}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("saved_segment", "finished", "final_video_path")

    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/VideoSegment"

    def save(self, video, project_name, total_seconds, segment_seconds, fps):
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        # Salva o novo segmento
        seg_path = os.path.join(project_path, f"segment_{current_segment:03d}.mp4")
        _save_video_tensor(video, seg_path, fps)
        current_segment += 1

        del video
        torch.cuda.empty_cache()

        # Verifica se finalizou e concatena
        if current_segment >= max_segments:
            final_video = os.path.join(project_path, "final_video.mp4")
            _concat_videos(project_path, final_video)
            return (int(current_segment), True, str(final_video))

        return (int(current_segment), False, "")
