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


def _save_video_tensor(video_tensor: torch.Tensor, output_path: str, fps: int) -> None:
    """Salva tensor de vídeo como arquivo .mp4 via ffmpeg pipe."""
    tensor = video_tensor.detach().cpu()

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


def _concat_videos(folder: str, output_path: str) -> None:
    """Concatena todos os segment_*.mp4 em um único vídeo final."""
    list_file = os.path.join(folder, "concat_list.txt")

    with open(list_file, "w", encoding="utf-8") as f:
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
        - Segmento 0  → retorna a initial_image fornecida.
        - Segmentos N → extrai e retorna o último frame do segmento anterior.

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
                "initial_image": ("IMAGE",),
                "project_name": ("STRING", {"default": "wan_project"}),
                "total_seconds": ("INT", {"default": 6, "min": 2, "max": 600}),
                "segment_seconds": ("INT", {"default": 2, "min": 1, "max": 10}),
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

    def prepare(self, initial_image, project_name, total_seconds, segment_seconds):
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        # Todos os segmentos já gerados → retorna caminho do vídeo final
        if current_segment >= max_segments:
            final_video = os.path.join(project_path, "final_video.mp4")
            final_path = final_video if os.path.exists(final_video) else ""
            return (initial_image, int(current_segment), True, str(final_path))

        # Primeiro segmento → usa imagem inicial
        if current_segment == 0:
            return (initial_image, 0, False, "")

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


# Mantido para compatibilidade com workflows antigos
class VideoSegmentExtender:
    """
    VideoSegmentExtender

    Custom node para ComfyUI que permite estender vídeos longos gerando
    segmentos curtos sequenciais (ex: 2 segundos por vez), ideal para
    GPUs com pouca VRAM.

    Funcionamento:
    --------------
    - Recebe uma imagem inicial.
    - Recebe opcionalmente um vídeo recém-gerado (segmento anterior).
    - Salva cada segmento em disco dentro de uma pasta de projeto.
    - Extrai automaticamente o último frame do segmento anterior.
    - Retorna esse frame como nova imagem inicial para o próximo ciclo.
    - Repete o processo até atingir o total de segundos desejado.
    - Ao finalizar, concatena todos os segmentos em um único vídeo final.

    Fluxo típico com Wan 2.2:
    --------------------------
    1. initial_image → Wan
    2. Wan → last_video (VideoSegmentExtender)
    3. VideoSegmentExtender → next_image → Wan
    4. Loop até finalizar
    5. Node gera final_video.mp4 concatenado

    Inputs:
    -------
    initial_image : IMAGE (torch.Tensor)
        Imagem inicial no formato [B, H, W, C], float32, valores 0-1.

    project_name : STRING
        Nome da pasta dentro de ComfyUI/output onde os segmentos serão salvos.

    total_seconds : INT
        Duração total desejada do vídeo final.

    segment_seconds : INT
        Duração de cada segmento gerado por vez.

    fps : INT
        Frames por segundo utilizados ao salvar os segmentos.

    last_video : VIDEO (opcional)
        Tensor de vídeo retornado pelo modelo (ex: Wan).
        Pode estar no formato:
            - [F, H, W, C]
            - [B, F, H, W, C]
        O node automaticamente trata batch dimension.

    Outputs:
    --------
    next_image : IMAGE
        Frame que deve ser usado como entrada para o próximo segmento.

    current_segment : INT
        Número atual de segmentos já gerados.

    finished : BOOLEAN
        Indica se o total de segmentos já foi atingido.

    final_video_path : STRING
        Caminho completo do vídeo final concatenado.
        Retorna string vazia até que o processo esteja concluído.

    Características Técnicas:
    --------------------------
    - Salva vídeos usando ffmpeg via pipe (baixo uso de memória).
    - Extrai último frame usando ffmpeg (-sseof).
    - Não mantém vídeos em RAM após salvar.
    - Compatível com GPUs fracas.
    - Resistente a crash (segmentos já salvos permanecem).
    - Garante tipos compatíveis com validação do ComfyUI.

    Requisitos:
    -----------
    - ffmpeg instalado e acessível no PATH do sistema.
    - ComfyUI com suporte a tipo VIDEO.

    Observações:
    ------------
    - Sempre reiniciar o ComfyUI após modificar INPUT_TYPES ou RETURN_TYPES.
    - Caso o workflow já tenha o node antigo, remover e adicionar novamente.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_image": ("IMAGE",),
                "project_name": ("STRING", {"default": "wan_project"}),
                "total_seconds": ("INT", {"default": 6, "min": 2, "max": 600}),
                "segment_seconds": ("INT", {"default": 2, "min": 1, "max": 10}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "last_video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("next_image", "current_segment", "finished", "final_video_path")

    FUNCTION = "process"
    CATEGORY = "MYNodes"

    def process(
        self,
        initial_image,
        project_name,
        total_seconds,
        segment_seconds,
        fps,
        last_video=None,
    ):

        base_output = folder_paths.get_output_directory()
        project_path = os.path.join(base_output, project_name)
        os.makedirs(project_path, exist_ok=True)

        max_segments = math.ceil(total_seconds / segment_seconds)

        existing_segments = sorted(
            [
                f
                for f in os.listdir(project_path)
                if f.startswith("segment_") and f.endswith(".mp4")
            ]
        )

        current_segment = len(existing_segments)

        # 🔹 Se recebeu vídeo → salvar imediatamente
        if last_video is not None:

            new_name = f"segment_{current_segment:03d}.mp4"
            new_path = os.path.join(project_path, new_name)

            self.save_video_tensor(last_video, new_path, fps)

            current_segment += 1

            del last_video
            torch.cuda.empty_cache()

        # 🔹 Se terminou todos segmentos
        if current_segment >= max_segments:
            final_video = os.path.join(project_path, "final_video.mp4")
            self.concat_videos(project_path, final_video)

            return (
                initial_image,
                int(current_segment),
                bool(True),
                str(final_video),
            )

        # 🔹 Se nenhum segmento ainda
        if current_segment == 0:
            return (
                initial_image,
                int(current_segment),
                bool(False),
                "",
            )

        # 🔹 Extrair último frame
        last_segment_path = os.path.join(
            project_path, f"segment_{current_segment-1:03d}.mp4"
        )

        frame = self.extract_last_frame_ffmpeg(last_segment_path, project_path)

        return (
            frame,
            int(current_segment),
            bool(False),
            "",
        )

    # ===============================
    # SALVAR VIDEO (SUPORTE BATCH)
    # ===============================
    def save_video_tensor(self, video_tensor, output_path, fps):

        tensor = video_tensor.detach().cpu()

        # Caso venha com batch dimension
        if tensor.dim() == 5:
            # [B, F, H, W, C] → usar primeiro batch
            tensor = tensor[0]

        if tensor.dim() != 4:
            raise ValueError("Formato de VIDEO inesperado")

        # Agora esperado: [F, H, W, C]
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

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for frame in frames:
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if not os.path.exists(output_path):
            raise RuntimeError("Falha ao salvar vídeo com ffmpeg")

    # ===============================
    # EXTRAIR ÚLTIMO FRAME
    # ===============================
    def extract_last_frame_ffmpeg(self, video_path, project_path):

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
            temp_frame_path,
            "-y",
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(temp_frame_path):
            raise RuntimeError("Falha ao extrair último frame")

        img = Image.open(temp_frame_path).convert("RGB")

        np_img = np.array(img).astype(np.float32) / 255.0
        np_img = np.expand_dims(np_img, 0)

        tensor_img = torch.from_numpy(np_img).float()

        return tensor_img

    # ===============================
    # CONCATENAR VÍDEOS
    # ===============================
    def concat_videos(self, folder, output_path):

        list_file = os.path.join(folder, "concat_list.txt")

        with open(list_file, "w") as f:
            for file in sorted(os.listdir(folder)):
                if file.startswith("segment_") and file.endswith(".mp4"):
                    full_path = os.path.join(folder, file)
                    f.write(f"file '{full_path}'\n")

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            output_path,
            "-y",
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(output_path):
            raise RuntimeError("Falha ao concatenar vídeos")
