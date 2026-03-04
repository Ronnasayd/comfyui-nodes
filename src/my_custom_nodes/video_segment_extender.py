import json
import logging
import math
import os
import shutil
import subprocess
import time

import folder_paths
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

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


def _extract_blended_frame(
    video_path: str,
    project_path: str,
    blend_count: int = 4,
    offset_percent: float = 90.0,
) -> torch.Tensor:
    """Extrai múltiplos frames finais de um vídeo e retorna a média para suavizar transições.

    Args:
        video_path: Caminho do vídeo
        project_path: Pasta temporária para extrair frames
        blend_count: Número de frames para fazer média (padrão: 4)
        offset_percent: Percentual do vídeo para começar extração (padrão: 90%)

    Returns:
        Tensor IMAGE [1, H, W, C] com média dos frames
    """
    # Obter duração total do vídeo
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True, check=False)
    duration = float(result.stdout.strip())

    # Calcular timestamp inicial (offset_percent do vídeo)
    start_time = duration * (offset_percent / 100.0)

    # Criar pasta temporária para frames
    frames_dir = os.path.join(project_path, "blend_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Limpar frames antigos
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    # Extrair frames a partir do offset
    frame_pattern = os.path.join(frames_dir, "frame_%03d.png")
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-vframes",
        str(blend_count),
        "-q:v",
        "1",
        "-y",
        frame_pattern,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )

    # Carregar os frames e fazer média
    frames = []
    for i in range(1, blend_count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            np_img = np.array(img).astype(np.float32) / 255.0
            frames.append(np_img)

    if not frames:
        raise RuntimeError(f"Falha ao extrair frames de: {video_path}")

    # Fazer média dos frames
    avg_frame = np.mean(frames, axis=0)
    avg_frame = np.expand_dims(avg_frame, 0)

    return torch.from_numpy(avg_frame).float()


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


def _trim_video_at_offset(
    input_path: str, output_path: str, offset_percent: float
) -> None:
    """Corta um vídeo até o ponto de offset especificado.

    Args:
        input_path: Caminho do vídeo original
        output_path: Caminho do vídeo cortado
        offset_percent: Percentual onde cortar (ex: 90.0 = manter 90% do vídeo)
    """
    # Obter duração do vídeo
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
    duration = float(result.stdout.strip())

    if not duration:
        raise RuntimeError(f"Não foi possível obter duração de: {input_path}")

    # Calcular duração cortada
    trim_duration = duration * (offset_percent / 100.0)

    # Cortar vídeo com re-encode para precisão de frame
    # Usa libx264 com preset fast para balance entre velocidade e qualidade
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        str(trim_duration),
        "-vcodec",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-avoid_negative_ts",
        "make_zero",
        output_path,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )


def _remove_frames_from_end(
    input_path: str, output_path: str, num_frames: int, fps: float
) -> None:
    """Remove N frames do final de um vídeo.

    Args:
        input_path: Caminho do vídeo original
        output_path: Caminho do vídeo resultante
        num_frames: Número de frames a remover do final
        fps: Taxa de frames por segundo do vídeo
    """
    if num_frames <= 0:
        # Se não há frames para remover, apenas copia o arquivo
        shutil.copy2(input_path, output_path)
        return

    # Obter duração total do vídeo
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
    duration = float(result.stdout.strip())

    if not duration:
        raise RuntimeError(f"Não foi possível obter duração de: {input_path}")

    # Calcular duração a manter (duração total - tempo dos frames a remover)
    time_to_remove = num_frames / fps
    new_duration = max(0.0, duration - time_to_remove)

    # Cortar vídeo mantendo apenas a duração calculada
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        str(new_duration),
        "-vcodec",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-avoid_negative_ts",
        "make_zero",
        output_path,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )


def _extract_latent_overlap(latent_dict: dict, overlap_frames: int) -> dict:
    """Extrai apenas os últimos N frames do latent para economia de memória.

    Args:
        latent_dict: Dict com chave "samples" contendo tensor latent [B, C, F, H, W]
        overlap_frames: Número de frames finais a extrair (0 = salva todos)

    Returns:
        Dict com latent contendo apenas os frames de overlap
    """
    if overlap_frames <= 0 or latent_dict is None:
        return latent_dict

    samples = latent_dict.get("samples")
    if samples is None:
        return latent_dict

    # Detectar dimensão temporal (frames)
    # Formatos possíveis: [B, C, F, H, W] ou [B, F, C, H, W]
    shape = samples.shape

    # Assumir formato [B, C, F, H, W] (formato padrão de latent de vídeo)
    if len(shape) == 5:
        num_frames = shape[2]  # F é a terceira dimensão
        if overlap_frames >= num_frames:
            # Se overlap >= total de frames, retorna tudo
            return latent_dict

        # Extrair apenas últimos N frames
        overlap_samples = samples[:, :, -overlap_frames:, :, :].contiguous()
        return {"samples": overlap_samples}
    elif len(shape) == 4:
        # Se for 4D [B, C, H, W], não é vídeo - retorna original
        logger.warning(
            f"Latent has 4 dimensions {shape}, expected 5D video latent. Saving as-is."
        )
        return latent_dict
    else:
        logger.warning(
            f"Unexpected latent shape {shape}. Expected 5D [B, C, F, H, W]. Saving as-is."
        )
        return latent_dict


def _crossfade_videos(
    folder: str,
    output_path: str,
    crossfade_frames: int,
    fps: float,
    include_initial: bool = True,
    frame_offset_percent: float = 100.0,
) -> None:
    """Concatena vídeos com transição crossfade suave usando ffmpeg xfade filter.

    Args:
        folder: Pasta do projeto contendo os segmentos
        output_path: Caminho do arquivo final
        crossfade_frames: Número de frames para transição (0 = sem crossfade)
        fps: Taxa de frames por segundo
        include_initial: Se True, inclui initial_video.mp4 no início
        frame_offset_percent: Percentual de offset para cortar segmentos
    """
    if crossfade_frames <= 0:
        # Sem crossfade, usar concatenação simples
        _concat_videos(
            folder,
            output_path,
            include_initial,
            frame_offset_percent,
            remove_duplicate_frames=1,  # valor padrão
            fps=fps,
        )
        return

    # Coletar todos os segmentos
    segments = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.startswith("segment_") and f.endswith(".mp4")
        ]
    )

    # Preparar lista de vídeos para processar
    video_files = []

    # Incluir vídeo inicial se existir
    if include_initial:
        initial_video = os.path.join(folder, "initial_video.mp4")
        if os.path.exists(initial_video):
            video_files.append(initial_video)

    # Adicionar segmentos
    for segment_file in segments:
        video_files.append(os.path.join(folder, segment_file))

    if len(video_files) == 0:
        raise RuntimeError("Nenhum vídeo encontrado para concatenar")

    if len(video_files) == 1:
        # Apenas um vídeo, copiar diretamente
        shutil.copy2(video_files[0], output_path)
        return

    # Calcular duração do crossfade em segundos
    crossfade_duration = crossfade_frames / fps

    # Construir filtro complexo para crossfade
    # Estratégia: usar xfade filter entre cada par de vídeos
    filter_parts = []
    inputs = []

    # Adicionar todos os vídeos como inputs
    for i, video_file in enumerate(video_files):
        inputs.extend(["-i", video_file])

    # Construir cadeia de xfade filters
    # [0][1]xfade[v01]; [v01][2]xfade[v012]; etc.
    current_label = "0"
    for i in range(1, len(video_files)):
        next_label = f"v{i}"
        filter_parts.append(
            f"[{current_label}][{i}]xfade=transition=fade:duration={crossfade_duration}:offset=0[{next_label}]"
        )
        current_label = next_label

    filter_complex = ";".join(filter_parts)

    # Comando ffmpeg com filtro complexo
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{current_label}]",
            "-vcodec",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
    )

    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )

    if not os.path.exists(output_path):
        raise RuntimeError("Falha ao concatenar vídeos com crossfade")


def _concat_videos(
    folder: str,
    output_path: str,
    include_initial: bool = True,
    frame_offset_percent: float = 100.0,
    remove_duplicate_frames: int = 1,
    fps: float = 16.0,
) -> None:
    """Concatena todos os segment_*.mp4 em um único vídeo final.

    Args:
        folder: Pasta do projeto contendo os segmentos
        output_path: Caminho do arquivo final
        include_initial: Se True, inclui initial_video.mp4 no início (se existir)
        frame_offset_percent: Percentual de offset usado na extração de frames.
            Se < 100%, os segmentos (exceto o último) serão cortados nesse ponto
            para evitar frames redundantes que não foram usados como base.
        remove_duplicate_frames: Número de frames a remover do final de cada segmento
            (exceto o último) para evitar duplicação nas junções. Padrão: 1
        fps: Taxa de frames por segundo
    """
    list_file = os.path.join(folder, "concat_list.txt")
    trimmed_folder = os.path.join(folder, "trimmed")
    os.makedirs(trimmed_folder, exist_ok=True)

    # Coletar todos os segmentos
    segments = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.startswith("segment_") and f.endswith(".mp4")
        ]
    )

    with open(list_file, "w", encoding="utf-8") as f:
        # Inclui vídeo inicial se existir
        if include_initial:
            initial_video = os.path.join(folder, "initial_video.mp4")
            if os.path.exists(initial_video):
                trimmed_initial = os.path.join(trimmed_folder, "initial_video.mp4")

                # Aplicar corte por offset se necessário
                if frame_offset_percent < 100.0:
                    _trim_video_at_offset(
                        initial_video, trimmed_initial, frame_offset_percent
                    )
                else:
                    # Apenas remover frames duplicados do final
                    _remove_frames_from_end(
                        initial_video, trimmed_initial, remove_duplicate_frames, fps
                    )
                f.write(f"file '{trimmed_initial}'\n")

        # Adiciona os segmentos gerados
        for i, segment_file in enumerate(segments):
            segment_path = os.path.join(folder, segment_file)
            is_last_segment = i == len(segments) - 1

            # Último segmento: usar original sem modificações
            if is_last_segment:
                f.write(f"file '{segment_path}'\n")
            else:
                # Segmentos intermediários: aplicar cortes
                trimmed_path = os.path.join(trimmed_folder, segment_file)

                if frame_offset_percent < 100.0:
                    # Cortar por offset
                    _trim_video_at_offset(
                        segment_path, trimmed_path, frame_offset_percent
                    )
                else:
                    # Apenas remover frames duplicados do final
                    _remove_frames_from_end(
                        segment_path, trimmed_path, remove_duplicate_frames, fps
                    )
                f.write(f"file '{trimmed_path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
    ]

    # Se houver cortes ou remoção de frames, fazer re-encode
    if frame_offset_percent < 100.0 or remove_duplicate_frames > 0:
        cmd.extend(
            [
                "-vcodec",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        )
    else:
        cmd.extend(
            [
                "-c",
                "copy",
                output_path,
            ]
        )

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
        - Segmento 0  → retorna a initial_image fornecida OU frame suavizado do initial_video.
        - Segmentos N → extrai e retorna frame(s) suavizado(s) do segmento anterior.

    Suavização de Transição:
        - frame_blend_count: Número de frames para fazer média (1-10, padrão: 4)
          * 1 = usa apenas último frame (sem suavização)
          * 4-6 = suavização recomendada para evitar cortes bruscos
        - frame_offset_percent: Percentual do vídeo para começar extração (50-100%, padrão: 90%)
          * 90% = pega frames de 90% até o final do vídeo
          * Evita frames que já estão "freando" o movimento

    Inputs:
        - initial_image (opcional): Imagem inicial para começar a geração
        - initial_video (opcional): Vídeo inicial - extrai frames e concatena no final
        - Se ambos fornecidos, initial_video tem prioridade
        - Se nenhum fornecido, usa imagem preta como fallback

    Fluxo sem ciclo (executar a fila N vezes):
        [VideoSegmentPrepare] ──next_image──► [Wan / gerador de vídeo]
                                                        │
                                                (vídeo gerado)
                                                        ▼
                                           [VideoSegmentSave]

    Outputs:
        next_image       : IMAGE   — frame suavizado para alimentar o gerador
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
                "fps": (
                    "FLOAT",
                    {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.01},
                ),
                "frame_blend_count": (
                    "INT",
                    {"default": 4, "min": 1, "max": 10, "step": 1},
                ),
                "frame_offset_percent": (
                    "FLOAT",
                    {"default": 90.0, "min": 50.0, "max": 100.0, "step": 1.0},
                ),
            },
            "optional": {
                "initial_image": ("IMAGE",),
                "initial_video": ("VIDEO",),
                "cache_window_size": ("INT", {"default": 4, "min": 1, "max": 20}),
                "overlap_frames": (
                    "INT",
                    {"default": 8, "min": 0, "max": 64, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "STRING", "LATENT")
    RETURN_NAMES = (
        "next_image",
        "current_segment",
        "finished",
        "final_video_path",
        "cached_latent",
    )

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
        frame_blend_count,
        frame_offset_percent,
        initial_image=None,
        initial_video=None,
        cache_window_size=4,
        overlap_frames=8,
    ):
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        # Create latent cache directory
        cache_dir = os.path.join(project_path, "latent_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Save cache_window_size to project state for VideoSegmentSave
        state_file = os.path.join(project_path, "project_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}

        state["cache_window_size"] = cache_window_size
        state["overlap_frames"] = overlap_frames
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Load cached latent from previous segment if available
        cached_latent = None
        if current_segment > 0:
            cache_path = os.path.join(
                cache_dir, f"segment_{current_segment - 1:03d}.pt"
            )
            if os.path.exists(cache_path):
                try:
                    cached_latent = torch.load(
                        cache_path, map_location="cpu", weights_only=False
                    )
                    logger.info(
                        f"Loaded cached latent from segment {current_segment - 1}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached latent from {cache_path}: {e}"
                    )
                    cached_latent = None
            else:
                logger.debug(
                    f"No cached latent found for segment {current_segment - 1}"
                )

        # Determinar a imagem inicial a ser usada
        start_image = None

        # Se recebeu vídeo inicial, processar uma única vez (segmento 0)
        if initial_video is not None and current_segment == 0:
            initial_video_path = os.path.join(project_path, "initial_video.mp4")
            # Salvar vídeo inicial apenas se ainda não existir
            if not os.path.exists(initial_video_path):
                _save_video_tensor(initial_video, initial_video_path, fps)
            # Extrair frame suavizado do vídeo inicial
            if frame_blend_count > 1:
                start_image = _extract_blended_frame(
                    initial_video_path,
                    project_path,
                    frame_blend_count,
                    frame_offset_percent,
                )
            else:
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
            return (
                start_image,
                int(current_segment),
                True,
                str(final_path),
                cached_latent,
            )

        # Primeiro segmento → usa imagem inicial (de video ou image)
        if current_segment == 0:
            return (start_image, 0, False, "", cached_latent)

        # Segmentos seguintes → extrai frame(s) suavizado(s) do segmento anterior
        last_segment_path = os.path.join(
            project_path, f"segment_{current_segment - 1:03d}.mp4"
        )

        # Usar blend se blend_count > 1, senão usar último frame direto
        if frame_blend_count > 1:
            frame = _extract_blended_frame(
                last_segment_path, project_path, frame_blend_count, frame_offset_percent
            )
        else:
            frame = _extract_last_frame(last_segment_path, project_path)

        return (frame, int(current_segment), False, "", cached_latent)


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

    Concatenação Inteligente:
        - Se frame_offset_percent < 100%, os vídeos (exceto o último) serão cortados
          no ponto de offset antes da concatenação
        - Isso remove os frames finais que não foram usados como base para o próximo segmento
        - Exemplo: offset=90% → cada vídeo é cortado em 90% antes de concatenar
        - Resultado: transição suave sem frames redundantes

    Remoção de Frames Duplicados:
        - remove_duplicate_frames: Número de frames a remover do final de cada segmento
          (exceto o último) para evitar duplicação nas junções (0-10, padrão: 1)
          * 0 = sem remoção (pode causar repetição visível)
          * 1 = remove 1 frame duplicado (recomendado para a maioria dos casos)
          * 2+ = remove mais frames se houver sobreposição maior
        - Como o modelo gera vídeos baseados no último frame do segmento anterior,
          o primeiro frame do novo segmento tende a ser muito similar ou idêntico,
          causando uma "pausa" perceptível na concatenação final
        - Remover 1 frame elimina essa repetição mantendo a fluidez do vídeo

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
                "fps": (
                    "FLOAT",
                    {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.01},
                ),
                "frame_offset_percent": (
                    "FLOAT",
                    {"default": 90.0, "min": 50.0, "max": 100.0, "step": 1.0},
                ),
                "remove_duplicate_frames": (
                    "INT",
                    {"default": 1, "min": 0, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "latent": ("LATENT",),
                "overlap_frames": (
                    "INT",
                    {"default": 8, "min": 0, "max": 64, "step": 1},
                ),
                "crossfade_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 30, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("saved_segment", "finished", "final_video_path")

    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/VideoSegment"

    def save(
        self,
        video,
        project_name,
        total_seconds,
        segment_seconds,
        fps,
        frame_offset_percent,
        remove_duplicate_frames,
        latent=None,
        overlap_frames=8,
        crossfade_frames=0,
    ):
        project_path = _get_project_path(project_name)
        max_segments = math.ceil(total_seconds / segment_seconds)
        current_segment = _count_segments(project_path)

        # Salva o novo segmento
        seg_path = os.path.join(project_path, f"segment_{current_segment:03d}.mp4")
        _save_video_tensor(video, seg_path, fps)

        # Save latent cache for temporal continuity
        if latent is not None:
            cache_dir = os.path.join(project_path, "latent_cache")
            os.makedirs(cache_dir, exist_ok=True)

            cache_path = os.path.join(cache_dir, f"segment_{current_segment:03d}.pt")

            try:
                # Extrair apenas os últimos N frames do latent para economizar memória
                overlap_latent = _extract_latent_overlap(latent, overlap_frames)
                torch.save(overlap_latent, cache_path)
                logger.info(
                    f"Saved latent overlap ({overlap_frames} frames) for segment {current_segment}"
                )
            except Exception as e:
                logger.warning(f"Failed to save latent cache to {cache_path}: {e}")

            # Cleanup old cache files beyond the window size
            state_file = os.path.join(project_path, "project_state.json")
            cache_window_size = 4  # default
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    cache_window_size = state.get("cache_window_size", 4)
                except Exception as e:
                    logger.warning(f"Failed to read cache_window_size from state: {e}")

            old_index = current_segment - cache_window_size
            if old_index >= 0:
                old_cache_path = os.path.join(cache_dir, f"segment_{old_index:03d}.pt")
                if os.path.exists(old_cache_path):
                    try:
                        os.remove(old_cache_path)
                        logger.debug(
                            f"Removed old latent cache for segment {old_index}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old cache {old_cache_path}: {e}"
                        )

        current_segment += 1

        del video
        torch.cuda.empty_cache()

        # Verifica se finalizou e concatena
        if current_segment >= max_segments:
            final_video = os.path.join(project_path, "final_video.mp4")

            # Usar crossfade se configurado, senão concatenação simples
            if crossfade_frames > 0:
                _crossfade_videos(
                    project_path,
                    final_video,
                    crossfade_frames=crossfade_frames,
                    fps=fps,
                    include_initial=True,
                    frame_offset_percent=frame_offset_percent,
                )
            else:
                _concat_videos(
                    project_path,
                    final_video,
                    include_initial=True,
                    frame_offset_percent=frame_offset_percent,
                    remove_duplicate_frames=remove_duplicate_frames,
                    fps=fps,
                )
            return (int(current_segment), True, str(final_video))

        return (int(current_segment), False, "")


# ============================================================
# Node 3 — LatentShapeDebug
# ============================================================


class LatentShapeDebug:
    """
    LatentShapeDebug
    ----------------
    Node de debug para visualizar as dimensões de um tensor latent.

    Útil para verificar se as operações de slice e concatenação de latents
    estão funcionando corretamente no pipeline de segmentos de vídeo.

    Inputs:
        latent: LATENT — tensor latent a ser inspecionado

    Outputs:
        shape_info : STRING — string formatada com as dimensões [B, C, F, H, W]
        latent     : LATENT — passa o latent adiante sem modificações
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("shape_info", "latent")

    FUNCTION = "debug_shape"
    OUTPUT_NODE = True
    CATEGORY = "MYNodes/VideoSegment"

    def debug_shape(self, latent):
        """Extrai e formata informações sobre as dimensões do latent."""
        if latent is None:
            return ("Latent is None", latent)

        samples = latent.get("samples")
        if samples is None:
            return ("No 'samples' key in latent dict", latent)

        shape = samples.shape
        shape_str = f"[{', '.join(str(d) for d in shape)}]"

        # Tentar identificar o formato
        if len(shape) == 5:
            b, c, f, h, w = shape
            info = (
                f"Shape: {shape_str}\n"
                f"Format: [B, C, F, H, W]\n"
                f"Batch: {b}, Channels: {c}, Frames: {f}, Height: {h}, Width: {w}\n"
                f"Total elements: {samples.numel():,}\n"
                f"Memory (approx): {samples.numel() * samples.element_size() / (1024**2):.2f} MB"
            )
        elif len(shape) == 4:
            b, c, h, w = shape
            info = (
                f"Shape: {shape_str}\n"
                f"Format: [B, C, H, W] (Image latent)\n"
                f"Batch: {b}, Channels: {c}, Height: {h}, Width: {w}\n"
                f"Total elements: {samples.numel():,}\n"
                f"Memory (approx): {samples.numel() * samples.element_size() / (1024**2):.2f} MB"
            )
        else:
            info = (
                f"Shape: {shape_str}\n"
                f"Dimensions: {len(shape)}\n"
                f"Total elements: {samples.numel():,}\n"
                f"Memory (approx): {samples.numel() * samples.element_size() / (1024**2):.2f} MB"
            )

        logger.info(f"LatentShapeDebug: {info.replace(chr(10), ' | ')}")
        return (info, latent)
