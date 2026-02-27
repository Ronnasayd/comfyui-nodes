import os
import subprocess

import folder_paths
import numpy as np
import torch
from PIL import Image


class VideoSegmentExtender:

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

        max_segments = total_seconds // segment_seconds

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

            # liberar memória
            del last_video
            torch.cuda.empty_cache()

        # 🔹 Se terminou todos segmentos
        if current_segment >= max_segments:
            final_video = os.path.join(project_path, "final_video.mp4")
            self.concat_videos(project_path, final_video)
            return (initial_image, current_segment, True, final_video)

        # 🔹 Se nenhum segmento ainda
        if current_segment == 0:
            return (initial_image, current_segment, False, "")

        # 🔹 Extrair último frame do último segmento salvo
        last_segment_path = os.path.join(
            project_path, f"segment_{current_segment-1:03d}.mp4"
        )

        frame = self.extract_last_frame_ffmpeg(last_segment_path, project_path)

        return (frame, current_segment, False, "")

    def save_video_tensor(self, video_tensor, output_path, fps):

        tensor = video_tensor.detach().cpu()

        # Se vier com batch dimension
        if tensor.dim() == 5:
            # [B, F, H, W, C] → pegar primeiro batch
            tensor = tensor[0]

        # Agora deve estar [F, H, W, C]
        if tensor.dim() != 4:
            raise ValueError("Formato de VIDEO inesperado")

        frames = (tensor.numpy() * 255).astype(np.uint8)

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

        img = Image.open(temp_frame_path).convert("RGB")
        np_img = np.array(img).astype(np.float32) / 255.0
        np_img = np.expand_dims(np_img, 0)

        return np_img

    def concat_videos(self, folder, output_path):

        list_file = os.path.join(folder, "concat_list.txt")

        with open(list_file, "w") as f:
            for file in sorted(os.listdir(folder)):
                if file.startswith("segment_") and file.endswith(".mp4"):
                    f.write(f"file '{os.path.join(folder, file)}'\n")

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
