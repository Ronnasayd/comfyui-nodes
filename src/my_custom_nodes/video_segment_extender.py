import os
import subprocess

import cv2
import folder_paths
import numpy as np
from PIL import Image


class VideoSegmentExtender:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_image": ("IMAGE",),
                "total_seconds": ("INT", {"default": 6, "min": 2, "max": 600}),
                "segment_seconds": ("INT", {"default": 2, "min": 1, "max": 10}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
                "output_folder": ("STRING", {"default": "wan_segments"}),
            },
            "optional": {
                "last_video": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN")
    RETURN_NAMES = ("next_image", "current_segment", "finished")

    FUNCTION = "process"
    CATEGORY = "MYNodes"

    def process(
        self,
        initial_image,
        total_seconds,
        segment_seconds,
        fps,
        output_folder,
        last_video=None,
    ):

        base_output = folder_paths.get_output_directory()
        save_path = os.path.join(base_output, output_folder)
        os.makedirs(save_path, exist_ok=True)

        max_segments = total_seconds // segment_seconds

        existing_videos = sorted(
            [f for f in os.listdir(save_path) if f.endswith(".mp4")]
        )

        current_segment = len(existing_videos)

        # Se recebeu novo video, salva
        if last_video:
            new_name = f"segment_{current_segment:03d}.mp4"
            new_path = os.path.join(save_path, new_name)
            os.rename(last_video, new_path)

        # Atualiza contagem
        existing_videos = sorted(
            [f for f in os.listdir(save_path) if f.endswith(".mp4")]
        )
        current_segment = len(existing_videos)

        # Se já terminou todos segmentos
        if current_segment >= max_segments:
            final_video_path = os.path.join(save_path, "final_video.mp4")
            self.concat_videos(save_path, final_video_path)
            return (initial_image, current_segment, True)

        # Se nenhum vídeo ainda → retorna imagem inicial
        if current_segment == 0:
            return (initial_image, current_segment, False)

        # Caso contrário → extrai último frame
        last_segment_path = os.path.join(save_path, existing_videos[-1])
        frame = self.extract_last_frame(last_segment_path)

        return (frame, current_segment, False)

    def extract_last_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        np_image = np.expand_dims(np_image, 0)
        return np_image

    def concat_videos(self, folder, output_path):
        list_file = os.path.join(folder, "videos.txt")
        with open(list_file, "w") as f:
            for vid in sorted(os.listdir(folder)):
                if vid.endswith(".mp4") and vid != "final_video.mp4":
                    f.write(f"file '{os.path.join(folder, vid)}'\n")

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
        ]

        subprocess.run(cmd)
