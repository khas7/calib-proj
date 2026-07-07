import cv2
import os
from calib_proj.video_generator.generate_video import load_seq_info_json
from tqdm import tqdm
import numpy as np
import subprocess
from pathlib import Path

def extract_video_frames_ffmpeg(
        video_path: str,
        output_folder: str,
        effective_start_idx: int,
        relative_frame_indices: list[int]
):
    
    output_folder.mkdir(parents=True, exist_ok=True)

    for grid_id in range(1, len(relative_frame_indices) + 1):
        output_file = output_folder / f"{grid_id:06d}.png"
        output_file.unlink(missing_ok=True)

    input_arguments = [
        "-i", str(video_path),
    ]
    ffmpeg_frame_indices = [
        effective_start_idx + relative_idx
        for relative_idx in relative_frame_indices
    ]

    select_expression = "+".join(
        f"eq(n\\,{frame_idx})"
        for frame_idx in ffmpeg_frame_indices
    )

    output_pattern = output_folder / "%06d.png"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        *input_arguments,
        "-map", "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf", f"select={select_expression}",
        # Do not duplicate selected frames to create a constant frame rate.
        "-fps_mode", "passthrough",
        # Stop decoding after all requested images were produced.
        "-frames:v", str(len(relative_frame_indices)),
        "-start_number", "1",
        str(output_pattern),
    ]
    
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        shell=True
    )

    missing_files = [
        output_folder / f"{grid_id:06d}.png"
        for grid_id in range(1, len(relative_frame_indices) + 1)
        if not (output_folder / f"{grid_id:06d}.png").exists()
    ]

    if missing_files:
        raise RuntimeError(
            f"FFmpeg extracted too few frames from {video_path}. "
            f"Missing {len(missing_files)} output files."
        )

def extract_frames(videos_folder, sequence_info_path, start_idx, end_idx, offset, out_folder_path):

    effective_start_idx = start_idx + (offset or 0)
    videos_path = {}
    for video_file in os.listdir(videos_folder):
        if video_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_path = os.path.join(videos_folder, video_file)
            videos_path[video_file.split('.')[0]] = video_path

    seq_info = load_seq_info_json(sequence_info_path)
    
    n_grids = len(seq_info["shift_scale_indices"])

    frame_span = end_idx - start_idx
    frames_per_grid = frame_span / n_grids


    relative_frame_indices = [
        round((grid_idx + 0.5) * frames_per_grid)
        for grid_idx in range(n_grids)
    ]

    videos_path = {}
    for video_file in os.listdir(videos_folder):
        if video_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_path = os.path.join(videos_folder, video_file)
            videos_path[video_file.split('.')[0]] = Path(video_path)

    def process_video(video_path: str, camera_name: str) -> str:
        camera_output = out_folder_path /  camera_name

        extract_video_frames_ffmpeg(
            video_path=video_path,
            output_folder=camera_output,
            effective_start_idx=effective_start_idx,
            relative_frame_indices=relative_frame_indices,
        )

        return camera_name
    
    for cam_name, video_path in videos_path.items():
        camera_name = process_video(video_path, cam_name)
        print(f"Processed {camera_name}")

    return None