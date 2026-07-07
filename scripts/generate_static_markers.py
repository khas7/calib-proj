import cv2
import numpy as np
import os
from calib_proj.sequence_generator.marker_config import MarkerConfig
from calib_proj.sequence_generator.marker_generator import MarkerGenerator
from tqdm import tqdm
import json
from pathlib import Path


marker_sizes = [45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

import argparse

class ORXMarkerSequence:
    def __init__(
        self, 
        marker_config: MarkerConfig, 
        save_path: str,
        effective_size: list[int] = [2160, 3840],
        marker_sizes: list[int] = marker_sizes,
        color_space: bool = False,
        video_len: float = 60.0,
        show_image: bool = False
    ):
        
        marker_grid_shape = marker_config.grid_shape
        max_num_marker_ids = marker_config.num_marker_ids
        assert marker_grid_shape[0] * marker_grid_shape[1] <= max_num_marker_ids

        self.marker_config = marker_config
        self.marker_generator = MarkerGenerator(marker_config)

        self.effective_size = effective_size
        self.save_path = save_path
        self.marker_sizes = marker_sizes
        self.color_space = color_space
        self.show_image = show_image
        
        self.fps = 60
        self.num_frames = int(video_len * self.fps)

        self.s_pause = 3
        self.frames_pause = self.s_pause * self.fps

    def generate_image(self, marker_size: int, conf: dict):

        safety_shift_x = int((self.marker_config.projector_resolution[1] - self.effective_size[1]) // 2)
        safety_shift_y = int((self.marker_config.projector_resolution[0] - self.effective_size[0]) // 2)
        max_x = self.marker_config.projector_resolution[1]
        max_y = self.marker_config.projector_resolution[0]

        grid_size = [int((2*self.effective_size[0] - marker_size) / (3 * marker_size)), int((2*self.effective_size[1] - marker_size) / (3 * marker_size))]
        np.random.seed(42)
        self.marker_ids = np.random.choice(np.arange(marker_config.num_marker_ids), grid_size[0] * grid_size[1], replace=False)

        distance_x = int((self.effective_size[1] - grid_size[1] * marker_size) / (grid_size[1] - 1))
        distance_y = int((self.effective_size[0] - grid_size[0] * marker_size) / (grid_size[0] - 1))

        safety_shift_x = safety_shift_x - (distance_x + marker_size) // 2
        safety_shift_y = safety_shift_y - (distance_y + marker_size) // 2

        image = np.ones((max_y, max_x), dtype=np.uint8) * 255

        i = 0
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                id = self.marker_ids[i]
                marker = self.marker_generator.generate_marker(marker_size, id)
                image[
                    safety_shift_y+row*(distance_y+marker_size):safety_shift_y+row*(distance_y+marker_size)+marker_size, 
                    safety_shift_x+col*(distance_x+marker_size):safety_shift_x+col*(distance_x+marker_size)+marker_size
                ] = marker
                i += 1

        conf["grid_size"] = grid_size
        return image, conf
    
    def generate_pause_image(self, marker_size: int):
        max_y, max_x = self.marker_config.projector_resolution[0], self.marker_config.projector_resolution[1]
        image = np.zeros((max_y, max_x), dtype=np.uint8)
        image = cv2.putText(image, f"{marker_size:03d}px", (max_x//2-700, max_y//2+100), cv2.FONT_HERSHEY_SIMPLEX, 15, 255, 40)
        return image

    def generate_video(self, marker_size: int):
        
        fourcc = cv2.VideoWriter().fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, self.marker_config.projector_resolution[::-1], self.color_space)
        conf = {
            "marker_system": "aruco_4X4_1000",
            "grid_fps": 6,
            "video_fps": 60,
            "msm_base_size": marker_size,
            "msm_scales": [1],
            "invert_colors": True,
            "shift_scale_indices": {}
        }

        pause_image = self.generate_pause_image(marker_size)
        for _ in range(self.frames_pause):
            video_writer.write(pause_image)

        image, conf = self.generate_image(marker_size, conf)
        image = 255 - image
        if self.color_space:
            black_mask = image < 128
            white_mask = image >= 128
            image = np.concatenate([image[:, :, None]]*3, axis=-1)
            image[black_mask] = [0, 0, 255]
            image[white_mask] = [255, 255, 255]
        
        if self.show_image:
            cv2.imshow("image", image)
            cv2.waitKey(0)

        for _ in tqdm(range(self.num_frames)):
            video_writer.write(image)

        with open(os.path.join(self.save_path.parent, f"{marker_size:03d}.json"), "w") as f:
            json.dump(conf, f)

        video_writer.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        help="path of the output file"
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        default=False,
        help="Shows image for debugging purposes when activated."
    )
    parser.add_argument(
        "--color_space",
        action="store_true",
        default=False,
        help="Whether to show black pixels in a red color space. This can help to maximize contrast dependent on the color of the projection surface."
    )
    parser.add_argument(
        "--marker_size",
        default="all"
    )

    args = parser.parse_args()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    target_marker_size = args.marker_size
    if isinstance(target_marker_size, str) and target_marker_size != "all":
        target_marker_size = int(target_marker_size)

    color_space = args.color_space
    show_image = args.show_image

    marker_config = MarkerConfig(
        num_marker_ids=1000,
        grid_shape=[8, 16]
    )
    effective_size_m = [0.4, 2.0] # [m] must be smaller eq [9/16*x, x] at x/2 projector distance -> corresponds to [1.2375, 2.2] m effective size @ 1.1m projector distance
    projector_height = 1.1
    m_per_px = projector_height * 2 / 3840 # [m / px] @ 1.1m proj. distance
    effective_size = [effective_size_m[0] / m_per_px, effective_size_m[1] / m_per_px]
    marker_generator = ORXMarkerSequence(
        marker_config,
        out_file,
        effective_size=effective_size,
        show_image=show_image,
        color_space=color_space,
    )

    if target_marker_size == "all":
        for marker_size in marker_sizes:
            marker_generator.generate_video(marker_size)
    else:
        marker_generator.generate_video(target_marker_size)
