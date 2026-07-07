import cv2
import numpy as np
import os
from calib_proj.sequence_generator.marker_config import MarkerConfig
from calib_proj.sequence_generator.marker_generator import MarkerGenerator
from tqdm import tqdm
import json
from pathlib import Path

import argparse

class ORXMarkerSequence:
    def __init__(
        self, 
        marker_config: MarkerConfig, 
        save_path: str,
        effective_size: list[int] = [2160, 3840],
        marker_sizes: list[int] = [45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
        frames_ms: float = 100.0, #ms
        num_shifts_x: int = 20,
        num_shifts_y: int = 20,
        color_space: bool = False,
        show_image: bool = False
    ):
        
        marker_grid_shape = marker_config.grid_shape # The grid shape of the whole projected image
        max_num_marker_ids = marker_config.num_marker_ids # The maximum number of marker ids (aruco dictionary)
        assert marker_grid_shape[0] * marker_grid_shape[1] <= max_num_marker_ids # Number of markers in the projected image cannot exceed the max number of markers from the dictionary

        self.marker_config = marker_config
        self.marker_generator = MarkerGenerator(marker_config)

        self.effective_size = effective_size # effective size from first marker to last marker in pixel
        self.save_path = save_path # path to save the output video
        self.marker_sizes = marker_sizes
        self.num_shifts_x = num_shifts_x # number of shifts in horizontal direction
        self.num_shifts_y = num_shifts_y # number of shifts in vertical direction
        self.color_space = color_space # whether to use a color space for black pixels (this can increase the contrast dependent on the color of the projection surface)
        self.show_image = show_image
        
        self.fps = 60 # projector fps
        ms_per_frame = 1.0 / self.fps * 1000.0
        assert frames_ms >= ms_per_frame
        self.num_frames = int(frames_ms / ms_per_frame) # number of frames to show a static marker image

        self.s_pause = 3 # [s], time to show pixel size before sequence starts
        self.frames_pause = self.s_pause * self.fps

    def generate_image(self, marker_size: int, shift_x: int, shift_y: int, shift_idx: int, conf: dict):

        max_x = self.marker_config.projector_resolution[1]
        max_y = self.marker_config.projector_resolution[0]

        # new grid size dependent on the effective size and marker size
        grid_size = [int((2*self.effective_size[0] - marker_size) / (3 * marker_size)), int((2*self.effective_size[1] - marker_size) / (3 * marker_size))]
        np.random.seed(42) # set seed for reproduction of code
        # choose random marker ids
        self.marker_ids = np.random.choice(np.arange(marker_config.num_marker_ids), grid_size[0] * grid_size[1], replace=False)

        # distance between markers in pixel
        distance_x = int((self.effective_size[1] - grid_size[1] * marker_size) / (grid_size[1] - 1))
        distance_y = int((self.effective_size[0] - grid_size[0] * marker_size) / (grid_size[0] - 1))

        # shift from left / top
        safety_shift_x = int((max_x - self.effective_size[1]) // 2) - (distance_x + marker_size) // 2
        safety_shift_y = int((max_y - self.effective_size[0]) // 2) - (distance_y + marker_size) // 2

        # the number of pixels to shift in x / y direction
        shift_x_px = int((distance_x + marker_size) / self.num_shifts_x * shift_x)
        shift_y_px = int((distance_y + marker_size) / self.num_shifts_y * shift_y)

        image = np.ones((max_y, max_x), dtype=np.uint8) * 255

        i = 0
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                # place markers in image
                id = self.marker_ids[i]
                marker = self.marker_generator.generate_marker(marker_size, id)
                image[
                    safety_shift_y+row*(distance_y+marker_size)+shift_y_px:safety_shift_y+row*(distance_y+marker_size)+shift_y_px+marker_size, 
                    safety_shift_x+col*(distance_x+marker_size)+shift_x_px:safety_shift_x+col*(distance_x+marker_size)+shift_x_px+marker_size
                ] = marker
                i += 1
        
        # write config file
        conf["shift_scale_indices"][str(shift_idx+1)] = [shift_idx, 1]
        conf["grid_size"] = grid_size

        return image, conf
    
    def generate_pause_image(self, marker_size: int):
        max_y, max_x = self.marker_config.projector_resolution[0], self.marker_config.projector_resolution[1]
        image = np.zeros((max_y, max_x), dtype=np.uint8)
        image = cv2.putText(image, f"{marker_size:03d}px", (max_x//2-700, max_y//2+100), cv2.FONT_HERSHEY_SIMPLEX, 15, 255, 40)
        return image


    def generate_video(self):
        
        fourcc = cv2.VideoWriter().fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, self.marker_config.projector_resolution[::-1], self.color_space)
            
        for marker_size in self.marker_sizes:
            shift_idx = 0
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

            for shift_y in tqdm(range(self.num_shifts_y)):
                for shift_x in range(self.num_shifts_x):
                    image, conf = self.generate_image(marker_size, shift_x, shift_y, shift_idx, conf)
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

                    for _ in range(self.num_frames):
                        video_writer.write(image)
                    
                    shift_idx += 1

            with open(os.path.join(self.save_path.parent, f"{marker_size:03d}.json"), "w") as f:
                json.dump(conf, f)

        video_writer.release()


def parse_size(size: str) -> list[float]:
    l = size.split(",")
    size = [float(s.strip(" ")) for s in l]
    if len(size) != 2:
        raise ValueError("Please provide a size of length 2, e.g. '0.4, 2.0'")
    return size

def parse_marker_size(marker_sizes: str) -> list[int]:
    if marker_sizes.lower() == "all":
        return [45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    
    ms = marker_sizes.split(",")
    return [int(m.strip(" ")) for m in ms]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        help="Out path to store the generated video"
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        default=False,
        help="Shows the first image when activated. For debugging purposes."
    )
    parser.add_argument(
        "--projector_height",
        type=float,
        required=False,
        default=1.1,
        help="Projector height above projection surface. Defaults to 1.1m"
    )
    parser.add_argument(
        "--effective_size",
        type=str,
        required=False,
        default="0.4, 2.0",
        help="The target size of the projected image in meters [y, x]. Defaults to [0.4m, 2.0m]"
    )
    parser.add_argument(
        "--color_space",
        action="store_true",
        required=False,
        default=False,
        help="Whether black pixels should be projected in a red color space. This can help to maximize the contrast dependent on the color of the surface."
    )
    parser.add_argument(
        "--marker_sizes",
        required=False,
        type=str,
        default="all",
        help="The marker sizes to generate. Defaults to 'all'"
    )

    args = parser.parse_args()
    out_file = Path(args.out_file)
    if out_file.suffix not in [".avi", ".AVI", ".mp4", ".MP4"]:
        raise ValueError(f"Out file does not end with '.mp4'")
    
    out_file.parent.mkdir(parents=True, exist_ok=True)

    color_space = args.color_space
    show_image = args.show_image

    marker_config = MarkerConfig(
        num_marker_ids=1000,
        grid_shape=[8, 16]
    )
    effective_size_m = parse_size(args.effective_size) # [m] must be smaller eq [9/16*x, x] at x/2 projector distance
    projector_height = args.projector_height
    assert projector_height * 2 >= effective_size_m[1]
    assert projector_height * 2 / marker_config.projector_resolution[1] * marker_config.projector_resolution[0] >= effective_size_m[0]
    m_per_px = projector_height * 2 / marker_config.projector_resolution[1] # [m / px] 
    effective_size = [effective_size_m[0] / m_per_px, effective_size_m[1] / m_per_px] # effective size in pixel

    marker_sizes = parse_marker_size(args.marker_sizes)
    marker_generator = ORXMarkerSequence(
        marker_config,
        out_file,
        effective_size=effective_size,
        frames_ms=100,
        show_image=show_image,
        color_space=color_space,
        num_shifts_x=20,
        num_shifts_y=20,
    )

    marker_generator.generate_video()
