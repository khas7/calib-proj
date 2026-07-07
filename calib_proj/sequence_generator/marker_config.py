import cv2
from typing import Sequence

ARUCO_VALUES = {
    name: value
    for name, value in cv2.aruco.__dict__.items()
    if name.startswith("DICT_")
}

class MarkerConfig:
    marker_grid: int
    num_marker_ids: int
    projector_resolution: Sequence[int]
    grid_shape: Sequence[int]
    
    def __init__(
            self,
            marker_grid: int = 4,
            num_marker_ids: int = 50,
            projector_resolution: Sequence[int] = [2160, 3840],
            grid_shape: Sequence[int] = [4, 8]
    ):
        
        if marker_grid not in [4, 5, 6, 7] or num_marker_ids not in [50, 100, 250, 1000]:
            raise ValueError(f"Marker configuration {marker_grid = } and {num_marker_ids = } not allowed")
        
        if not isinstance(projector_resolution, Sequence) or len(projector_resolution) != 2:
            raise ValueError(f"{projector_resolution = } not allowed. Only Sequences with len = 2 allowed")
        
        self.marker_grid = marker_grid
        self.num_marker_ids = num_marker_ids
        self.projector_resolution = projector_resolution
        self.grid_shape = grid_shape

        for i, r in enumerate(self.projector_resolution):
            if not isinstance(r, int):
                self.projector_resolution[i] = int(r)