import cv2
from .marker_config import ARUCO_VALUES, MarkerConfig
from numpy.typing import NDArray


class MarkerGenerator:

    aruco_dict: int
    dictionary: cv2.aruco.Dictionary
    
    def __init__(self, marker_config: MarkerConfig):

        self.marker_config = marker_config

        self.generate_aruco_dict()

        # generate cv2 dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict)

    def generate_aruco_dict(self):
        # get value for dictionary
        self.aruco_dict = ARUCO_VALUES[f"DICT_{self.marker_config.marker_grid}X{self.marker_config.marker_grid}_{self.marker_config.num_marker_ids}"]

    def generate_marker(self, marker_size: int, id: int, ) -> NDArray:
        return cv2.aruco.generateImageMarker(self.dictionary, id, marker_size)