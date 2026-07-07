import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
from enum import Enum

from calib_commons.data.load_calib import load_intrinsics


from calib_proj.preprocessing.marker_systems import detect_aruco_markers, detect_apriltag_markers_detector



# class MarkerSystems(Enum): 
#     ARUCO = "aruco"
#     APRILTAG = "apriltag"


def intersection_diagonales(points):
    # points est un tableau numpy de forme (4, 2) représentant les 4 sommets du quadrilatère
    # Les points sont dans l'ordre [A, B, C, D]
    
    def line_intersection(p1, p2, p3, p4):
        """Calculer l'intersection entre deux lignes définies par les points p1, p2 et p3, p4."""
        # Représentation paramétrique des deux lignes : p1 + t*(p2-p1) et p3 + u*(p4-p3)
        A1 = p2 - p1
        A2 = p4 - p3
        b = p3 - p1
        
        # Résoudre le système d'équations linéaires pour t et u
        A = np.array([A1, -A2]).T
        t_u = np.linalg.solve(A, b)
        t = t_u[0]
        
        # Calculer l'intersection
        intersection = p1 + t * A1
        return intersection

    # Extraire les points A, B, C, D du tableau de points
    A, B, C, D = points[0], points[1], points[2], points[3]
    
    # Calcul de l'intersection des diagonales AC et BD
    point_intersection = line_intersection(A, C, B, D)
    return point_intersection


def detect_marker_centers(images_parent_folder, 
                   intrinsics_folder, 
                   marker_system: str,
                   inverted_projections = True,
                   show_detections = False,
                   normalization: list[int] | None = None,
                   gamma: float | None = None,
                   clahe: cv.CLAHE = None,
                   debug_preprocessing = False
                   ): 
    
    print(f"\nDetecting markers started...")

    
    # if marker_system not in MarkerSystems:
    #     raise ValueError(f"Unknown marker system: {marker_system}.")
    
    marker_system_name, *rest = marker_system.split('_', 1)
    dict = rest[0] 

    centers = {}

    if marker_system_name == "apriltag":
        
        os.add_dll_directory("C:/Users/timfl/miniconda3/envs/master_thesis/Lib/site-packages/pupil_apriltags/lib")
        os.add_dll_directory("C:/Program Files/MATLAB/R2023b/bin/win64")
        from pupil_apriltags import Detector

        at_detector = Detector(
                    families=dict, # tag16h5, tag25h9,tag36h11
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0
                )
        
    image_folders = {cam: os.path.join(images_parent_folder, cam) for cam in os.listdir(images_parent_folder) if os.path.isdir(os.path.join(images_parent_folder, cam))}
    intrinsics_paths = {cam: os.path.join(intrinsics_folder, cam + "_intrinsics.json") for cam in image_folders}    

    for cam, image_folder in tqdm(image_folders.items(), desc=f"Markers detection", total=len(image_folders), leave=False): 
        centers[cam] = {}

        camera_matrix, distortion_coeffs = load_intrinsics(intrinsics_paths[cam])
        
        files = os.listdir(image_folder)
        files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=lambda x: int(x.split('.')[0]))
        
        for filename in tqdm(files, desc=f"Processing {cam} ...", total=len(files), leave=False):        
            k = int(filename.split('.')[0])
            centers[cam][k] = {}

            file_path = os.path.join(image_folder, filename)
            # print(filename)

            img = cv.imread(file_path)
            before = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
            if normalization is not None:
                img = cv.normalize(img, None, alpha=normalization[0], beta=normalization[1], norm_type=cv.NORM_MINMAX)
            
            if gamma is not None:
                img = np.clip(np.power(img.astype(np.float32) / 255, gamma) * 255, 0, 255).astype(np.uint8)

            if clahe is not None:
                img = clahe.apply(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
            
            if debug_preprocessing:
                deb = np.hstack((before, img))
                cv.imshow("Before and after preprocessing", cv.resize(deb, (deb.shape[1]//8, deb.shape[0]//8)))
                cv.waitKey(0)

            if inverted_projections: 
                img = cv.bitwise_not(img)
            
            if marker_system_name == 'aruco':
                # dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
                markers, img_draw = detect_aruco_markers(img, dict, show_draw_img=show_detections, add_half_pixel_shift=False)
                if debug_preprocessing:
                    print(f"{len(markers) = }")
                    markers_before, _ = detect_aruco_markers(cv.bitwise_not(before), dict, show_draw_img=show_detections, add_half_pixel_shift=False)
                    print(f"{len(markers_before) = }")
            elif marker_system_name == 'apriltag':
                markers, img_draw = detect_apriltag_markers_detector(img, at_detector, show_draw_img=show_detections)

            for marker_id, _4corners in markers.items():
                    points_reshaped = _4corners.reshape(-1, 1, 2)
                    undistorted_points = cv.undistortPoints(points_reshaped, camera_matrix, distortion_coeffs, P=camera_matrix)
                    corners_undistorted = undistorted_points.reshape(-1, 2)                
                    
                    center = intersection_diagonales(corners_undistorted)
                    centers[cam][k][marker_id] = center

    print(f"Detected markers successfully.")
    return centers