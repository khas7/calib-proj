import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import cv2

from calib_commons.data.load_calib import construct_cameras_intrinsics
from calib_commons.data.data_pickle import save_to_pickle, load_from_pickle
from calib_commons.eval_generic_scene import eval_generic_scene
from calib_commons.viz import visualization as generic_vizualization
from calib_commons.world_frame import WorldFrame
from calib_commons.observation import Observation

from calib_proj.utils import visualization
from calib_proj.core.external_calibrator import ExternalCalibrator
from calib_proj.core.config import ExternalCalibratorConfig, SolvingLevel
from calib_proj.utils.data import convert_to_correspondences
from calib_proj.preprocessing.detection import detect_marker_centers
from calib_proj.preprocessing.preprocess import order_centers, msm_centers_from_marker_centers
from calib_proj.video_generator.generate_video import load_seq_info_json
from calib_proj.synch.synch import synch
from calib_proj.synch.extract_frames import extract_frames

# random seed
np.random.seed(3)

import argparse

def parse_time(start_time, end_time) -> tuple[int]:
    if start_time is None:
        start_idx = 0
    else:
        t = start_time.split(":")
        t = float(t[0]) * 3600 + float(t[1]) * 60 + float(t[2])
        start_idx = int(t * 30)
    if end_time is None:
        end_idx = -1
    else:
        t = end_time.split(":")
        t = float(t[0]) * 3600 + float(t[1]) * 60 + float(t[2])
        end_idx = int(t * 30)

    return start_idx, end_idx

def parse_clahe(clahe_clip=None, clahe_grid: str=None):
    if clahe_clip is None and clahe_grid is None:
        return None, None
    
    if clahe_grid is None:
        return clahe_clip, (8, 8)
    
    clahe_grid = tuple([int(clahe_grid.split(",")[i].strip(" ")) for i in range(len(clahe_grid.split(",")))])
    if clahe_clip is None:
        return 2, clahe_grid

    return clahe_clip, clahe_grid

def parse_norm(alpha: int = None, beta: int = None) -> list[int]:
    if alpha is None and beta is None:
        return None
    if alpha is None:
        return [0, beta]
    if beta is None:
        return [alpha, 255]
    return [alpha, beta]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--video_folder",
    type=str,
    required=True,
    help="Path to the video folder, containing synchronized videos."
)

parser.add_argument(
    "--intrinsics_folder",
    type=str,
    required=True,
    help="Path to the intrinsics folder containing the intrinsics json files. Make sure the camera names in the intrinsics folder are the same as in the videos."
)

parser.add_argument(
    "--output_folder",
    type=str,
    default=r".\results",
    help="Definition of the output parent folder. Defaults to 'results'"
)

parser.add_argument(
    "--sequence_info_path",
    type=str,
    required=True,
    help="Path to the sequence info. Only important for the projector fps, and number of marker positions, so that frames can correctly be extracted from the videos."
)

parser.add_argument(
    "--start_time",
    type=str,
    required=True,
    help="The exact start time of the marker sequence in the video. This must be determined beforehand."
)

parser.add_argument(
    "--end_time",
    type=str,
    required=True,
    help="The exact end time of the marker sequence in the video. This must be determined beforehand."
)

parser.add_argument(
    "--alpha",
    type=int,
    default=None,
    help="Lower bound for image normalization. Defaults to None. If BETA is provided, but no ALPHA, ALPHA defaults to 0. If no ALPHA and no BETA are provided, no image normalization is applied."
)

parser.add_argument(
    "--beta",
    type=int,
    default=None,
    help="Upper bound for image normalization. Defaults to None. If ALPHA is provided, but no BETA, BETA defaults to 255. If no ALPHA and no BETA are provided, no image normalization is applied."
)

parser.add_argument(
    "--gamma",
    type=float,
    default=None,
    help="Gamma correction value. GAMMA > 1.0 brightens the image, GAMMA < 1.0 darkens the image. Defaults to None."
)

parser.add_argument(
    "--clahe_clip",
    type=int,
    default=None,
    help="Clahe clip limit for contrast enhancement using CLAHE. Defaults to None."
)

parser.add_argument(
    "--clahe_grid",
    type=str,
    default=None,
    help="Clahe grid size for contrast enhancement using CLAHE. Defaults to None."
)

parser.add_argument(
    "--debug_preprocessing",
    action="store_true",
    default=False,
    help="Shows the image before and after preprocessing, including number of marker detections in an image."
)

parser.add_argument(
    "--save_scene",
    action="store_true",
    default=False,
    help="Saves the scene as a pickle file in the output directory."
)

args = parser.parse_args()

start_idx, end_idx = parse_time(args.start_time, args.end_time)

clahe_clip, clahe_grid = parse_clahe(args.clahe_clip, args.clahe_grid)
norm_bounds = parse_norm(args.alpha, args.beta)
clahe = None
if clahe_clip is None and clahe_grid is None and norm_bounds is None and args.gamma is None:
    preprocess_images = False
else:
    preprocess_images = True

if clahe_clip is not None and clahe_grid is not None:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

if args.video_folder is None:
    raise ValueError("Please provide a path for gopro videos '--video_folder'")
if args.intrinsics_folder is None:
    raise ValueError("Please provide a path for intrinsics '--intrinsics_path")
if args.sequence_info_path is None:
    raise ValueError("Please provide a path for sequence_info_path '--sequence_info_path'")

############################### USER INTERFACE ####################################
# PATHS
videos_folder = Path(args.video_folder)
intrinsics_folder = Path(args.intrinsics_folder)

# CALIBRATION PARAMETERS
external_calibrator_config = ExternalCalibratorConfig(
    reprojection_error_threshold = 1,
    camera_score_threshold = 200, 
    verbose = 2, # 0: only final report, 1: only camera name when added, 2: full verbose
    least_squares_verbose = 0, # 0: silent, 1: report only final results, 2: report every iteration
)

# PRE-PROCESSING PARAMETERS
show_detection_images = False
save_detection_images = True
show_viz = False
save_viz = True
save_eval_metrics_to_json = True
save_scene = args.save_scene
save_final_correspondences = False

############################### END USER INTERFACE ####################################

out_folder_calib = Path(args.output_folder)
sequence_info_path = Path(args.sequence_info_path)

###################### IMAGES PARENT FOLDER ###########################
if preprocess_images:
    images_parent_folder = out_folder_calib / ".." / "frames"
else:
    images_parent_folder = out_folder_calib / "frames"

###################### FRAMES EXTRACTION ###########################
if not images_parent_folder.exists():
    extract_frames(videos_folder, sequence_info_path, start_idx, end_idx, images_parent_folder)

###################### PRE-PROCESSING: MARKER DETECTION ###########################
correspondences_path = out_folder_calib / "correspondences.json"
if not correspondences_path.exists():
    seq_info = load_seq_info_json(sequence_info_path)
    centers_unordered_path = out_folder_calib / "preprocessing" / "centers_unordered.pkl"
    centers_unordered = detect_marker_centers(images_parent_folder,
                                            intrinsics_folder,
                                            marker_system=seq_info['marker_system'],
                                            inverted_projections=seq_info['invert_colors'],
                                            show_detections=show_detection_images,
                                            normalization=norm_bounds,
                                            gamma=args.gamma,
                                            clahe=clahe,
                                            debug_preprocessing=args.debug_preprocessing)
    # save_to_pickle(centers_unordered_path, centers_unordered)
    # centers_unordered = load_from_pickle(centers_unordered_path)

    centers_ordered = order_centers(centers_unordered, seq_info)
    msm_centers = msm_centers_from_marker_centers(centers_ordered)


    ###################### EXTERNAL CALIBRATION ###########################
    correspondences = convert_to_correspondences(msm_centers)
    save_corr = True
else:
    with open(correspondences_path, "r") as f:
        corr = json.load(f)
    correspondences = {}
    save_corr = False
    for cam in corr.keys():
        correspondences[cam] = {}
        for id, obs in corr[cam].items():
            correspondences[cam][id] = Observation(_2d=np.array(obs))

out_folder_calib.mkdir(parents=True, exist_ok=True)
intrinsics = construct_cameras_intrinsics(images_parent_folder, intrinsics_folder)

external_calibrator = ExternalCalibrator(correspondences=correspondences,
                                        intrinsics=intrinsics,
                                        config=external_calibrator_config,
                                        save_corr=save_corr,
                                        out_path=out_folder_calib
                                        )
print(f"\nCalibration started...")
success = external_calibrator.calibrate()
if success:
    scene_proj_estimate = external_calibrator.get_scene(world_frame=WorldFrame.CAM_FIRST_CHOOSEN)
    generic_scene = scene_proj_estimate.generic_scene
    generic_obsv = external_calibrator.correspondences
    generic_scene.print_cameras_poses()

    # Save files
    generic_scene.save_cameras_poses_to_json(out_folder_calib / "camera_poses.json")
    print("camera poses saved to", out_folder_calib / "camera_poses.json")

    if save_scene:
        scene_estimate_file = out_folder_calib / "scene_estimate.pkl"
        save_to_pickle(scene_estimate_file, generic_scene)
        print("scene estimate saved to", scene_estimate_file)
    if save_final_correspondences:
        correspondences_file = out_folder_calib / "correspondences.pkl"
        save_to_pickle(correspondences_file, generic_obsv)
        print("correspondences saved to", correspondences_file)
    metrics = eval_generic_scene(generic_scene, generic_obsv, camera_groups=None, save_to_json=save_eval_metrics_to_json, output_path=out_folder_calib / "metrics.json", print_ = True)
    print("")

    # Visualization
    if show_viz or save_viz:
        dpi = 300
        save_path = out_folder_calib / "scene.png"
        visualization.visualize_scenes(scene_proj_estimate, show_ids=False, show_fig=show_viz, save_fig=save_viz, save_path=save_path)
        # visualization.visualize_scenes([checkerboard_scene_estimate], show_ids=False, show_fig=show_viz, save_fig=save_viz, save_path=save_path)
        if save_viz:
            print("scene visualization saved to", save_path)
        save_path = out_folder_calib / "2d.png"
        visualization.visualize_2d(scene=scene_proj_estimate, observations=generic_obsv, show_only_points_with_both_obsv_and_repr=0,  show_ids=False, which="both", show_fig=show_viz, save_fig=save_viz, save_path=save_path)

        if save_viz:
            print("2d visualization saved to", save_path)
        save_path = out_folder_calib / "2d_errors.png"
        generic_vizualization.plot_reprojection_errors(scene_estimate=generic_scene,
                                            observations=generic_obsv,
                                            show_fig=show_viz,
                                            save_fig=save_viz,
                                            save_path=save_path)
        if save_viz:
            print("2d errors visualization saved to", save_path)

        if show_viz:
            plt.show()

