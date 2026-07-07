import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from calib_commons.data.load_calib import construct_cameras_intrinsics
from calib_commons.data.data_pickle import save_to_pickle, load_from_pickle
from calib_commons.eval_generic_scene import eval_generic_scene
from calib_commons.viz import visualization as generic_vizualization
from calib_commons.world_frame import WorldFrame

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

############################### USER INTERFACE ####################################
# PATHS
videos_folder = Path(r"C:\Users\timfl\Documents\test_calibProj\video")
intrinsics_folder = Path(r"C:\Users\timfl\Documents\test_calibProj\intrinsics_tim\calibrate_intrinsics_output\camera_intrinsics")

# CALIBRATION PARAMETERS
external_calibrator_config = ExternalCalibratorConfig(
    reprojection_error_threshold = 1,
    camera_score_threshold = 200, 
    verbose = 1, # 0: only final report, 1: only camera name when added, 2: full verbose
    least_squares_verbose = 0, # 0: silent, 1: report only final results, 2: report every iteration
)

# PRE-PROCESSING PARAMETERS
show_detection_images = False
save_detection_images = False
show_viz = True
save_viz = False
save_eval_metrics_to_json = True
save_scene = False
save_final_correspondences = False

############################### END USER INTERFACE ####################################



out_folder_calib = Path("results")
sequence_info_path = Path(r"video\seq_info.json")



###################### TEMPORAL SYNCHRONIZATION ###########################
start_end_frames = synch(videos_folder, sequence_info_path, threshold=0.6)
images_parent_folder = Path(videos_folder) / "frames"

###################### FRAMES EXTRACTION ###########################
extract_frames(videos_folder, sequence_info_path, start_end_frames, images_parent_folder)

###################### PRE-PROCESSING: MSMs DETECTION ###########################
seq_info = load_seq_info_json(sequence_info_path)
centers_unordered_path = out_folder_calib / "preprocessing" / "centers_unordered.pkl"
centers_unordered = detect_marker_centers(images_parent_folder,
                                          intrinsics_folder,
                                          marker_system=seq_info['marker_system'],
                                          inverted_projections=seq_info['invert_colors'],
                                          show_detections=show_detection_images)
# save_to_pickle(centers_unordered_path, centers_unordered)
# centers_unordered = load_from_pickle(centers_unordered_path)

centers_ordered = order_centers(centers_unordered, seq_info)
msm_centers = msm_centers_from_marker_centers(centers_ordered)


###################### EXTERNAL CALIBRATION ###########################
out_folder_calib.mkdir(parents=True, exist_ok=True)
intrinsics = construct_cameras_intrinsics(images_parent_folder, intrinsics_folder)
correspondences = convert_to_correspondences(msm_centers)
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

