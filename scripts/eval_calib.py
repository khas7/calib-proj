import json
import os
from pathlib import Path
from numpy.typing import NDArray

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

import argparse

def load_json(gt_path: str, est_path: str) -> tuple[dict]:

    with open(gt_path, "r") as f:
        gt_poses: dict = json.load(f)
    
    with open(est_path, "r") as f:
        est_poses: dict = json.load(f)

    gt_cams = set(gt_poses)
    est_cams = set(est_poses)

    gt_missing = est_cams - gt_cams
    est_missing = gt_cams - est_cams

    for cam in gt_missing:
        est_poses.pop(cam)
    
    for cam in est_missing:
        gt_poses.pop(cam)
    
    if not len(gt_poses) or not len(est_poses):
        raise ValueError("No cam correspondences found in pose files")
    
    return gt_poses, est_poses

def get_rotation_and_translation(gt_poses: dict, est_poses: dict) -> tuple[dict[str, NDArray]]:
    R_gt_wc = {}
    t_gt_wc = {}
    R_est_wc = {}
    t_est_wc = {}

    for cam, gt_pose in gt_poses.items():
        
        _R_gt_wc = SciRotation.from_euler("ZYX", gt_pose["euler_ZYX"], True).as_matrix()
        _t_gt_wc = np.asarray(gt_pose["t"], dtype=float)

        est_pose = est_poses[cam]

        _R_est_wc = SciRotation.from_euler("ZYX", est_pose["euler_ZYX"], True).as_matrix()
        _t_est_wc = np.asarray(est_pose["t"], dtype=float)

        R_gt_wc[cam] = _R_gt_wc
        t_gt_wc[cam] = _t_gt_wc
        R_est_wc[cam] = _R_est_wc
        t_est_wc[cam] = _t_est_wc

    return R_gt_wc, t_gt_wc, R_est_wc, t_est_wc

def estimate_position_alignment(gt_centers: dict[str, NDArray], est_centers: dict[str, NDArray], scale: bool = False) -> dict[str, NDArray]:

    gt_centers = np.asarray(list(gt_centers.values()), dtype=float)
    est_centers = np.asarray(list(est_centers.values()), dtype=float)

    centroid_gt = gt_centers.mean(axis=0)
    centroid_est = est_centers.mean(axis=0)

    centered_gt = gt_centers - centroid_gt
    centered_est = est_centers - centroid_est

    cross_cov = centered_est.T @ centered_gt

    U, E, Vt = np.linalg.svd(cross_cov)

    V = Vt.T

    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(V @ U.T))

    rotation_alignment = V @ correction @ U.T

    if scale:
        sq_norm_est = np.sum(centered_est**2)

        scale = float(np.sum(E * np.diag(correction)) / sq_norm_est)

    else:
        scale = 1.0

    translation_align = centroid_gt - scale * rotation_alignment @ centroid_est

    return {
        "scale": scale,
        "R": rotation_alignment,
        "t": translation_align
    }

def align_est_poses(R_est_wc: dict[str, NDArray], t_est_wc: dict[str, NDArray], alignment: dict[str, NDArray]) -> dict[str, NDArray]:

    aligned_poses = {}

    for cam in R_est_wc.keys():
        rot = alignment["R"] @ R_est_wc[cam]
        euler_ZYX = SciRotation.from_matrix(rot).as_euler("ZYX", True)
        t = alignment["scale"] * alignment["R"] @ t_est_wc[cam] + alignment["t"]
        aligned_poses[cam] = {
            "euler_ZYX": euler_ZYX.tolist(),
            "t": t.tolist()
        }
    
    aligned_poses["scale"] = alignment["scale"]

    return aligned_poses

def rot_error(R1, R2, degrees: bool=True):
    rel_rot = R1.T @ R2

    angle_rad = SciRotation.from_matrix(rel_rot).magnitude()

    if degrees:
        angle_deg = angle_rad / np.pi * 180.0
        return angle_deg
    
    return angle_rad

def calculate_pose_errors(gt_poses: dict[str, dict[str, list]], aligned_est_poses: dict[str, dict[str, list]]):
    per_cam = {}
    translation_errors = []
    rotation_errors = []

    R_gt_wc, t_gt_wc, R_est_wc, t_est_wc = get_rotation_and_translation(gt_poses, aligned_est_poses)

    for cam in R_gt_wc.keys():
        translation_error = float(np.linalg.norm(t_gt_wc[cam] - t_est_wc[cam]))

        rotation_error = rot_error(R_gt_wc[cam], R_est_wc[cam], degrees=True)

        translation_errors.append(translation_error)
        rotation_errors.append(rotation_error)

        per_cam[cam] = {
            "translation_error": translation_error,
            "rotation_error": rotation_error
        }

        print(f"{cam}: translation: {translation_error:.8f}, rotation: {rotation_error:.6f}")

    translation_errors = np.asarray(translation_errors, dtype=float)
    rotation_errors = np.asarray(rotation_errors, dtype=float)

    summary = {
        "N_c": len(gt_poses),
        "translation_mean": float(translation_errors.mean()),
        "translation_rmse": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_median": float(np.median(translation_errors)),
        "translation_min": float(np.min(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
        "rotation_mean": float(rotation_errors.mean()),
        "rotation_rmse": float(np.sqrt(np.mean(rotation_errors**2))),
        "rotation_median": float(np.median(rotation_errors)),
        "rotation_min": float(np.min(rotation_errors)),
        "rotation_max": float(np.max(rotation_errors))
    }

    print(f'N_c: {len(gt_poses)} \n'
            f'translation_mean: {float(translation_errors.mean())}\n'
            f'translation_rmse: {float(np.sqrt(np.mean(translation_errors**2)))}\n'
            f'translation_median: {float(np.median(translation_errors))}\n'
            f'translation_min: {float(np.min(translation_errors))}\n'
            f'translation_max: {float(np.max(translation_errors))}\n'
            f'rotation_mean: {float(rotation_errors.mean())}\n'
            f'rotation_rmse: {float(np.sqrt(np.mean(rotation_errors**2)))}\n'
            f'rotation_median: {float(np.median(rotation_errors))}\n'
            f'rotation_min: {float(np.min(rotation_errors))}\n'
            f'rotation_max: {float(np.max(rotation_errors))}\n'
    )

    return {
        "per_camera": per_cam,
        "summary": summary
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        required=True,
        type=str,
        default=None,
        help="Path to the camera pose file."
    )

    parser.add_argument(
        "--gt_path",
        required=True,
        type=str,
        default=None,
        help="Path to the reference calibration file."
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Activate if the units of the translation vector in the camera poses is not known (e.g. when calibrated using projected markers)."
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        default=False,
        help="Saves the aligned camera poses, the scale factor, and the evaluation metrics (translation and rotation errors) to a json file located in a subfolder in the camera poses parent directory."
    )

    args = parser.parse_args()

    if args.path is None or args.gt_path is None:
        print("No path given, exiting")
        exit()

    gt_path = args.gt_path
    if not gt_path.endswith(".json"):
        gt_path = os.path.join(gt_path, "camera_poses.json")
    
    est_path = args.path
    if not est_path.endswith(".json"):
        est_path = os.path.join(est_path, "camera_poses.json")
        output_directory = Path(args.path, "pose_evaluation")
    else:
        output_directory = Path(est_path, "..", "pose_evaluation")

    gt_poses, est_poses = load_json(gt_path, est_path)

    R_gt_wc, t_gt_wc, R_est_wc, t_est_wc = get_rotation_and_translation(gt_poses, est_poses)

    alignment = estimate_position_alignment(
        gt_centers=t_gt_wc,
        est_centers=t_est_wc,
        scale=args.scale
    )

    aligned_est_poses = align_est_poses(
        R_est_wc,
        t_est_wc,
        alignment
    )

    errors = calculate_pose_errors(
        gt_poses=gt_poses,
        aligned_est_poses=aligned_est_poses
    )

    if args.save_json:
        os.makedirs(output_directory, exist_ok=True)
        with open(os.path.join(output_directory, "pose_errors.json"), "w") as f:
            json.dump(errors, f, indent=4)
        with open(os.path.join(output_directory, "camera_poses_aligned.json"), "w") as f:
            json.dump(aligned_est_poses, f, indent=4)