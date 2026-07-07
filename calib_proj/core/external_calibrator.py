from typing import Dict, List, Tuple
import numpy as np
import itertools
import cv2
import copy
import json

from calib_commons.types import idtype
from calib_commons.world_frame import WorldFrame 
import calib_commons.utils.se3 as se3
from calib_commons.utils.se3 import SE3
from calib_commons.correspondences import get_conform_obs_of_cam, filter_correspondences_with_track_length, get_tracks
from calib_commons.intrinsics import Intrinsics
from calib_commons.scene import SceneType
from calib_commons.camera import Camera
from calib_commons.objectPoint import ObjectPoint

from calib_proj.core.projector_scene import ProjectorScene
from calib_proj.core.estimate import Estimate
from calib_proj.utils.plane import Plane
from calib_proj.core.config import ExternalCalibratorConfig, SolvingLevel
import calib_proj.core.ba as ba
from calib_proj.utils.homography import retrieve_motion_using_homography



class ExternalCalibrator: 

    def __init__(self, 
                 correspondences, 
                 intrinsics: Dict[idtype, Intrinsics], 
                 config: ExternalCalibratorConfig,
                 save_corr: bool = False,
                 out_path = None
                ):
        
        self.out_path = out_path
        self.config = config
        self.intrinsics = intrinsics
        self.correspondences = copy.deepcopy(correspondences)
        if save_corr: 
            self.tmp_save_correspondences()
        self.estimate = Estimate(SOLVING_LEVEL=self.config.SOLVING_LEVEL)  # current estimate
    
    def tmp_save_correspondences(self):
        new_corr = {}
        for cam in self.correspondences.keys():
            new_corr[cam] = {}
            for ids, obs in self.correspondences[cam].items():
                new_corr[cam][ids] = obs._2d.tolist()

        with open(self.out_path / "correspondences.json", "w") as f:
            json.dump(new_corr, f)

    def calibrate(self): 
        # bootstrapping
        success = False
        initial_camera_pair_ids = self.select_initial_camera_pair()
        success = self.bootstrapping_from_initial_pair(cam_id_1=initial_camera_pair_ids[0], 
                                                        cam_id_2=initial_camera_pair_ids[1])
        if not success: 
            print("Bootstrapping failed.")
            return False
        self.iterative_filtering()
        
        # loop over remaining cameras: 
            # 1. add camera using observation with estimated 3d points
            # 2. triangulate new points
            # 3. iterative filtering (BA + filtering, repeated)
        while len(self.get_remaining_camera_ids()): 
            cam_id = self.select_next_best_cam()
            self.add_camera(cam_id)
            self.add_points_with_new_camera(cam_id)
            self.iterative_filtering()


        print("\n ##################### CALIBRATION TERMINATED #####################")

        scores = self.get_cameras_scores()
        print("scores: ", scores)
        if min(scores.values()) > self.config.camera_score_threshold: 
            print ("Calibration Status: each camera has sufficient distribution score.")
        else: 
            print (f"Calibration Status: some cameras do not have a sufficient distribution score.")
        return True

    def select_initial_camera_pair(self) -> Tuple[idtype, idtype]:
        cameras_id = self.correspondences.keys()
        arrangements = list(itertools.combinations(cameras_id, 2))
        best_score = -np.inf
        best_cam_ids = None
        for arrangement in arrangements:
            cam0_id, cam1_id = arrangement            
            common_points_id = set(self.get_conform_obs_in_cam(cam0_id).keys()) & set(self.get_conform_obs_in_cam(cam1_id).keys())
            image_points1 = [self.correspondences[cam1_id][pt_id]._2d for pt_id in common_points_id]
            image_points2 = [self.correspondences[cam0_id][pt_id]._2d for pt_id in common_points_id]
            score1 = self.view_score(np.vstack(image_points1), self.intrinsics[cam1_id].resolution) if image_points1 else 0
            score2 = self.view_score(np.vstack(image_points2), self.intrinsics[cam0_id].resolution) if image_points2 else 0
            score = score1 + score2
            if score > best_score:
                best_score = score
                best_cam_ids = (cam0_id, cam1_id)
        return best_cam_ids
    
    def select_next_best_cam(self) -> int:
        candidates = self.get_remaining_camera_ids() 
        best_score = -np.inf
        best_cam_id = None
        for cam_id in candidates:
            score = self.get_camera_score(cam_id)
            if score > best_score:
                best_score = score
                best_cam_id = cam_id
        return best_cam_id
    
    def get_camera_score(self, 
                         cam_id: idtype) -> float:
        common_points_id = set(self.get_conform_obs_in_cam(cam_id).keys()) & self.estimate.scene.generic_scene.get_point_ids()
        image_points = [self.correspondences[cam_id][pt_id]._2d for pt_id in common_points_id]
        score = self.view_score(np.vstack(image_points), self.intrinsics[cam_id].resolution) if image_points else 0
        return score
    
    def get_cameras_scores(self): 
        return {cam_id: self.get_camera_score(cam_id) for cam_id in self.estimate.scene.generic_scene.cameras.keys()}

    def view_score(self, 
                   image_points: np.ndarray, 
                   image_resolution: Tuple):
        s, L = 0, 3
        width, height = image_resolution
        for l in range(1, L+1):
            K_l = 2**l
            w_l = K_l  # Assuming w_l = K_l is correct
            grid = np.zeros((K_l, K_l), dtype=bool)
            for point in image_points:
                u,v = point
                u = np.clip(u, 0, width-1)
                v = np.clip(v, 0, height-1)
                x = int(np.ceil(K_l * u / width)) - 1
                y = int(np.ceil(K_l * v / height)) - 1
                if grid[x, y] == False:
                    grid[x, y] = True  # Mark the cell as full
                    s += w_l  # Increase the score
        return s
    
   
   
    def bootstrapping_from_initial_pair(self, 
                                        cam_id_1: idtype, 
                                        cam_id_2: idtype) -> bool: 
        print(f"*********** Bootstrapping from initial camera pair (cam1, cam2) = ({cam_id_1}, {cam_id_2}) ***********")
        pts_per_cam = {}
        pts_per_cam[cam_id_1] = {point_id: self.correspondences[cam_id_1][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_1).keys()}
        pts_per_cam[cam_id_2] = {point_id: self.correspondences[cam_id_2][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_2).keys()}
        sols_intial_pair = retrieve_motion_using_homography(self.intrinsics[cam_id_1].K, self.intrinsics[cam_id_2].K, pts_per_cam[cam_id_1], pts_per_cam[cam_id_2])
        if not sols_intial_pair:
            return False

        if len(sols_intial_pair) == 1:
            cam1_final = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
            cam2_final = Camera(cam_id_2, sols_intial_pair[0][0], self.intrinsics[cam_id_2])
            n_1_final = sols_intial_pair[0][1]
        
        else: # two solutions
            # select third camera 
            in_pair_with_cam, third_cam_id = self.select_third_camera_for_bootsrapping(cam_id_1, cam_id_2)
            if third_cam_id is None:
                print("There is no third camera to remove the ambiguity.")
                return False
            if self.config.verbose >= 2:
                print(f"Ambiguity removal with third camera {third_cam_id}.")
            world_id = in_pair_with_cam
          
            pts_per_cam[third_cam_id] = {point_id: self.correspondences[third_cam_id][point_id]._2d for point_id in self.get_conform_obs_in_cam(third_cam_id).keys()}
            sols_second_pair = retrieve_motion_using_homography(self.intrinsics[world_id].K, self.intrinsics[third_cam_id].K, pts_per_cam[world_id], pts_per_cam[third_cam_id], verbose = self.config.verbose)

            # for idx_second_pair, (T_W_3_, n_W_from_2nd_pair) in enumerate(sols_second_pair): 
            #     print(f"n_world_from_2nd_pair: {n_W_from_2nd_pair}")

            errors_on_plane_normal = {}
            for idx_first_pair, (T_1_2_, n_1_) in enumerate(sols_intial_pair):
                if world_id == cam_id_1: 
                    n_W_ = n_1_
                else: 
                    T_W_1_ = T_1_2_.inv()
                    R_W_1_ = T_W_1_.get_R()
                    n_W_ = R_W_1_ @ n_1_
                for idx_second_pair, (T_W_3_, n_W_from_2nd_pair) in enumerate(sols_second_pair): 
                    errors_on_plane_normal[(idx_first_pair, idx_second_pair)] = np.linalg.norm(n_W_ - n_W_from_2nd_pair)
            
            # print(f"Errors on plane normal:")
            # print(errors_on_plane_normal)
            min_key = min(errors_on_plane_normal, key=errors_on_plane_normal.get)
            best_sol_idx = min_key[0]
            cam1_final = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
            cam2_final = Camera(cam_id_2, sols_intial_pair[best_sol_idx][0], self.intrinsics[cam_id_2])
            n_1_final = sols_intial_pair[best_sol_idx][1]
        
        plane_final = Plane.from_normal_and_d(n_1_final, d=-1, id=0)
        self.estimate.scene.generic_scene.add_camera(cam1_final)
        self.estimate.scene.generic_scene.add_camera(cam2_final)
        if self.config.verbose >= 1:
            print("\n *********** Cam " + str(cam_id_1) + " added ***********")
        if self.config.verbose >= 1:
            print("\n *********** Cam " + str(cam_id_2) + " added ***********")
        self.correspondences = filter_correspondences_with_track_length(copy.deepcopy(self.correspondences), min_track_length=2)

        # structure parameters estimation depending on solving level
        # free          -> N 3d object points
        # planarity     -> plane + N 2d coordinates of points in the plane

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            object_points = self.triangulate_new_points(cam_id_1=cam_id_1, 
                                                    cam_id_2=cam_id_2)
            self.estimate.scene.generic_scene.object_points = object_points

        elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
            self.estimate.scene.plane = plane_final
            # here we can use all points (including the ones seen in only one of the two camera of the pair)

            # 1: points in common
            points = None
            ids = []
            object_points_in_world = self.triangulate_new_points(cam_id_1=cam_id_1, 
                                                            cam_id_2=cam_id_2)
            if object_points_in_world:
                points_in_common_ids = object_points_in_world.keys()
                ids += points_in_common_ids
                points = np.array([pt.position for pt in object_points_in_world.values()]).squeeze()
            else: 
                points_in_common_ids = []
            
            # 2: points seen in only one of the two cameras
            # in cam 1
            points_only_cam_1_ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_1_ids:
                img_pts = np.array([self.correspondences[cam_id_1][point_id]._2d for point_id in points_only_cam_1_ids], dtype='float32').squeeze()
                points_cam_1 = self.estimate.scene.generic_scene.cameras[cam_id_1].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                points = np.vstack((points, points_cam_1)) if points is not None else points_cam_1
                ids += points_only_cam_1_ids
                
            # in cam 2
            points_only_cam_2_ids = list(set(self.get_conform_obs_in_cam(cam_id_2).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_2_ids:
                img_pts = np.array([self.correspondences[cam_id_2][point_id]._2d for point_id in points_only_cam_2_ids], dtype='float32').squeeze()
                points_cam_2 = self.estimate.scene.generic_scene.cameras[cam_id_2].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                points = np.vstack((points, points_cam_2)) if points is not None else points_cam_2
                ids += points_only_cam_2_ids
            
            points_plane = se3.change_coords(se3.inv_T(self.estimate.scene.plane.pose.mat), points)
            self.estimate._2d_plane_points = {id: points_plane[i,:2] for i, id in enumerate(ids)}
            self.estimate.update_points()
    
        return True 
    
    def select_third_camera_for_bootsrapping(self, cam_id_1, cam_id_2):
        cameras_id =  list(set(self.correspondences.keys()) - {cam_id_1, cam_id_2})
        arrangements_1 = [(cam_id_1, cam_id) for cam_id in cameras_id]
        arrangements_2 = [(cam_id_2, cam_id) for cam_id in cameras_id]
        arrangements = arrangements_1 + arrangements_2
        best_score = -np.inf
        best_cam_ids = None

        for arrangement in arrangements:
            cam0_id, cam1_id = arrangement            
            common_points_id = set(self.get_conform_obs_in_cam(cam0_id).keys()) & set(self.get_conform_obs_in_cam(cam1_id).keys())
            image_points1 = [self.correspondences[cam1_id][pt_id]._2d for pt_id in common_points_id]
            image_points2 = [self.correspondences[cam0_id][pt_id]._2d for pt_id in common_points_id]
            score1 = self.view_score(np.vstack(image_points1), self.intrinsics[cam1_id].resolution) if image_points1 else 0
            score2 = self.view_score(np.vstack(image_points2), self.intrinsics[cam0_id].resolution) if image_points2 else 0
            score = score1 + score2
            if score > best_score:
                best_score = score
                best_cam_ids = (cam0_id, cam1_id)

        third_cam_id = best_cam_ids[1]
        return best_cam_ids[0], third_cam_id

   
    def triangulate_new_points(self, 
                                cam_id_1: idtype, 
                                cam_id_2: idtype) -> Dict[idtype, ObjectPoint]: 
        common_points_ids_ = set(self.get_conform_obs_in_cam(cam_id_1).keys()) & \
                            set(self.get_conform_obs_in_cam(cam_id_2).keys())
        new_points_ids = list(common_points_ids_ - self.estimate.scene.generic_scene.get_point_ids())
        if new_points_ids:
            pts_src = np.array([self.correspondences[cam_id_1][point_id]._2d for point_id in new_points_ids], dtype='float32').squeeze()
            pts_dst = np.array([self.correspondences[cam_id_2][point_id]._2d for point_id in new_points_ids], dtype='float32').squeeze()
            points = cv2.triangulatePoints(self.estimate.scene.generic_scene.cameras[cam_id_1].get_projection_matrix(), 
                                        self.estimate.scene.generic_scene.cameras[cam_id_2].get_projection_matrix(), 
                                        pts_src.T, 
                                        pts_dst.T)
            points = points / points[3,:]
            points = points[:3, :]
            return {id: ObjectPoint(id, points[:,i]) for i, id in enumerate(new_points_ids)}

         
    def iterative_filtering(self):
        if self.config.verbose >= 2:
            print("\n ----- Iterative Filtering -----")
        iter = 1
        continue_ = True
        while continue_:
            if self.config.verbose >= 2:
                print(f"\n ----- Iteration: {iter} -----" )
            iter += 1
            self.BA()
            continue_ = self.filtering()
    
    def filtering(self) -> bool: 
        if self.config.verbose >= 2:
            print("\n ** Filtering: **")
        num_point_filtered = 0
        points_removed = []
        point_ids = sorted(list(self.estimate.scene.generic_scene.object_points.keys()))
        for camera in self.estimate.scene.generic_scene.cameras.values(): 
            for point_id in point_ids: 
                point = self.estimate.scene.generic_scene.object_points.get(point_id)
                if point:
                    observation = self.correspondences[camera.id].get(point.id)
                    if observation and observation._is_conform:
                        _2d_reprojection = camera.reproject(point.position)
                        errors_xy = observation._2d - _2d_reprojection
                        error = np.sqrt(np.sum(errors_xy**2))
                        is_obsv_conform = error < self.config.reprojection_error_threshold
                        #     print(f"observation of point {point.id:>3} in cam {camera.id} error: {error:.2f} [pix]")
                        if not is_obsv_conform: 
                            num_point_filtered += 1
                            self.correspondences[camera.id][point.id]._is_conform = False 
                            if self.config.SOLVING_LEVEL == SolvingLevel.FREE:
                                min_track_length = 2
                            elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
                                min_track_length = 1
                            if len(self.get_tracks()[point.id]) < min_track_length:
                                points_removed.append(point.id)
                                del self.estimate.scene.generic_scene.object_points[point.id]
                                if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
                                    del self.estimate._2d_plane_points[point.id]
                           
        if self.config.verbose >= 2:
            print(f" -> Number of observations filtered: {num_point_filtered}")
            print(f" -> Number of points removed from estimate: {len(points_removed)}")
        return num_point_filtered > 0

    def add_camera(self, cam_id: idtype): 
        K = self.intrinsics[cam_id].K
        common_points_id = set(self.get_conform_obs_in_cam(cam_id).keys()) & self.estimate.scene.generic_scene.get_point_ids()

        # PnP
        _2d = [self.correspondences[cam_id][point_id]._2d for point_id in list(common_points_id)]
        _3d = [self.estimate.scene.generic_scene.object_points[point_id].position for point_id in list(common_points_id)]
        _2d = np.vstack(_2d)
        _3d = np.vstack(_3d)

        T_W_C = self.pnp(_2d, _3d, K)
        self.estimate.scene.generic_scene.add_camera(Camera(cam_id, SE3(T_W_C), self.intrinsics[cam_id]))
        if self.config.verbose >= 1:
            print("\n *********** Cam " + str(cam_id) + " added ***********")
        if self.config.verbose >= 2:
            print(f"estimated using PnP over {len(common_points_id)} points")

        return
    
    def add_points_with_new_camera(self, new_cam_id: idtype) -> None: 
        new_points_ids = []
        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            for camera_id in self.estimate.scene.generic_scene.cameras: 
                if camera_id == new_cam_id:
                    continue
                object_points = self.triangulate_new_points(cam_id_1=camera_id, cam_id_2=new_cam_id)
                if object_points:
                    self.estimate.scene.generic_scene.object_points.update(object_points)
                    new_points_ids += list(object_points.keys())
                    if self.config.verbose >= 2:
                        print(f"new points triangulated using cameras ({camera_id}, {new_cam_id}): {len(list(object_points.keys()))}")
                else: 
                    if self.config.verbose >= 2:
                        print(f"new points triangulated using cameras ({camera_id}, {new_cam_id}): none")

        if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            new_points_ids = list(set(self.get_conform_obs_in_cam(new_cam_id).keys()) - self.estimate.scene.generic_scene.get_point_ids())
            if new_points_ids:
                img_pts = np.array([self.correspondences[new_cam_id][point_id]._2d for point_id in new_points_ids], dtype='float32').squeeze()
                points = self.estimate.scene.generic_scene.cameras[new_cam_id].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                points_plane = se3.change_coords(se3.inv_T(self.estimate.scene.plane.pose.mat), points)
                new_2d_plane_points = {id: points_plane[i,:2] for i, id in enumerate(new_points_ids)}
                self.estimate._2d_plane_points.update(new_2d_plane_points)
                self.estimate.update_points()

        if self.config.verbose >= 2:
            if len(new_points_ids):
                print("New points: " + str(len(sorted(new_points_ids))))
            else: 
                print("New points: None")   
    
    
    def pnp(self, 
            _2d: np.ndarray, 
            _3d: np.ndarray, 
            K: np.ndarray) -> np.ndarray: 
        """
       Solve PnP using RANSAC.

        Args : 
            _2d: np.ndarray, dim: (N, 3)
                image-points
            _3d: np.ndarray, dim: (M, 3)
                 object-points
            K: np.ndarray, dim: (3,3)
                intrinsics matrix
        Returns : 
            T_W_C: np.ndarray, dim: (4, 4)
                pose of the camera in the world frame {w} (the frame of the object-points)
        """

        _, rvec, tvec, _ = cv2.solvePnPRansac(imagePoints=_2d, 
                                                    objectPoints=_3d, 
                                                    cameraMatrix=K, 
                                                    distCoeffs=None, 
                                                    useExtrinsicGuess=False)
        
        T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
        T_W_C = se3.inv_T(T_C_W)
        return T_W_C
     
    def BA(self):
        if self.config.verbose >= 2:
            print("** BA: **")

        intrinsics = [{'fx': cam.intrinsics.K[0, 0], 'fy': cam.intrinsics.K[1, 1], 'cx': cam.intrinsics.K[0, 2], 'cy': cam.intrinsics.K[1, 2]} 
                      for cam in self.estimate.scene.generic_scene.cameras.values()]
        camera_pose_array = np.zeros((len(self.estimate.scene.generic_scene.cameras) , 6))
        for i, camera in enumerate(self.estimate.scene.generic_scene.cameras.values()):
            camera_pose_array[i, :3], camera_pose_array[i, 3:] = se3.rvec_tvec_from_T(np.linalg.inv(camera.pose.mat))
        points_2d = []
        camera_indices = []
        point_indices = []
        for cam_idx, (cam_id, camera) in enumerate(self.estimate.scene.generic_scene.cameras.items()):
            for pt_idx, (pt_id, point) in enumerate(self.estimate.scene.generic_scene.object_points.items()):
                observation = self.correspondences[cam_id].get(pt_id)
                if observation and observation._is_conform:
                    points_2d.append(observation._2d)
                    camera_indices.append(cam_idx)
                    point_indices.append(pt_idx)
        points_2d = np.array(points_2d, dtype='float32').squeeze()
        camera_indices = np.array(camera_indices, dtype='int')
        point_indices = np.array(point_indices, dtype='int')
    
        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            self.BA_free(camera_pose_array, intrinsics, points_2d, camera_indices, point_indices)
        elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            self.BA_planarity(camera_pose_array, intrinsics, points_2d, camera_indices, point_indices)
        return

    def BA_free(self, camera_pose_array, intrinsics, points_2d, camera_indices, point_indices):
        points_3d = np.array([point.position for point in self.estimate.scene.generic_scene.object_points.values()])
        sba = ba.BA_Free(camera_pose_array, points_3d, points_2d, camera_indices, point_indices, intrinsics, self.config.ba_least_square_ftol, self.config.least_squares_verbose)
        optimized_camera_params, optimized_points_3d = sba.bundleAdjust()
        # Update the scene with the optimized parameters
        for i, (cam_id, camera) in enumerate(self.estimate.scene.generic_scene.cameras.items()):
            rvec, tvec = optimized_camera_params[i, :3], optimized_camera_params[i, 3:]
            T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
            T_W_C = se3.inv_T(T_C_W)
            camera.pose = SE3(T_W_C)
        for i, (pt_id, point) in enumerate(self.estimate.scene.generic_scene.object_points.items()):
            point.position = optimized_points_3d[i]
        return

    def BA_planarity(self, camera_pose_array, intrinsics, points_2d, camera_indices, point_indices):
        plane_pose = np.hstack(se3.rvec_tvec_from_T(self.estimate.scene.plane.pose.mat))
        points_2d_in_plane = np.array(list(self.estimate._2d_plane_points.values()))
        sba = ba.BA_PlaneConstraint(camera_pose_array, plane_pose, points_2d_in_plane, points_2d, camera_indices, point_indices, intrinsics, self.config.ba_least_square_ftol,  self.config.least_squares_verbose)
        optimized_camera_params, optimized_plane_pose, optimized_points_2d_plane = sba.bundleAdjust()

        # Update the scene with the optimized parameters
        for i, (cam_id, camera) in enumerate(self.estimate.scene.generic_scene.cameras.items()):
            rvec = optimized_camera_params[i, :3]
            tvec = optimized_camera_params[i, 3:]
            T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
            T_W_C = se3.inv_T(T_C_W)
            camera.pose = SE3(T_W_C)

        rvec, tvec = optimized_plane_pose[:3], optimized_plane_pose[3:]
        T_W_plane = se3.T_from_rvec_tvec(rvec, tvec)
        self.estimate.scene.plane = Plane(SE3(T_W_plane))
        for i, (pt_id, point) in enumerate(self.estimate._2d_plane_points.items()):
            self.estimate._2d_plane_points[pt_id] = optimized_points_2d_plane[i]
        self.estimate.update_points()
    
        return
    
  
    def get_scene(self, world_frame) -> ProjectorScene:
        if world_frame == WorldFrame.CAM_FIRST_CHOOSEN: 
            T = np.eye(4)
        elif self.estimate.scene.generic_scene.cameras.get(world_frame): 
            T = se3.inv_T(self.estimate.scene.generic_scene.cameras[world_frame].pose.mat)
        # elif world_frame == WorldFrame.CAM_ID_1: 
        #     T = se3.inv_T(self.estimate.scene.generic_scene.cameras[1].pose.mat)
        else: 
            raise ValueError("World Frame not valid.")
        
        cameras = {id: Camera(id, SE3(T @ camera.pose.mat), camera.intrinsics) for id, camera in self.estimate.scene.generic_scene.cameras.items()}
        object_points = {id: ObjectPoint(id, (T @ np.append(p.position, 1))[:3]) for id, p in self.estimate.scene.generic_scene.object_points.items()}
        if self.estimate.scene.plane: 
            plane = Plane(pose=SE3(T@self.estimate.scene.plane.pose.mat))
        else: 
            plane = None
        return ProjectorScene(cameras=cameras, 
                      object_points=object_points, 
                      plane = plane,
                      scene_type=SceneType.ESTIMATE)
    
    def get_conform_obs_in_cam(self, 
                               cam_id: idtype): 
        return get_conform_obs_of_cam(cam_id=cam_id, correspondences=self.correspondences)

    def get_common_obs_ids_in_cameras(self, 
                                  cam_1_id: idtype, 
                                  cam_2_id: idtype): 
        return set(self.get_conform_obs_in_cam(cam_1_id).keys()) & set(self.get_conform_obs_in_cam(cam_2_id).keys())
    
    def get_camera_ids(self) -> set[idtype]: 
        return set(self.correspondences.keys())

    def get_num_cameras(self) -> int: 
        return len(self.correspondences)
    
    def get_tracks(self) -> Dict[idtype, set[idtype]]: 
       return get_tracks(self.correspondences)
    
    def get_remaining_camera_ids(self) -> set[idtype]: 
        return self.get_camera_ids() - self.estimate.scene.generic_scene.get_camera_ids()
    
