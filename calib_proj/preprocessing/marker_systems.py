import os
# os.add_dll_directory("C:/Users/timfl/miniconda3/envs/master_thesis/Lib/site-packages/pupil_apriltags/lib")
# os.add_dll_directory("C:/Program Files/MATLAB/R2023b/bin/win64")
import cv2
import numpy as np

from pathlib import Path



ARUCO_DICTIONARIES = {'4X4_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                        '4X4_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
                        '4X4_1000': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
                        '5X5_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
                        '6X6_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
                        '7X7_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)}

def detect_aruco_markers(image, 
                         dictionary: str, 
                         parameters: cv2.aruco_DetectorParameters = None, 
                         draw: bool = True, 
                         show_draw_img: bool = False,
                         print_centers: bool = False, 
                         print_corners: bool = False, 
                         add_half_pixel_shift = None, 
                         refinement_method = cv2.aruco.CORNER_REFINE_SUBPIX):

    if add_half_pixel_shift is None:
        raise ValueError("add_half_pixel_shift must be set to True or False")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    if parameters is None:
        parameters =  cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = refinement_method
        # parameters.cornerRefinementMaxIterations = 100000
        # parameters.cornerRefinementMinAccuracy = 1e-6
        # parameters.cornerRefinementWinSize = 
        # parameters.markerBorderBits = 1
        # parameters.detectInvertedMarker = 0

        # parameters.adaptiveThreshWinSizeMin = 3
        # parameters.adaptiveThreshWinSizeMax = 23
        # parameters.adaptiveThreshWinSizeStep = 1
        # parameters.adaptiveThreshConstant = 0
        # parameters.minMarkerPerimeterRate = 0.01
        # parameters.maxMarkerPerimeterRate = 4.0

    detector = cv2.aruco.ArucoDetector(ARUCO_DICTIONARIES[dictionary], parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    if draw:
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        if show_draw_img:   
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    markers_dict = {}
    if ids is not None:
        for i in range(len(ids)):
            if add_half_pixel_shift:
                markers_dict[ids[i][0]] = corners[i][0] + 0.5
            else: 
                markers_dict[ids[i][0]] = corners[i][0]
            
        
    markers_dict = dict(sorted(markers_dict.items(), key=lambda x: x[0]))

    if print_centers:
        for marker_id in markers_dict: 
            center = np.mean(markers_dict[marker_id], axis=0)
            print(f"id: {marker_id:2}, center: ({center[0]:7.2f}, {center[1]:7.2f})")
    
    if print_corners:
        for marker_id in markers_dict: 
            print(f"corners of marker {marker_id}")
            print(markers_dict[marker_id])

    return markers_dict, image


def detect_apriltag_markers_detector(image, 
                            at_detector,
                            draw: bool = True, 
                            show_draw_img: bool = False,
                            print_centers: bool = False, 
                            print_corners: bool = False):
    
    # from pupil_apriltags import Detector

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # if parameters is None:
    #     at_detector = Detector(
    #         families=dict, # tag16h5, tag25h9,tag36h11
    #         nthreads=1,
    #         quad_decimate=1.0,
    #         quad_sigma=0.0,
    #         refine_edges=1,
    #         decode_sharpening=0.25,
    #         debug=0
    #     )
    # else: 
    #     raise ValueError("parameters gestion not implemented for now (tim) pupil_apriltags")

    
    tags = at_detector.detect(gray)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    corners = {}

  
    for tag in tags:
        
        detected_corners = np.zeros((4,2))
        detected_corners[0,:] = tag.corners[3,:]
        detected_corners[1,:] = tag.corners[2,:]
        detected_corners[2,:] = tag.corners[1,:]
        detected_corners[3,:] = tag.corners[0,:]

        corners[tag.tag_id] = detected_corners

        if draw:
            for i, corner in enumerate(detected_corners):
                cv2.putText(
                    img,
                    str(i),
                    org=(corner[0].astype(int) + 10, corner[1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 0, 0),
                )
            for idx in range(len(tag.corners)):
                cv2.line(
                    img,
                    tuple(tag.corners[idx - 1, :].astype(int)),
                    tuple(tag.corners[idx, :].astype(int)),
                    (0, 255, 0),
                )

            cv2.putText(
                img,
                str(tag.tag_id),
                org=(
                    tag.corners[0, 0].astype(int) + 10,
                    tag.corners[0, 1].astype(int) + 10,
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
            )
    if show_draw_img:   
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    if print_centers:
        for marker_id in corners: 
            center = np.mean(corners[marker_id], axis=0)
            print(f"id: {marker_id:2}, center: ({center[0]:7.2f}, {center[1]:7.2f})")
    if print_corners:
        for marker_id in corners: 
            print(f"corners of marker {marker_id}")
            print(corners[marker_id])

    return corners, img