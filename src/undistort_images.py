import numpy as np
import cv2
import os
import undisto

path_directory_images_input = "/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/004M1/"
path_file_camera_calibration = "/datadisk/data/agh_projects/miss/camera_calibration/20251017_calibration_2_7mm_lens/004M1/calibration.yaml"

fs = cv2.FileStorage(path_file_camera_calibration, cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat()
distortion = fs.getNode("dist_coeffs").mat()
image_width = fs.getNode("image_width")
image_height = fs.getNode("image_height")
image_size = (int(image_width.real()), int(image_height.real()))
fs.release()

camera_matrix_new = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 0.25, image_size, centerPrincipalPoint = True)
rectification_map = undisto.generate_undistortion_map(camera_matrix, distortion, camera_matrix_new[0], image_size)

undisto.undistort_images(path_directory_images_input, rectification_map[0], rectification_map[1])