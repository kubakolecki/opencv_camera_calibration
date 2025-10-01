import cv2
import os
import glob
import numpy as np
from chessboard import detect_chessboard_corners

def create_transformation_matrix(R, T):
    transformation_matrix = np.eye(4,4)
    transformation_matrix[0:3, 0:3] = R
    transformation_matrix[0:3, 3] = T.flatten()
    return transformation_matrix

def write_transformation_matrix_to_file(file_path, transformation_matrix):
    if transformation_matrix.shape != (4,4):
        raise ValueError("Transformation matrix must be of shape (4,4)")
    with open(file_path, 'w') as f:
        for i in range(4):
            for j in range(4):
                f.write(f"{transformation_matrix[i,j]:0,.14f}")
                if j < 3:
                    f.write(",")
            f.write("\n")

#path_directory_images_left = "/datadisk/data/agh_projects/miss/camera_calibration/20250831_stereo_calibration/selected_pairs_renamed/004LZ"
#path_directory_images_right = "/datadisk/data/agh_projects/miss/camera_calibration/20250831_stereo_calibration/selected_pairs_renamed/004M1"

path_directory_images_left = "/datadisk/data/agh_projects/miss/camera_calibration/20250916_stereo_calibration_2_7mm_lens/004LZ"
path_directory_images_right = "/datadisk/data/agh_projects/miss/camera_calibration/20250916_stereo_calibration_2_7mm_lens/004M1"

path_file_camera_calibration_left = "/datadisk/data/agh_projects/miss/camera_calibration/20250916_calibration_2_7mm_lens/004LZ/calibration.yaml"
path_file_camera_calibration_right = "/datadisk/data/agh_projects/miss/camera_calibration/20250916_calibration_2_7mm_lens/004M1/calibration.yaml"

path_file_output_stereo_calibration_c2_c1 = "/datadisk/data/agh_projects/miss/camera_calibration/20250916_stereo_calibration_2_7mm_lens/stereo_calibration_c2_c1.txt"
path_file_output_stereo_calibration_c1_c2 = "/datadisk/data/agh_projects/miss/camera_calibration/20250916_stereo_calibration_2_7mm_lens/stereo_calibration_c1_c2.txt"

chessboard_field_size = 0.073


fs = cv2.FileStorage(path_file_camera_calibration_left, cv2.FILE_STORAGE_READ)
camera_matrix_left= fs.getNode("camera_matrix").mat()
dist_coeffs_left = fs.getNode("dist_coeffs").mat()
fs.release()

print(camera_matrix_left)
print(dist_coeffs_left)

fs = cv2.FileStorage(path_file_camera_calibration_right, cv2.FILE_STORAGE_READ)
camera_matrix_right= fs.getNode("camera_matrix").mat()
dist_coeffs_right = fs.getNode("dist_coeffs").mat()
fs.release()

print(camera_matrix_right)
print(dist_coeffs_right)

image_points_left, object_points_left, image_file_names_left = detect_chessboard_corners(path_directory_images_left, 10, 12, chessboard_field_size)
image_points_right, object_points_right, image_file_names_right = detect_chessboard_corners(path_directory_images_right, 10, 12, chessboard_field_size)

if len(image_points_left) != len(image_points_right):
    print("The number of corner detection results does not match between left and right images! Probaly the number of images in left and right directories is different. Check the directories!")
    print(f"Left: {len(image_points_left)}, Right: {len(image_points_right)}")
    exit(1)

number_of_pairs = len(image_points_left)

image_points_left_valid = []
image_points_right_valid = []
object_points_left_valid = []
object_points_right_valid = []
image_file_names_left_valid = []
image_file_names_right_valid = []


for i in range(number_of_pairs):
    if image_points_left[i].shape==() or image_points_right[i].shape==():
        print(f"Skipping pair {i}  - no corners detected in one of the images!")
        continue
    image_points_left_valid.append(image_points_left[i])
    image_points_right_valid.append(image_points_right[i])
    object_points_left_valid.append(object_points_left[i])
    object_points_right_valid.append(object_points_right[i])
    image_file_names_left_valid.append(image_file_names_left[i])
    image_file_names_right_valid.append(image_file_names_right[i])


flags = cv2.CALIB_FIX_INTRINSIC
_, _, _, _, _, R, T, E, F, rvects, tvects, per_view_errors = cv2.stereoCalibrateExtended(objectPoints=object_points_left_valid,
                    imagePoints1=image_points_left_valid,
                    imagePoints2=image_points_right_valid,
                    cameraMatrix1=camera_matrix_left,
                    distCoeffs1=dist_coeffs_left,
                    cameraMatrix2=camera_matrix_right,
                    distCoeffs2=dist_coeffs_right,
                    imageSize=(1280, 960),
                    R=None,
                    T=None,
                    flags=flags) 

print("Rotation matrix between the coordinate systems of the first and the second camera:")
print(R)
print("\nTranslation vector between the coordinate systems of the first and the second camera:")
print(T)

print("\nPer view reprojection errors:")
for i in range(len(per_view_errors)):
    print(f"View {i} (Left: {image_file_names_left_valid[i]}, Right: {image_file_names_right_valid[i]}): {per_view_errors.flatten()[i]:.4f} pixels")

transformation_matrix = create_transformation_matrix(R, T)
write_transformation_matrix_to_file(path_file_output_stereo_calibration_c2_c1, transformation_matrix)
write_transformation_matrix_to_file(path_file_output_stereo_calibration_c1_c2, np.linalg.inv(transformation_matrix))
