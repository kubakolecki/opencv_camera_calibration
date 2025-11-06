import cv2
import os
import numpy as np
from chessboard import detect_chessboard_corners
import undisto
import argparse


def compute_reprojection_residuals(object_points_all_views, image_points_all_views, rvecs, tvecs, camera_matrix, distortion):
    reprojection_residuals_for_all_views = []
    for i in range(len(object_points_all_views)):
        projected_points, _ = cv2.projectPoints(object_points_all_views[i], rvecs[i], tvecs[i], camera_matrix, distortion)
        reprojection_residuals = image_points_all_views[i] - projected_points
        reprojection_residuals_for_all_views.append(reprojection_residuals)
    return reprojection_residuals_for_all_views

def compute_per_view_rmse(reprojection_residuals_for_all_views):
    per_view_rmse = []
    for residuals in reprojection_residuals_for_all_views:
        rmse = np.sqrt(np.mean(residuals**2))
        per_view_rmse.append(rmse)
    return per_view_rmse

def compute_total_rmse(reprojection_residuals_for_all_views):
    all_residuals = np.vstack(reprojection_residuals_for_all_views)
    total_rmse = np.sqrt(np.mean(all_residuals**2))
    return total_rmse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Camera Calibration using Chessboard Images")
    parser.add_argument("-i", "--image_dir", type=str, required = True, help="Directory containing chessboard images")
    parser.add_argument("-r", "--chessboard_rows", type=int, required=False, help="Number of inner corners per chessboard row. Defaults to 10.")
    parser.add_argument("-c", "--chessboard_columns", type=int, required=False, help="Number of inner corners per chessboard column. Defaults to 12.")
    parser.add_argument("-s", "--square_size", type=float, required=False, help="Size of a chessboard square in meters. Default is 0.073m.")
    parser.add_argument("-d", "--distortion_rational_model", type=bool, required=False, help="Use rational distortion model (k4, k5, k6). Default is False.")
    args = parser.parse_args()

    image_directory = args.image_dir
    chessboard_rows = args.chessboard_rows if args.chessboard_rows else 10  # Number of inner corners per a chessboard row (adjust according to your chessboard)
    chessboard_columns = args.chessboard_columns if args.chessboard_columns else 12  # Number of inner corners per a chessboard column (adjust according to your chessboard)
    chessboard_square_size = args.square_size if args.square_size else 0.073  # Size of a chessboard square (in meters, adjust according to your chessboard)
    use_rational_model = args.distortion_rational_model if args.distortion_rational_model==True else False  # Whether to use rational model (k4, k5, k6)
    flags = 0
    flags = flags | cv2.CALIB_FIX_ASPECT_RATIO
    if use_rational_model:
        flags = cv2.CALIB_RATIONAL_MODEL |  cv2.CALIB_FIX_K5 |  cv2.CALIB_FIX_K6 # | cv2.CALIB_FIX_ASPECT_RATIO


    #deriving image size from the first image in the directory and checking if all images have the same size
    print("Checking image sizes...")
    image_size = (0, 0)
    file_names = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
    valid_ext = ['.jpg', '.JPG', '.png', '.PNG']
    image_counter = 0
    for file_name in file_names:
        root_name, ext = os.path.splitext(file_name)
        if not ext in valid_ext:
            continue
        path_file_image = os.path.join(image_directory, file_name)
        image = cv2.imread(path_file_image)
        if image_counter == 0:
            image_size = (image.shape[1], image.shape[0])
        else:
            if image_size != (image.shape[1], image.shape[0]):
                raise ValueError("All images must have the same size, but image {0} has size {1}x{2} pixels instead of {3}x{4} pixels.".format(file_name, image.shape[1], image.shape[0], image_size[0], image_size[1]))
        image_counter += 1

    print(f"All images have the same size: {image_size[0]}x{image_size[1]} pixels") 


    print("Trying to detect chessboard corners. This may take a while...")
    image_points_all_views, object_points_all_views, image_file_names = detect_chessboard_corners(image_directory, chessboard_rows, chessboard_columns, chessboard_square_size)

    # Remove entries where no corners were detected
    image_points_all_views = [lst for lst in image_points_all_views if lst.shape!=()]
    object_points_all_views = [lst for lst in object_points_all_views if lst.shape!=()]
    image_file_names = [name for name in image_file_names if name!=None]

    print(np.zeros((1,8)))

    ret, camera_matrix, distortion, rvecs, tvecs, std_dev_int, std_dev_ext, _ = cv2.calibrateCameraExtended(object_points_all_views, image_points_all_views, image_size, cameraMatrix=None, distCoeffs=np.zeros((1,8)), flags=flags)
    
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion.flatten())  # Flatten to 1D array for easier reading

    reprojection_residuals_for_all_views = compute_reprojection_residuals(object_points_all_views, image_points_all_views, rvecs, tvecs, camera_matrix, distortion)
    per_view_rmse = compute_per_view_rmse(reprojection_residuals_for_all_views)

    for rmse in zip(image_file_names, per_view_rmse):
        print(f"View {rmse[0]}: RMSE = {rmse[1]:.4f} pixels")

    total_rmse = compute_total_rmse(reprojection_residuals_for_all_views)

    print(f"\nTotal RMSE across all views: {total_rmse:.4f} pixels")


    distortion_coeffs_names = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6','s1', 's2', 's3', 's4', 'tau1', 'tau2']
    intrinsics_names = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4', 'tau1', 'tau2']
    with open(os.path.join(image_directory, "calibration_report.txt"), "w") as file_report:
        file_report.write("Camera matrix:\n")
        file_report.write(np.array2string(camera_matrix, precision=6, separator=', ') + "\n\n")
        file_report.write("\nCamera intrinsic parameters:\n")
        file_report.write(f"fx: {camera_matrix[0,0]:.6f}\n")
        file_report.write(f"fy: {camera_matrix[1,1]:.6f}\n")
        file_report.write(f"cx: {camera_matrix[0,2]:.6f}\n")
        file_report.write(f"cy: {camera_matrix[1,2]:.6f}\n\n")
        file_report.write("Distortion coefficients:\n")
        for i in range(distortion.shape[1]):
            file_report.write(f"{distortion_coeffs_names[i]}: {distortion[0, i]:.12f}\n")

        file_report.write("\nPer-view RMSE:\n")
        for rmse in zip(image_file_names, per_view_rmse):
            file_report.write(f"View {rmse[0]}: RMSE = {rmse[1]:.4f} pixels\n")

        file_report.write(f"\nTotal RMSE across all views: {total_rmse:.4f} pixels\n")

        file_report.write("\nStandard deviations of intrinsic parameters:\n")
        for i in range(std_dev_int.shape[0]):
            file_report.write(f"{intrinsics_names[i]}: {std_dev_int.flatten()[i]:.12f}\n")


    fs = cv2.FileStorage(os.path.join(image_directory,"calibration.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", distortion)
    fs.release()

 
    print("running undistortion...\n")


    camera_matrix_new = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 0.00, image_size, centerPrincipalPoint = True)
    print("New camera matrix:")
    print(camera_matrix_new)

    rectification_map = undisto.generate_undistortion_map(camera_matrix, distortion, camera_matrix_new[0], image_size)
    print(rectification_map[0].shape)
    print(rectification_map[1].shape)

    undisto.undistort_images(image_directory, rectification_map[0], rectification_map[1])

    fs_new = cv2.FileStorage(os.path.join(image_directory,"calibration_undistorted.yaml"), cv2.FILE_STORAGE_WRITE)
    fs_new.write("image_width", image_size[0])
    fs_new.write("image_height", image_size[1])
    fs_new.write("camera_matrix", camera_matrix_new[0])
    fs_new.write("dist_coeffs", np.zeros((1,5)))  # Normally no distortion after undistortion
    fs_new.release()
    