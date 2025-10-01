import cv2
import os
import numpy as np
from chessboard import detect_chessboard_corners
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

def write_orbslam3_mono_yaml_config(path_to_yaml_file, camera_matri, image_size, distortion_coeffs):
    if camera_matri.shape != (3,3):
        raise ValueError("Camera matrix must be of shape (3,3)")
    if distortion_coeffs.shape[0] != 1 or distortion_coeffs.shape[1] not in [4,5,8]:
        raise ValueError("Distortion coefficients must be of shape (1,4), (1,5) or (1,8)")
    with open(path_to_yaml_file, 'w') as f:
        f.write("%YAML:1.0\n\n")
        f.write("# Camera Configuration File\n\n")
        f.write("File.version: \"1.0\" \n")
        f.write("\n# Camera Parameters\n\n")
        f.write(f"Camera.type: \"PinHole\"\n")
        f.write(f"Camera1.fx: {camera_matri[0,0]:.6f}\n")
        f.write(f"Camera1.fy: {camera_matri[1,1]:.6f}\n")
        f.write(f"Camera1.cx: {camera_matri[0,2]:.6f}\n")
        f.write(f"Camera1.cy: {camera_matri[1,2]:.6f}\n")
        f.write(f"Camera1.k1: {distortion_coeffs[0,0]:.12f}\n")
        f.write(f"Camera1.k2: {distortion_coeffs[0,1]:.12f}\n")
        if distortion_coeffs.shape[1] >= 3:
            f.write(f"Camera1.p1: {distortion_coeffs[0,2]:.12f}\n")
        else:
            f.write(f"Camera.1p1: 0.0\n")
        if distortion_coeffs.shape[1] >= 4:
            f.write(f"Camera1.p2: {distortion_coeffs[0,3]:.12f}\n")
        else:
            f.write(f"Camera.1p2: 0.0\n")
        if distortion_coeffs.shape[1] >= 5:
            f.write(f"Camera1.k3: {distortion_coeffs[0,4]:.12f}\n")
        else:
            f.write(f"Camera1.k3: 0.0\n")

        f.write(f"Camera.width: {image_size[0]}\n")
        f.write(f"Camera.height: {image_size[1]}\n")
        f.write(f"Camera.newWidth: {image_size[0]}\n")
        f.write(f"Camera.newHeight: {image_size[1]}\n")
        f.write("Camera.fps: 10\n")
        f.write("Camera.RGB: 1\n")


        f.write("\n# ORB Parameters\n\n")

        f.write("ORBextractor.nFeatures: 1500\n")
        f.write("ORBextractor.scaleFactor: 1.2\n")
        f.write("ORBextractor.nLevels: 8\n")
        f.write("ORBextractor.iniThFAST: 20\n")
        f.write("ORBextractor.minThFAST: 7\n")

        f.write("\n# Viewer Parameters\n\n")

        f.write("Viewer.KeyFrameSize: 0.05\n")
        f.write("Viewer.KeyFrameLineWidth: 1.0\n")
        f.write("Viewer.GraphLineWidth: 0.9\n")
        f.write("Viewer.PointSize: 2.0\n")
        f.write("Viewer.CameraSize: 0.08\n")
        f.write("Viewer.CameraLineWidth: 3.0\n")
        f.write("Viewer.ViewpointX: 0.0\n")
        f.write("Viewer.ViewpointY: -0.7\n")
        f.write("Viewer.ViewpointZ: -1.8\n")
        f.write("Viewer.ViewpointF: 500.0\n")


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
    use_rational_model = args.distortion_rational_model if args.distortion_rational_model else False  # Whether to use rational model (k4, k5, k6)
    flags = cv2.CALIB_FIX_ASPECT_RATIO
    if use_rational_model:
        flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_ASPECT_RATIO


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

    ret, camera_matrix, distortion, rvecs, tvecs, std_dev_int, std_dev_ext, _ = cv2.calibrateCameraExtended(object_points_all_views, image_points_all_views, image_size, cameraMatrix=None, distCoeffs=None, flags=flags)
    
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

    write_orbslam3_mono_yaml_config(os.path.join(image_directory, "orbslam3_mono_camera_config.yaml"), camera_matrix, image_size, distortion[:,1:5])









