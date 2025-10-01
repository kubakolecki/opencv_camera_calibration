import numpy as np
import glob
import cv2
import os

def detect_chessboard_corners(image_dir, rows=10, cols=12, corner_size=0.073):
    # Define the size of the chessboard (rows, columns)
    checkerboard_size = (cols, rows)  # OpenCV expects (columns, rows)

    image_points_all_views = []  # 2D points in image plane
    object_points_all_views = []  # 3D points in real world space
    image_file_names = []

    object_points = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    object_points[0,:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    object_points *= corner_size  # Scale by the size of a square

    # Get all image paths from the directory
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))

    path_directory_corners = os.path.join(image_dir, "corners")
    if not os.path.exists(path_directory_corners):
        os.makedirs(path_directory_corners)


    if not image_paths:
        print("No images found in the given directory.")
        return

    valid_image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_paths = [img for img in image_paths if img.lower().endswith(valid_image_formats)]
    image_paths.sort()  # Sort the image paths for consistent processing order, this is needed for stereo calibration to match pairs correctly!


    for img_path in image_paths:
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image {img_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try to find the chessboard corners
        is_successfull, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if is_successfull:
            #print(f"Checkerboard detected in {os.path.basename(img_path)}")

            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw the corners
            cv2.drawChessboardCorners(image, checkerboard_size, corners, is_successfull)

            path_file_corners = os.path.join(path_directory_corners, os.path.basename(img_path))
            cv2.imwrite(path_file_corners, image)
            image_points_all_views.append(corners)
            object_points_all_views.append(object_points)
            #print(f"Saved image with corners to {path_file_corners}")

            image_file_names.append(os.path.basename(img_path))

        else:
            print(f"No checkerboard found in {os.path.basename(img_path)} !")
            #appending empty arrays to keep the indexing consistent - this is needed for stereo calibration where you have to match pairs of images!
            image_points_all_views.append(np.array(None))
            object_points_all_views.append(np.array(None))
            image_file_names.append(None)
    
    return image_points_all_views, object_points_all_views, image_file_names
