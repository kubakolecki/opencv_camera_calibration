import numpy as np
import cv2
import os

def generate_undistortion_map(camera_matrix, distoriton, new_camera_matrix, image_size):
    return cv2.initUndistortRectifyMap(camera_matrix, distoriton, R=None, newCameraMatrix=new_camera_matrix, size=image_size, m1type=cv2.CV_32FC1)

def undistort_images(image_dir, map1, map2):

    path_directory_undisto = os.path.join(image_dir, "undistorted")
    if not os.path.exists(path_directory_undisto):
        os.makedirs(path_directory_undisto)

    entries = os.listdir(image_dir)
    
    valid_image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_file_names = [f for f in entries if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(valid_image_formats)]


    for image_file_name in image_file_names:
        img_path = os.path.join(image_dir, image_file_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image {img_path}")
            continue

        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join(path_directory_undisto, os.path.basename(img_path))
        cv2.imwrite(output_path, undistorted_image)
        #print(f"Undistorted image saved to {output_path}")