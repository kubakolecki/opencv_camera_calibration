import cv2
import os
import numpy as np
import scipy.spatial.transform as transform

class ArucCornerDatra:
    def __init__(self, marker_id, corner_id, x, y, z):
        self.marker_id = marker_id
        self.corner_id = corner_id
        self.id = f"{marker_id}_{corner_id}"  # unique id for marker corner
        self.x = x
        self.y = y
        self.z = z

def load_calibration_data(path_file_calibration_data):
    """Load camera calibration data from a YAML file."""
    fs = cv2.FileStorage(path_file_calibration_data, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def load_image_size(path_file_calibration_data):
    """Load image size from a YAML file."""
    fs = cv2.FileStorage(path_file_calibration_data, cv2.FILE_STORAGE_READ)
    image_width = fs.getNode("image_width").real()
    image_height = fs.getNode("image_height").real()
    fs.release()
    return [int(image_width), int(image_height)]

def load_aruco_3D_coordinates(path_file_aruco_coords_3D):
    #The ID mask should be MMMC, where M is the marker id and C is the corner ID
    aruco_3D_coordinates = {}
    with open(path_file_aruco_coords_3D, 'r') as f:
        lines = f.readlines()
        for line in lines:  # Skip header
            parts = line.strip().split(',')
            marker_id_raw = parts[0]
            marker_id = int(marker_id_raw[0:3])
            corner_id = int(marker_id_raw[3]) - 1 #for image coordinates the first id is 0           
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            aruco_3D_coordinates[f"{marker_id}_{corner_id}"] = ArucCornerDatra(marker_id, corner_id, x, y, z)
    return aruco_3D_coordinates

def make_transformation_matrix(rotation_matrix, translation_vector):
    T = np.eye(4)
    T[0:3, 0:3] = rotation_matrix
    T[0:3, 3] = translation_vector.flatten()
    return T

def write_nodistortion_camera_calibration_parameters_to_file(file, camera_matrix, camera_name):
    file.write(f"{camera_name}.fx: {camera_matrix[0,0]:.5f}\n")
    file.write(f"{camera_name}.fy: {camera_matrix[1,1]:.5f}\n")
    file.write(f"{camera_name}.cx: {camera_matrix[0,2]:.5f}\n")
    file.write(f"{camera_name}.cy: {camera_matrix[1,2]:.5f}\n")
    file.write(f"#{camera_name}.k1: 0.0\n")
    file.write(f"#{camera_name}.k2: 0.0\n")
    file.write(f"#{camera_name}.p1: 0.0\n")
    file.write(f"#{camera_name}.p2: 0.0\n")
    file.write(f"#{camera_name}.k3: 0.0\n")

path_file_aruco_measurements_left = "/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/004LZ/undistorted/aruco_image_coordinates_refined_verified.txt"
path_file_aruco_measurements_right = "/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/004M1/undistorted/aruco_image_coordinates_refined_verified.txt"
path_file_aruco_3D_coordinates =  "/datadisk/data/agh_projects/aruco_lab_05/tachymetry/opal_output/landmarks_optimized.txt"
path_file_calibration_data_left = "/datadisk/data/agh_projects/miss/camera_calibration/20251017_calibration_2_7mm_lens/004LZ/calibration_undistorted.yaml"
path_file_calibration_data_right = "/datadisk/data/agh_projects/miss/camera_calibration/20251017_calibration_2_7mm_lens/004M1/calibration_undistorted.yaml"
path_initial_transformation_matrix = "/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/T.txt"
path_directory_output = "/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/"
refine_focal_length = True
image_ids_to_exclude  = ['21.png','26.png','27.png','55.png','01.png','37.png']

aruco_measurements_left = np.loadtxt(path_file_aruco_measurements_left, skiprows=1,  delimiter=',', dtype=str)
aruco_measurements_right = np.loadtxt(path_file_aruco_measurements_right, skiprows=1,  delimiter=',', dtype=str)
aruco_3D_coordinates = load_aruco_3D_coordinates(path_file_aruco_3D_coordinates)
camera_matrix_left, dist_coeffs_left = load_calibration_data(path_file_calibration_data_left)
camera_matrix_right, dist_coeffs_right = load_calibration_data(path_file_calibration_data_right)

image_size_left = load_image_size(path_file_calibration_data_left)
image_size_right = load_image_size(path_file_calibration_data_right)
if image_size_left != image_size_right:
    raise ValueError("Left and right camera image sizes do not match!")
image_size = tuple(image_size_left)

# Filter out excluded images
mask_left = np.isin(aruco_measurements_left[:,0], image_ids_to_exclude, invert=True)
aruco_measurements_left = aruco_measurements_left[mask_left]
mask_right = np.isin(aruco_measurements_right[:,0], image_ids_to_exclude, invert=True)
aruco_measurements_right = aruco_measurements_right[mask_right]

image_ids_left = list(set(aruco_measurements_left[:,0].flatten().tolist()))
image_ids_right = list(set(aruco_measurements_right[:,0].flatten().tolist()))
image_ids_left.sort()
image_ids_right.sort()  

if set(image_ids_left) != set(image_ids_right):
    raise ValueError("The sets of image IDs in left and right measurements do not match!")

image_coordinates_for_all_images_left = []
image_coordinates_for_all_images_right = []
object_coordinates_for_all_images = []

for image_id in image_ids_left:
    #print(f"Processing image ID: {image_id}")
    mask_left = np.isin(aruco_measurements_left[:,0], [image_id], invert=False)
    mask_right = np.isin(aruco_measurements_right[:,0], [image_id], invert=False)
    aruco_measurements_left_selected = aruco_measurements_left[mask_left]
    aruco_measurements_right_selected = aruco_measurements_right[mask_right]
    aruco_measurements_left_selected = aruco_measurements_left_selected[:,1:].astype('float') 
    aruco_measurements_right_selected = aruco_measurements_right_selected[:,1:].astype('float')
    image_points_left = []
    image_points_right = []
    object_points = []
    for aruco_marker_left in aruco_measurements_left_selected:
        point_id_left = str(int(aruco_marker_left[0])) + "_" + str(int(aruco_marker_left[1]))
        if not point_id_left in aruco_3D_coordinates:
            print(f"point id {point_id_left} not found in set of 3D marker coordinates")
            continue
        for aruco_marker_right in aruco_measurements_right_selected:
            if aruco_marker_right[0] == aruco_marker_left[0] and aruco_marker_right[1] == aruco_marker_left[1]:
                image_points_left.append(aruco_marker_left[2:4].tolist())
                image_points_right.append(aruco_marker_right[2:4].tolist())
                x = aruco_3D_coordinates[point_id_left].x
                y = aruco_3D_coordinates[point_id_left].y
                z = aruco_3D_coordinates[point_id_left].z
                object_points.append([x,y,z])
                break
    image_coordinates_for_all_images_left.append(np.array(image_points_left).astype(np.float32))
    image_coordinates_for_all_images_right.append(np.array(image_points_right).astype(np.float32))
    object_coordinates_for_all_images.append(np.array(object_points).astype(np.float32))

#checking sizes
if len(image_coordinates_for_all_images_left) != len(image_coordinates_for_all_images_left):
    raise ValueError("We have to observe same number of views from left and from right camera.")

for image_points_left, image_points_right in zip(image_coordinates_for_all_images_left, image_coordinates_for_all_images_right):
    if image_points_left.shape != image_points_right.shape:
        raise ValueError("Number of points observed in each view must be the same for left and for right camera.")
    
initial_T_l_r = np.loadtxt(path_initial_transformation_matrix, delimiter=',', dtype=float)
initial_T_r_l = np.linalg.inv(initial_T_l_r)
initial_T_r_l_copy = initial_T_r_l.copy() #to use later for comparison!
#************** cv2.stereoCalibrateExtended MODIFIES initial_R_r_l and initial_t_r_l !!!!!!***************
#same for camera matrices if CALIB_USE_INTRINSIC_GUESS flag is used
camera_matrix_left_copy = camera_matrix_left.copy()
camera_matrix_right_copy = camera_matrix_right.copy()


print("Initial transformation matrix T_l_r:")
print(initial_T_l_r)

initial_t_r_l = initial_T_r_l[0:3,3].reshape((3,1))
initial_R_r_l = initial_T_r_l[0:3,0:3]

#print(object_coordinates_for_all_images[0])
#print(image_coordinates_for_all_images_left[0])
#print(image_coordinates_for_all_images_right[0])

flags = 0
flags = flags | cv2.CALIB_USE_EXTRINSIC_GUESS
if not refine_focal_length:
    flags = flags | cv2.CALIB_FIX_INTRINSIC
else:
    flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT #| cv2.CALIB_FIX_ASPECT_RATIO 
    flags = flags | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6


_, _, _, _, _, R, T, E, F, rvects, tvects, per_view_errors = cv2.stereoCalibrateExtended(
                    objectPoints=object_coordinates_for_all_images,
                    imagePoints1=image_coordinates_for_all_images_left,
                    imagePoints2=image_coordinates_for_all_images_right,
                    cameraMatrix1=camera_matrix_left,
                    distCoeffs1=dist_coeffs_left,
                    cameraMatrix2=camera_matrix_right,
                    distCoeffs2=dist_coeffs_right,
                    imageSize=image_size,
                    R=initial_R_r_l,
                    T=initial_t_r_l,
                    flags=flags)

estimated_T_r_l = make_transformation_matrix(R, T)
estimated_T_l_r = np.linalg.inv(estimated_T_r_l)

print("Estimated transformation matrix T_r_l:")
print(estimated_T_l_r)

T_l_l = estimated_T_l_r @ initial_T_r_l_copy
print("Difference beetween initial and estimated transformation (T_r_r = T_r_l_estimated * T_l_r_initial):")
print(T_l_l)

angular_difference_as_euler = transform.Rotation.from_matrix(T_l_l[0:3,0:3]).as_euler('XYZ', degrees=True)
print("Angular difference (degrees), Euler angles XYZ:")
print(angular_difference_as_euler)
print("Translation difference (units of 3D marker coordinates):")
print(T_l_l[0:3,3].flatten())

if refine_focal_length:
    print("difference in left camera matrix:")
    print(camera_matrix_left - camera_matrix_left_copy)
    print("difference in right camera matrix:")
    print(camera_matrix_right - camera_matrix_right_copy)
    path_file_refined_camera_matrix_left = os.path.join(path_directory_output, "refined_camera_matrix_left.txt")
    path_file_refined_camera_matrix_right = os.path.join(path_directory_output, "refined_camera_matrix_right.txt")
    np.savetxt(path_file_refined_camera_matrix_left, camera_matrix_left, fmt='%.5f', delimiter=',')
    np.savetxt(path_file_refined_camera_matrix_right, camera_matrix_right, fmt='%.5f', delimiter=',')

print("\nPer view reprojection errors:")
for i in range(len(per_view_errors)):
    print(f"View {i} (Image ID: {image_ids_left[i]}): {per_view_errors.flatten()[i]:.4f} pixels")

path_file_output_stereo_calibration = os.path.join(path_directory_output, "stereo_calibration_T_l_r.txt")
np.savetxt(path_file_output_stereo_calibration, estimated_T_l_r, delimiter=',', fmt='%.18f')

path_file_output_orbslam_config = os.path.join(path_directory_output, "orbslam_stereo_config.yaml")
with open(path_file_output_orbslam_config,'w') as file_orbslam_config:
    file_orbslam_config.write("%YAML:1.0\n")
    file_orbslam_config.write("File.version: \"1.0\"\n\n")
    file_orbslam_config.write("Camera.type: \"PinHole\"\n\n")
    file_orbslam_config.write("# Camera calibration and distortion parameters (OpenCV)\n")
    write_nodistortion_camera_calibration_parameters_to_file(file_orbslam_config, camera_matrix_left, "Camera1" )
    file_orbslam_config.write("\n")
    write_nodistortion_camera_calibration_parameters_to_file(file_orbslam_config, camera_matrix_right, "Camera2" )
    file_orbslam_config.write("\n")

    file_orbslam_config.write(f"Camera.width: {image_size[0]}\n")
    file_orbslam_config.write(f"Camera.height: {image_size[1]}\n")
    file_orbslam_config.write(f"Camera.fps: 10\n")
    file_orbslam_config.write(f"Camera.RGB: 0\n")
    file_orbslam_config.write(f"Stereo.ThDepth: 60.0\n")
    file_orbslam_config.write(f"Stereo.T_c1_c2: !!opencv-matrix\n")
    file_orbslam_config.write(f"  rows: 4\n")
    file_orbslam_config.write(f"  cols: 4\n")
    file_orbslam_config.write(f"  dt: f\n")
    file_orbslam_config.write(f"  data: [{estimated_T_l_r[0,0]:.18f},{estimated_T_l_r[0,1]:.18f},{estimated_T_l_r[0,2]:.18f},{estimated_T_l_r[0,3]:.18f},\n")
    file_orbslam_config.write(f"         {estimated_T_l_r[1,0]:.18f},{estimated_T_l_r[1,1]:.18f},{estimated_T_l_r[1,2]:.18f},{estimated_T_l_r[1,3]:.18f},\n")
    file_orbslam_config.write(f"         {estimated_T_l_r[2,0]:.18f},{estimated_T_l_r[2,1]:.18f},{estimated_T_l_r[2,2]:.18f},{estimated_T_l_r[2,3]:.18f},\n")
    file_orbslam_config.write(f"         {estimated_T_l_r[3,0]:.18f},{estimated_T_l_r[3,1]:.18f},{estimated_T_l_r[3,2]:.18f},{estimated_T_l_r[3,3]:.18f}]\n")
    file_orbslam_config.write("\n\n")

    file_orbslam_config.write("\n# ORB Parameters\n\n")
    file_orbslam_config.write("ORBextractor.nFeatures: 1500\n")
    file_orbslam_config.write("ORBextractor.scaleFactor: 1.2\n")
    file_orbslam_config.write("ORBextractor.nLevels: 8\n")
    file_orbslam_config.write("ORBextractor.iniThFAST: 20\n")
    file_orbslam_config.write("ORBextractor.minThFAST: 7\n")
    file_orbslam_config.write("\n# Viewer Parameters\n\n")
    file_orbslam_config.write("Viewer.KeyFrameSize: 0.05\n")
    file_orbslam_config.write("Viewer.KeyFrameLineWidth: 1.0\n")
    file_orbslam_config.write("Viewer.GraphLineWidth: 0.9\n")
    file_orbslam_config.write("Viewer.PointSize: 2.0\n")
    file_orbslam_config.write("Viewer.CameraSize: 0.08\n")
    file_orbslam_config.write("Viewer.CameraLineWidth: 3.0\n")
    file_orbslam_config.write("Viewer.ViewpointX: 0.0\n")
    file_orbslam_config.write("Viewer.ViewpointY: -0.7\n")
    file_orbslam_config.write("Viewer.ViewpointZ: -1.8\n")
    file_orbslam_config.write("Viewer.ViewpointF: 500.0\n")




