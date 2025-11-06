import numpy as np
import scipy.spatial.transform as transform

def make_transformation_matrix(rotation_matrix, translation_vector):
    T = np.eye(4)
    T[0:3, 0:3] = rotation_matrix
    T[0:3, 3] = translation_vector.flatten()
    return T

path_file_poses_left = '/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/poses_004LZ_verified.txt'
path_file_poses_right = '/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/poses_004M1_verified.txt'
path_file_output_transformation_matrix_l_r = '/datadisk/data/agh_projects/miss/camera_calibration/20251031_stereo_calibration_005lab_2_7mm_lens/T.txt'
#poses_to_exclude  = [8,20,25,26, 31]
#poses_to_exclude  = [8]
poses_to_exclude  = [20,25,26,54,0]

#math convention:
#l - left camera
#r - right camera
#w - world
#R - rotation matrix
#q - quaternion
#t - position vector
#T - transformation matrix

poses_left = np.loadtxt(path_file_poses_left, skiprows=1, usecols=(1,2,3,4,5,6,7), dtype=float, delimiter=',')
poses_right = np.loadtxt(path_file_poses_right, skiprows=1, usecols=(1,2,3,4,5,6,7), dtype=float, delimiter=',')

if poses_left.shape[0] != poses_right.shape[0]:
    raise ValueError("Number of poses in left and right files do not match before exclusion.")

id_of_poses = np.arange(poses_left.shape[0])

poses_left = np.delete(poses_left, poses_to_exclude, axis=0)
poses_right = np.delete(poses_right, poses_to_exclude, axis=0)
id_of_poses = np.delete(id_of_poses, poses_to_exclude, axis=0)

num_of_poses = poses_left.shape[0]
 
if num_of_poses != poses_right.shape[0]:
    raise ValueError("Number of poses in left and right files do not match.")

T_l_r_list = []
q_l_r_list = []

for i in range(num_of_poses):
    # Left camera pose in world coordinates
    t_w_l = poses_left[i, 0:3].reshape((3,1))
    q_w_l = poses_left[i, 3:7]
    R_w_l = transform.Rotation.from_quat([q_w_l[0], q_w_l[1], q_w_l[2], q_w_l[3]], scalar_first=True ).as_matrix()
    T_w_l = make_transformation_matrix(R_w_l, t_w_l)

    # Right camera pose in world coordinates
    t_w_r = poses_right[i, 0:3].reshape((3,1))
    q_w_r = poses_right[i, 3:7]
    R_w_r = transform.Rotation.from_quat([q_w_r[0], q_w_r[1], q_w_r[2], q_w_r[3]], scalar_first=True).as_matrix()
    T_w_r = make_transformation_matrix(R_w_r, t_w_r)

    T_l_r = np.linalg.inv(T_w_l) @ T_w_r
    print(f"Pose {i}:")
    print(T_l_r)
    T_l_r_list.append(T_l_r)

    q_l_r = transform.Rotation.from_matrix(T_l_r[0:3,0:3]).as_quat()
    q_l_r_list.append(q_l_r)


rotation_l_r = transform.Rotation.from_quat(q_l_r_list)
mean_R_l_r = rotation_l_r.mean().as_matrix()

#computing rotation residuals:

rotation_residuals_absolute_deg = []

for pose_id, T_l_r in zip (id_of_poses, T_l_r_list):
    residual_R_l_r = np.transpose(T_l_r[0:3,0:3])@mean_R_l_r
    residual_rotvec = transform.Rotation.from_matrix(residual_R_l_r).as_rotvec()
    residual_angle_absolute_deg = np.linalg.norm(residual_rotvec)*180.0/np.pi
    rotation_residuals_absolute_deg.append((pose_id,residual_angle_absolute_deg))
    #print(f"pose: {id}, residual: {residual_angle_absolute_deg} [deg]")

rotation_residuals_absolute_deg.sort(key=lambda x: x[1], reverse=True)
print("Rotation residuals (absolute) SORTED [deg]:")
for res in rotation_residuals_absolute_deg:
    print(f"pose: {res[0]}, residual: {res[1]} [deg]")

translation_vectors = [T_l_r[0:3,3] for T_l_r in T_l_r_list]
mean_t_l_r = np.mean(translation_vectors, axis=0).reshape((3,1))

#computing translation residuals:
translation_residuals_absolute = []
for id, T_l_r in zip(id_of_poses, T_l_r_list):
    residual_t_l_r = T_l_r[0:3,3].reshape((3,1)) - mean_t_l_r
    residual_distance_absolute = np.linalg.norm(residual_t_l_r)
    translation_residuals_absolute.append((id,residual_distance_absolute))

translation_residuals_absolute.sort(key=lambda x: x[1], reverse=True)
print("Translation residuals (absolute) SORTED:")
for res in translation_residuals_absolute:
    print(f"pose: {res[0]}, residual: {res[1]}")

mean_T_l_r = make_transformation_matrix(mean_R_l_r, mean_t_l_r)

print("Mean transformation matrix T_l_r:")
print(mean_T_l_r)

np.savetxt(path_file_output_transformation_matrix_l_r, mean_T_l_r, fmt='%.15f', delimiter=',')

