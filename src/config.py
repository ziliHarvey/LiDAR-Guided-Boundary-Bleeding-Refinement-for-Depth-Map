import numpy as np

# directory to data
path_image_02 = "../data/image_02/"
path_image_03 = "../data/image_03/"
path_velodyne_points = "../data/velodyne_points/"

# stereo camera parameters
# left: camera 02 (reference) | right: camera 03
# x = right, y = down, z = forward
# focal length, u means horizontally and v means vertically
# fu = fv in KITTI dataset
fu = 7.215377e+02
fv = 7.215377e+02
# baseline
baseline = (4.485728e+01 + 3.395242e+02) / 7.215377e+02
# principle points
cu = 6.095593e+02
cv = 1.728540e+02
# relative distance
bx = 4.485728e+01 / (-fu)
by = 2.163791e-01 / (-fv)
# projection matrix
# Velodyne: x = forward, y = left, z = up
# from LiDAR to camera 00 
# (u, v, 1)' = P_rect_0 * R_rect_0 * T_velo_cam * (X, Y, Z, 1)'
P_rect_2 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                     [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])
# R_rect_0 has been expanded by appending a fourth zero-row and column, and setting R_rect_0(4, 4) = 1
R_rect_0 = np.array([[ 9.999239e-01, 9.837760e-03, -7.445048e-03, 0.000000e+00],
                     [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.000000e+00],
                     [ 7.402527e-03, 4.351614e-03,  9.999631e-01, 0.000000e+00],
                     [ 0.000000e+00, 0.000000e+00,  0.000000e+00, 1.000000e+00]])
# transformation matrix from velodyne coordinates to camera coordinates
T_velo_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                       [1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                       [9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
                       [0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
# bilateral filter size
winsize = 13