import numpy as np
import cv2 as cv
from config import P_rect_2, R_rect_0, T_velo_cam, f, b
from utils import dataLoader
import matplotlib.pyplot as plt

def project_lidar_points(img, XYZ, inverse=False):
    """
    img: (h, w, 3) uint8
    XYZ: (3, num_pts) np.float32
    inverse: True: disparity map | False: depth map, bool
    """
    h, w = img.shape[:2]
    num_pts = XYZ.shape[1]
    XYZ_h = np.concatenate((XYZ, np.ones((1, num_pts))), axis=0)
    uv_h = np.dot(P_rect_2, np.dot(R_rect_0, np.dot(T_velo_cam, XYZ_h)))
    uv = (uv_h / uv_h[2, :])[:2, :].T
    proj_map = np.zeros((h, w))
    for i in range(num_pts):
        depth = XYZ[0, i]
        if depth < 0:
            continue 
        u, v = uv[i].astype(int)
        if u < 0 or u >= w:
            continue
        if v < 0 or v >= h:
            continue
        if inverse:
            proj_map[v, u] = f * b / depth
        else:
            proj_map[v, u] = depth
    return proj_map



if __name__ == "__main__":
    data = dataLoader("0000000000")
    imgL = data.imgL
    pc = data.pc
    disp_map = project_lidar_points(imgL, pc[:, :3].T, True)
    plt.figure()
    plt.imshow(disp_map, 'rainbow')
    plt.colorbar()
    plt.show()
