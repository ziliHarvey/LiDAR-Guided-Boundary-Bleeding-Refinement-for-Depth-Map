import numpy as np
import cv2
from config import P_rect_2, R_rect_0, T_velo_cam, fu, fv, baseline, winsize
from utils import dataLoader, reproject_to_3D, save_ply
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
            proj_map[v, u] = fu * baseline/ depth
        else:
            proj_map[v, u] = depth
    return proj_map

def search_nearest_r(patch):
    # find the min(ri) for points within the patch
    min_r = 1000
    for i in range(winsize):
        for j in range(winsize):
            cur_r = patch[i][j]
            if cur_r > 0 and cur_r < min_r:
                min_r = cur_r
    return min_r

def compute_weighted_r(patch, r0):
    wr, w = 0.0, 0.0
    # set center to the nearest sampled range
    for i in range(winsize):
        for j in range(winsize):
            cur_r = patch[i][j]
            if cur_r == 0:
                continue
            gs = 1.0 / (1 + np.sqrt((i-winsize/2)**2 + (j-winsize/2)**2))
            gr = 1.0 / (1 + abs(cur_r - r0))
            w += gs * gr
            wr += gs * gr * cur_r
    return wr / w

def bf_vanilla(depth_map):
    h, w = depth_map.shape
    # set bilateral filter size to be 9
    disp_map = np.zeros((h, w))
    for r in range(winsize/2, h-winsize/2):
        for c in range(winsize/2, w-winsize/2):
            # for position (r, c)
            if depth_map[r, c]:
                # continue if at sampled position
                disp_map[r, c] = fu * baseline / depth_map[r, c]
                continue
            # at unsample position, find the nearest point as ri
            patch = depth_map[(r-winsize/2):(r+winsize/2+1), (c-winsize/2):(c+winsize/2+1)]
            r0 = search_nearest_r(patch)
            if r0 == 1000:
                # no points in current patch
                continue
            r0_star = compute_weighted_r(patch, r0)
            disp_map[r, c] = fu * baseline / r0_star
    return disp_map

if __name__ == "__main__":
    data = dataLoader("005500")
    imgL = data.imgL
    pc = data.pc
    depth_map = project_lidar_points(imgL, pc[:, :3].T)
    disp_lidar_raw = project_lidar_points(imgL, pc[:, :3].T, True)
    plt.figure()
    plt.imshow(disp_lidar_raw, 'rainbow', vmin=5, vmax=70)
    plt.axis('off')
    # plt.colorbar()
    plt.show()

    disp_lidar_filtered = bf_vanilla(depth_map)
    plt.figure()
    plt.imshow(disp_lidar_filtered, 'rainbow', vmin=5, vmax=70)
    plt.axis('off')
    # plt.colorbar()
    plt.show()

    disp_psmnet = np.load("../data/prediction/005500.npy")
    plt.figure()
    plt.imshow(disp_psmnet, 'rainbow', vmin=5, vmax=70)
    plt.axis('off')
    # plt.colorbar()
    plt.show()

    points, colors = reproject_to_3D(disp_psmnet, imgL)
    save_ply("../output/psmnet.ply", points, colors)
    points, colors = reproject_to_3D(disp_lidar_filtered, imgL)
    save_ply("../output/bflidar.ply", points, colors)

    print("x")
    print(max(points[:, 0]))
    print(min(points[:, 0]))
    print("y")
    print(max(points[:, 1]))
    print(min(points[:, 1]))
    print("z")
    print(max(points[:, 2]))
    print(min(points[:, 2]))
    # plt.figure()
    # plt.imshow(disp_stereo, 'rainbow')
    # plt.colorbar()
    # plt.show()

