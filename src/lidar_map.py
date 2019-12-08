import numpy as np
import cv2
from config import P_rect_2, R_rect_0, T_velo_cam, fu, fv, baseline, winsize
from sklearn.cluster import DBSCAN
from numba import jit

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
        u, v = int(uv[i, 0]), int(uv[i, 1])
        if u < 0 or u >= w:
            continue
        if v < 0 or v >= h:
            continue
        if inverse:
            proj_map[v, u] = fu * baseline/ depth
        else:
            proj_map[v, u] = depth
    return proj_map

@jit(nopython=True, fastmath=True)
def find_horizontal_line(depth_map):
    h, w = depth_map.shape
    hrz_sum = 0
    for j in range(w):
        cur_bin = depth_map[:, j]
        i = 0
        while not depth_map[i, j]:
            i += 1
        hrz_sum += i
    return int(hrz_sum/w)

def bf_vanilla_accelerated(imgL, pc):
    depth_map = project_lidar_points(imgL, pc[:, :3].T)
    h, w = depth_map.shape
    disp_map = np.zeros((h, w))
    # find the horizontal line
    i_start = find_horizontal_line(depth_map)
    for i in range(i_start, h-winsize//2):
        for j in range(winsize//2, w-winsize//2):
            if depth_map[i, j]:
                disp_map[i, j] = fu * baseline / depth_map[i, j]
                continue
            # at unsampled position
            patch = depth_map[(i-winsize//2):(i+winsize//2+1), (j-winsize//2):(j+winsize//2+1)]
            i_valid, j_valid = np.nonzero(patch)
            if len(i_valid) < 1:
                continue
            # extract corresponding valid value
            r_valid = patch[patch > 0]
            # form data list [[i1, j1, r1], [i2, j2, r2], ...] to feed into DBSCAN
            X = np.concatenate((i_valid.reshape(-1, 1), j_valid.reshape(-1, 1), r_valid.reshape(-1, 1)), axis=1)
            r0 = np.min(X[:, 2])
            r0_star = compute_weighted_r_sparse(r0, X)
            disp_map[i, j] = fu * baseline / r0_star
    return disp_map

def dispersion(r1, r2):
    # define distance function according to dispersion degree
    return abs((r1-r2)/(r1+r2))

@jit(nopython=True, fastmath=True)
def compute_weighted_r_sparse(r0, IJR):
    # IJR is a (n, 3) array with row, col, range
    # r0 is the nearest value to the central position
    w, wr = 0.0, 0.0
    for idx in range(len(IJR)):
        i, j, r = IJR[idx, 0], IJR[idx, 1], IJR[idx, 2]
        gs = 1.0 / (1 + np.sqrt((i-winsize//2)**2 + (j-winsize//2)**2))
        gr = 1.0 / (1 + abs(r - r0))
        w += gs * gr
        wr += gs * gr * r
    return wr / w

def measure_dispersion(imgL, pc):
    depth_map = project_lidar_points(imgL, pc[:, :3].T)
    h, w = depth_map.shape
    edge_map = np.zeros((h, w), dtype=np.uint8)
    disp_map = np.zeros((h, w))
    count = 0
    count1 = 0
    # find the horizontal line
    i_start = find_horizontal_line(depth_map)
    for i in range(i_start, h-winsize//2):
        for j in range(winsize//2, w-winsize//2):
            # local neighborhood to measure dispersion
            patch = depth_map[(i-winsize//2):(i+winsize//2+1), (j-winsize//2):(j+winsize//2+1)]
            # extract nonzero idx
            i_valid, j_valid = np.nonzero(patch)
            if len(i_valid) <= 5:
                # if there are so few number of points available, skip it
                continue
            # extract corresponding valid value
            r_valid = patch[patch > 0]
            # form data list [[i1, j1, r1], [i2, j2, r2], ...] to feed into DBSCAN
            X = np.concatenate((i_valid.reshape(-1, 1), j_valid.reshape(-1, 1), r_valid.reshape(-1, 1)), axis=1)
            # DBSCAN according to range of each pixel
            clustering = DBSCAN(eps=0.08, min_samples=2, metric=dispersion).fit(X[:, 2].reshape(-1, 1))
            labels = clustering.labels_
            num_cluster = max(labels)
            if num_cluster > 1:
#                 print("*****************************************")
#                 print("number of clusters: " + str(num_cluster+1))
#                 print("-1: " + str(np.count_nonzero(labels==-1)))
#                 print(" 0: " + str(np.count_nonzero(labels== 0)))
#                 print(" 1: " + str(np.count_nonzero(labels== 1)))
#                 print(" 2: " + str(np.count_nonzero(labels== 2)))
#                 print("*****************************************")
                count += 1
            if num_cluster > 0:
                count1 += 1
                # mark this edge pixel
                edge_map[i, j] = 255
                # if alreay point projected
                if depth_map[i, j]:
                    disp_map[i, j] = fu * baseline / depth_map[i, j]
                    continue
                r0 = min(X[labels >= 0][:, 2])
                r0_star = compute_weighted_r_sparse(r0, X[labels >= 0])
                disp_map[i, j] = fu * baseline / r0_star

                # -----------------------------------------------------------------------
                # experimenting with penalizing minor cluster based on threshold as paper suggests
                # but the threshold is hard to empirically determined

                # # most of the time there are only 2 clusters, we assume there're only 2
                # s1_idx = labels == 0
                # s1_num = len(labels[s1_idx])
                # s1_mean = sum(labels[s1_idx])/s1_num
                # s2_idx = labels == 1
                # s2_num = len(labels[s2_idx])
                # s2_mean = sum(labels[s2_idx])/s2_num
                # # here set thr for picking cluster
                # thr = 1
                # # find cluster s1, which is the cluster with min average r
                # if (s1_mean < s2_mean and s1_num/s2_num > thr) or \
                #    (s1_mean > s2_mean and s2_num/s1_num < thr):
                #     r0 = min(X[s1_idx][:, 2])
                #     r0_star = compute_weighted_r_sparse(r0, X[s1_idx])
                # else:
                #     r0 = min(X[s2_idx][:, 2]) 
                #     r0_star = compute_weighted_r_sparse(r0, X[s2_idx])
                # # if s1_mean > s2_mean:
                # #     # switch
                # #     s1_idx, s2_idx = s2_idx, s1_idx
                # #     s1_num, s2_num = s2_num, s1_num
                # # if s1_num/s2_num > 1:
                # #     # run bilater filter only on s1
                # #     r0 = min(X[s1_idx][:, 2])
                # #     r0_star = compute_weighted_r_sparse(r0, X[s1_idx])
                # # else:
                # #     # run bilateral filter only on s2
                # #     r0 = min(X[s2_idx][:, 2]) 
                # #     r0_star = compute_weighted_r_sparse(r0, X[s2_idx])
                # disp_map[i, j] = fu * baseline / r0_star
#     print(">=3 clusters: " + str(count))
#     print(">=2 clusters: " + str(count1))
    return edge_map, disp_map

def replace_boundary(disp_psmnet, disp_bf):
    disp_refined = disp_psmnet.copy()
    h, w = disp_refined.shape
    for i in range(h):
        for j in range(w):
            if disp_bf[i, j] != 0:
                disp_refined[i, j] = disp_bf[i, j]
    return disp_refined
            










