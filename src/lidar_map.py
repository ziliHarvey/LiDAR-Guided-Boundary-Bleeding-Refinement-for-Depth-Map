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

@jit(nopython=True, fastmath=True)
def compute_weighted_r(patch, r0):
    wr, w = 0.0, 0.0
    # set center to the nearest sampled range
    for i in range(winsize):
        for j in range(winsize):
            cur_r = patch[i][j]
            if cur_r == 0:
                continue
            gs = 1.0 / (1 + np.sqrt((i-winsize//2)**2 + (j-winsize//2)**2))
            gr = 1.0 / (1 + abs(cur_r - r0))
            w += gs * gr
            wr += gs * gr * cur_r
    return wr / w

def bf_vanilla(depth_map):
    h, w = depth_map.shape
    # set bilateral filter size to be 9
    disp_map = np.zeros((h, w))
    for r in range(winsize//2, h-winsize//2):
        for c in range(winsize//2, w-winsize//2):
            # for position (r, c)
            if depth_map[r, c]:
                # continue if at sampled position
                disp_map[r, c] = fu * baseline / depth_map[r, c]
                continue
            # at unsample position, find the nearest point as ri
            patch = depth_map[(r-winsize//2):(r+winsize//2+1), (c-winsize//2):(c+winsize//2+1)]
            r0 = search_nearest_r(patch)
            if r0 == 1000:
                # no points in current patch
                continue
            r0_star = compute_weighted_r(patch, r0)
            disp_map[r, c] = fu * baseline / r0_star
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
    for i in range(winsize//2, h-winsize//2):
        for j in range(winsize//2, w-winsize//2):
            # local neighborhood to measure dispersion
            patch = depth_map[(i-winsize//2):(i+winsize//2+1), (j-winsize//2):(j+winsize//2+1)]
            # extract nonzero idx
            i_valid, j_valid = np.nonzero(patch)
            # extract corresponding valid value
            r_valid = patch[patch > 0]
            # form data list [[i1, j1, r1], [i2, j2, r2], ...] to feed into DBSCAN
            X = np.concatenate((i_valid.reshape(-1, 1), j_valid.reshape(-1, 1), r_valid.reshape(-1, 1)), axis=1)
            if len(X) <= 5:
                # if there are so few number of points available, skip it
                continue
            # DBSCAN according to range of each pixel
            clustering = DBSCAN(eps=0.05, min_samples=2, metric=dispersion).fit(X[:, 2].reshape(-1, 1))
            labels = clustering.labels_
            num_cluster = max(labels)
            if num_cluster > 0:
                # mark this edge pixel
                edge_map[i, j] = 255
                # if alreay point projected
                if depth_map[i, j]:
                    disp_map[i, j] = fu * baseline / depth_map[i, j]
                    continue
                r0 = search_nearest_r(patch)
                r0_star = compute_weighted_r(patch, r0)
                disp_map[i, j] = fu * baseline / r0_star

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
    return edge_map, disp_map

def replace_boundary(disp_psmnet, disp_bf):
    disp_refined = disp_psmnet.copy()
    h, w = disp_refined.shape
    for i in range(h):
        for j in range(w):
            if disp_bf[i, j] != 0:
                disp_refined[i, j] = disp_bf[i, j]
    return disp_refined
            










