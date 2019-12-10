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
    start = h
    for j in range(w):
        i = 0
        while not depth_map[i, j]:
            i += 1
        start = min(i, start)
    return start

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

@jit(nopython=True)
def update_occupancy(occupancy, bbox):
    """
    occupancy: (h, w) matrix with 1 representing searching areas
    bbox: (n, 4) with left, top, right, bottom pixel coordinates
    """
    margin = 5
    # intialize occupancy grid
    occupancy += -1
    h, w = occupancy.shape
    for i in range(len(bbox)):
        j_start, i_start, j_end, i_end = bbox[i]
        # given relatively loose margin constraint finding both sides of edges
        for r in range(i_start-margin, i_end+margin+1):
            for c in range(j_start-margin, j_end+margin+1):
                if r < 0 or r > h:
                    continue
                if c < 0 or c > w:
                    continue
                occupancy[r, c] = 255

def measure_dispersion(imgL, pc, bbox=[]):
    """
    imgL: (h, w, 3) stereo-left image
    pc: (N, 3) pointcloud
    bbox: (n, 4) with left, top, right, bottom pixel coordinates
    """
    depth_map = project_lidar_points(imgL, pc[:, :3].T)
    h, w = depth_map.shape
    edge_map = np.zeros((h, w), dtype=np.uint8)
    disp_map = np.zeros((h, w))
    # find the horizontal line
    i_start = find_horizontal_line(depth_map)
    # exhaustive search on all pixels if 2D bboxes not provided
    occupancy = np.ones((h, w))
    if len(bbox) > 0:
        # searching on potential foreground areas only
        # instead of doing exhaustive search
        update_occupancy(occupancy, bbox)
    for i in range(i_start, h-winsize//2):
        for j in range(winsize//2, w-winsize//2):
            if occupancy[i, j] == 0:
                continue
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
            if num_cluster > 0:
                # mark this edge pixel
                edge_map[i, j] = 255
                # if alreay point projected
                if depth_map[i, j]:
                    disp_map[i, j] = fu * baseline / depth_map[i, j]
                    continue
                r0 = min(X[labels >= 0][:, 2])
                r0_star = compute_weighted_r_sparse(r0, X[labels >= 0])
                disp_map[i, j] = fu * baseline / r0_star
    return edge_map, disp_map

def replace_boundary(disp_psmnet, disp_bf):
    disp_refined = disp_psmnet.copy()
    h, w = disp_refined.shape
    for i in range(h):
        for j in range(w):
            if disp_bf[i, j] != 0:
                disp_refined[i, j] = disp_bf[i, j]
    return disp_refined
            










