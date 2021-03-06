import numpy as np
import cv2
import skimage
from config import (path_image_02, path_image_03, path_velodyne_points,
                    baseline, cu, cv, fu, fv, bx, by, R_rect_0, T_velo_cam)

class dataLoader:
    def __init__(self, index):
        self.index = index
        self.imgL = None
        self.imgR = None
        self.pc = None
        self.__initialize()

    def __initialize(self):
        self.imgL = self.__read_img(True)
        self.imgR = self.__read_img(False)
        self.pc   = self.__read_points().reshape(-1, 4) 

    def __read_img(self, is_left):
        if is_left:
            file_path = path_image_02 + self.index + ".png" 
        else:
            file_path = path_image_03 + self.index + ".png" 
        try:
            img = cv2.imread(file_path, -1)
#             img = skimage.io.imread(file_path).astype('uint16')
            return img
        except:
            print("The image file doesn't exisit...\n")
        

    def __read_points(self):
        file_path = path_velodyne_points + self.index + ".bin"
        try:
            pc = np.fromfile(file_path, dtype=np.float32)
            return pc 
        except:
            print("The pointcloud file doesn't exist...\n")

def reproject_to_3D(disp, img):
    # following https://github.com/mileyan/pseudo_lidar/blob/1ed136473047219caf24a563396eddb6d12923a2/preprocessing/kitti_util.py
    # reproject disparity map to 3D points
    print('generating 3d point cloud ...')
    h, w = img.shape[:2]
    points = []
    colors = []
    # image to rect
    for v in range(h):
        for u in range(w):
            if disp[v, u] == 0:
                continue
            z = fu * baseline / disp[v, u]
            x = (u - cu) * z / fu + bx
            y = (v - cv) * z / fv + by
            r = img[v, u, 0]
            g = img[v, u, 1]
            b = img[v, u, 2]
            points.append([x, y, z])
            colors.append([r, g, b])
    points = np.array(points).reshape(-1, 3)
    colors = np.array(colors).reshape(-1, 3)
    # rect to ref
    points = np.dot(np.linalg.inv(R_rect_0[:3, :3]), points.T).T
    # ref to velo
    T_cam_velo = inverse_rigid_transform(T_velo_cam[:3, :])
    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    points = np.dot(points_h, T_cam_velo.T)
    mask = points[:, 2] < 1
    colors = colors[mask]
    points = points[mask]
    print("points generated successfully...")
    return points, colors



def inverse_rigid_transform(T):
    T_inv = np.zeros_like(T)
    T_inv[0:3, 0:3] = T[0:3, 0:3].T
    T_inv[0:3, 3] = np.dot(-T[0:3, 0:3].T, T[0:3, 3])
    return T_inv 


def save_bin(fn, points):
    # save points in bin file
    # points (n, 3)
    # bin (n, 4)
    num = points.shape[0]
    points_h = np.concatenate((points, np.ones(num).reshape(-1, 1)), axis=1).astype(np.float32)
    points_h.tofile(fn)
    
def save_ply(fn, points, colors):
    # save points to ply file
    # points: (n, 3)
    num = points.shape[0]
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = np.hstack([points, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=num)).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        
def compute_error(gt_disp,pred_disp, obj_map):
    '''
    input: 
          gt_disp: ground truth disparity, if saved as .png file, need to use 
                   (skimage.io.imread(filepath).astype('float32')) / 256.0 to process
          pred_disp: predicted disparity, if saved as .png file, following the same procedure.
          obj_map: mask of foreground objects
    output:
          return error percentage
    '''

    valid_index_gt = np.argwhere(gt_disp > 0) 
    valid_index_pred = np.argwhere(pred_disp > 0)
    valid_index_fg = np.argwhere(obj_map > 0)
    valid_index_bg = np.argwhere(obj_map == 0)
    # computing error of all objects
    valid_index = np.array([x for x in set(tuple(x) for x in valid_index_gt) & set(tuple(x) for x in valid_index_pred)])
    valid_gt_disp = gt_disp[valid_index[:,0],valid_index[:,1]]
    valid_pred_disp = pred_disp[valid_index[:,0],valid_index[:,1]]
    error = np.zeros_like(gt_disp)-5
    error[valid_index[:,0],valid_index[:,1]] =  np.abs(valid_gt_disp - valid_pred_disp)
    correct_count = (error[valid_index[:,0],valid_index[:,1]] < 3) | \
                    (error[valid_index[:,0],valid_index[:,1]] < valid_gt_disp * 0.05)
    
    # computing error of foreground objects
    valid_index_fg = np.array([x for x in (set(tuple(x) for x in valid_index_gt) & set(tuple(x) for x in valid_index_pred)) & set(tuple(x) for x in valid_index_fg)])
    valid_gt_disp_fg = gt_disp[valid_index_fg[:,0],valid_index_fg[:,1]]
    valid_pred_disp_fg = pred_disp[valid_index_fg[:,0],valid_index_fg[:,1]]
    error_fg = np.zeros_like(gt_disp)-5
    error_fg[valid_index_fg[:,0],valid_index_fg[:,1]] =  np.abs(valid_gt_disp_fg - valid_pred_disp_fg)
    correct_count_fg = (error_fg[valid_index_fg[:,0],valid_index_fg[:,1]] < 3) | \
                    (error_fg[valid_index_fg[:,0],valid_index_fg[:,1]] < valid_gt_disp_fg * 0.05)
    
    # computing error of background objects
    valid_index_bg = np.array([x for x in (set(tuple(x) for x in valid_index_gt) & set(tuple(x) for x in valid_index_pred)) & set(tuple(x) for x in valid_index_bg)])
    valid_gt_disp_bg = gt_disp[valid_index_bg[:,0],valid_index_bg[:,1]]
    valid_pred_disp_bg = pred_disp[valid_index_bg[:,0],valid_index_bg[:,1]]
    error_bg = np.zeros_like(gt_disp)-5
    error_bg[valid_index_bg[:,0],valid_index_bg[:,1]] =  np.abs(valid_gt_disp_bg - valid_pred_disp_bg)
    correct_count_bg = (error_bg[valid_index_bg[:,0],valid_index_bg[:,1]] < 3) | \
                    (error_bg[valid_index_bg[:,0],valid_index_bg[:,1]] < valid_gt_disp_bg * 0.05)
    count_above_15 = np.count_nonzero(error >= 15)
    return 1 - (float(sum(correct_count))/ float(valid_index.shape[0])), 1 - (float(sum(correct_count_fg))/ float(valid_index_fg.shape[0])), 1 - (float(sum(correct_count_bg))/ float(valid_index_bg.shape[0])), error, count_above_15