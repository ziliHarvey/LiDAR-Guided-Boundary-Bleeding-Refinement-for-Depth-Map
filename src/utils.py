import numpy as np
import cv2
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
    mask = points[:, 2] < 2
    colors = colors[mask]
    points = points[mask]
    print("points generated successfully...")
    return points, colors



def inverse_rigid_transform(T):
    T_inv = np.zeros_like(T)
    T_inv[0:3, 0:3] = T[0:3, 0:3].T
    T_inv[0:3, 3] = np.dot(-T[0:3, 0:3].T, T[0:3, 3])
    return T_inv 



def save_ply(fn, points, colors):
    # save points to ply file
    # points: 3Xn
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

if __name__ == "__main__":
    data = dataLoader("0000000000")
    cv2.imshow("stereo_left", data.imgL)
    cv2.waitKey(0)
    print(data.pc)