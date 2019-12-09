import numpy as np
import cv2
from utils import dataLoader, reproject_to_3D, save_bin
from lidar_map import measure_dispersion, replace_boundary, bf_vanilla_accelerated

if __name__ == "__main__":
    filename = "000036"
    data = dataLoader(filename)
    imgL = data.imgL
    pc = data.pc
    
    # upsampling
    disp_lidar = bf_vanilla_accelerated(imgL, pc)
    points, colors = reproject_to_3D(disp_lidar, imgL)
    save_bin("../output/" + filename + "_upsampled.bin", points)
    
    # testcase
    file = "../output/" + filename + "_upsampled.bin"
    pts = np.fromfile(file, np.float32)
    pts_reshaped = pts.reshape(-1, 4)
    
    # prediction
    disp_psmnet = cv2.imread("../data/prediction/" + filename + ".png", -1)/256.0
    points, colors = reproject_to_3D(disp_psmnet, imgL)
    save_bin("../output/" + filename + "_predicted.bin", points)
    
    # refined
    edge_map, disp_bf = measure_dispersion(imgL, pc)
    disp_refined = replace_boundary(disp_psmnet, disp_bf)
    points, colors = reproject_to_3D(disp_refined, imgL)
    save_bin("../output/" + filename + "_refined.bin", points)
    
    