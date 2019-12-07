import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import dataLoader, reproject_to_3D, save_ply, compute_error
from lidar_map import measure_dispersion, replace_boundary
if __name__ == "__main__":

    # test DBSCAN
    filename = "000052"
    data = dataLoader(filename)
    imgL = data.imgL
    pc = data.pc
    edge_map, disp_bf = measure_dispersion(imgL, pc)
    disp_psmnet = cv2.imread("../data/prediction/" + filename + ".png", -1)/256.0
    disp_gt = cv2.imread("../data/gt/disp_occ_0/" + filename + ".png", -1)/256.0
    disp_refined = replace_boundary(disp_psmnet, disp_bf)
    error1, error_map1 = compute_error(disp_gt, disp_refined)    
    error2, error_map2 = compute_error(disp_gt, disp_psmnet)
    print("before refinement..." + str(error2))
    print("after refinement..." + str(error1))

    f = plt.figure()

    ax1 = f.add_subplot(2,3, 1)
    plt.imshow(error_map2, 'rainbow', vmin=-5, vmax=15)
    plt.axis('off')
    ax1.set_title("Error predicted")
    
    ax2 = f.add_subplot(2,3, 2)
    plt.imshow(error_map1, 'rainbow', vmin=-5, vmax=15)
    plt.axis('off')
    ax2.set_title("Error refined")
    
    ax3 = f.add_subplot(2,3, 3)
    plt.imshow(edge_map)
    plt.axis('off')
    ax3.set_title("Edges")
    
    ax4 = f.add_subplot(2,3, 4)
    plt.imshow(disp_psmnet, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax4.set_title("Disparity predicted")
    
    ax5 = f.add_subplot(2,3, 5)
    plt.imshow(disp_refined, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax5.set_title("Disparity refined")

    ax3 = f.add_subplot(2,3, 6)
    plt.imshow(imgL)
    plt.axis('off')
    ax3.set_title("Image")

    plt.show(block=True)

    # points, colors = reproject_to_3D(disp_psmnet, imgL)
    # save_ply("../output/psmnet_" + filename + ".ply", points, colors)
    # points, colors = reproject_to_3D(disp_lidar_filtered, imgL)
    # save_ply("../output/bflidar_" + filename + ".ply", points, colors)