import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import dataLoader, reproject_to_3D, save_ply, compute_error
from lidar_map import measure_dispersion, replace_boundary, bf_vanilla_accelerated
import time

if __name__ == "__main__":

    # test DBSCAN
    filename = "000051"
    data = dataLoader(filename)
    imgL = data.imgL
    pc = data.pc
    
    print("Upsampling(accelerated) begins...")
    start_acc = time.time()
    disp_lidar = bf_vanilla_accelerated(imgL, pc)
    end_acc = time.time()
    elapse_acc = end_acc - start_acc
    print("Upsampling(accelerated) on raw points takes " + str(elapse_acc) + " seconds...\n")
    
    print("Refinement begins...")
    start_refine = time.time()
    edge_map, disp_bf = measure_dispersion(imgL, pc)
    end_refine = time.time()
    elapse_refine = end_refine - start_refine
    print("Refinement takes " + str(elapse_refine) + " seconds...\n")
    
    disp_psmnet = cv2.imread("../data/prediction/" + filename + ".png", -1)/256.0
    disp_gt = cv2.imread("../data/gt/disp_occ_0/" + filename + ".png", -1)/256.0
    disp_refined = replace_boundary(disp_psmnet, disp_bf)
    
    error1, error_map1 = compute_error(disp_gt, disp_refined)    
    error2, error_map2 = compute_error(disp_gt, disp_psmnet)
    error3, error_map3 = compute_error(disp_gt, disp_lidar)
    print("LiDAR points upsampling..." + str(error3))
    print("before refinement..." + str(error2))
    print("after refinement..." + str(error1))

    f = plt.figure()

    ax1 = f.add_subplot(2,4, 1)
    plt.imshow(error_map3, 'rainbow', vmin=-5, vmax=20)
    plt.axis('off')
    ax1.set_title("Error upsampled", fontsize=10)

    ax2 = f.add_subplot(2,4, 2)
    plt.imshow(error_map2, 'rainbow', vmin=-5, vmax=20)
    plt.axis('off')
    ax2.set_title("Error predicted", fontsize=10)
    
    ax3 = f.add_subplot(2,4, 3)
    plt.imshow(error_map1, 'rainbow', vmin=-5, vmax=20)
    plt.axis('off')
    ax3.set_title("Error refined", fontsize=10)
    
    ax4 = f.add_subplot(2,4, 4)
    plt.imshow(edge_map)
    plt.axis('off')
    ax4.set_title("Edges")

    ax5 = f.add_subplot(2,4, 5)
    plt.imshow(disp_lidar, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax5.set_title("Disparity upsampled", fontsize=10)
    
    ax6 = f.add_subplot(2,4, 6)
    plt.imshow(disp_psmnet, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax6.set_title("Disparity predicted", fontsize=10)
    
    ax7 = f.add_subplot(2,4, 7)
    plt.imshow(disp_refined, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax7.set_title("Disparity refined", fontsize=10)

    ax8 = f.add_subplot(2,4, 8)
    plt.imshow(imgL)
    plt.axis('off')
    ax8.set_title("Image", fontsize=10)

#     plt.show(block=True)
    plt.tight_layout()
    plt.savefig("../output/" + filename + "_compare.png", dpi=1200)

    points, colors = reproject_to_3D(disp_lidar, imgL)
    save_ply("../output/" + filename + "_upsampled.ply", points, colors)
    points, colors = reproject_to_3D(disp_psmnet, imgL)
    save_ply("../output/" + filename + "_predicted.ply", points, colors)
    points, colors = reproject_to_3D(disp_refined, imgL)
    save_ply("../output/" + filename + "_refined.ply", points, colors)
    points, colors = reproject_to_3D(disp_gt, imgL)
    save_ply("../output/" + filename + "_gt.ply", points, colors)