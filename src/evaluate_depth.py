import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import dataLoader, reproject_to_3D, save_ply, compute_error
from lidar_map import measure_dispersion, replace_boundary, bf_vanilla_accelerated
import time

def evaluate_one_file(filename):
    # evaluate on one file pair
    data = dataLoader(filename)
    imgL = data.imgL
    pc = data.pc
    print("Processing data " + filename + "...\n")
    
#     print("Upsampling(accelerated) begins...")
    start_acc = time.time()
    disp_lidar = bf_vanilla_accelerated(imgL, pc)
    end_acc = time.time()
    elapse_acc = end_acc - start_acc
#     print("Upsampling(accelerated) on raw points takes " + str(elapse_acc) + " seconds...\n")
    
#     print("Refinement begins...")
    start_refine = time.time()
    edge_map, disp_bf = measure_dispersion(imgL, pc)
    end_refine = time.time()
    elapse_refine = end_refine - start_refine
#     print("Refinement takes " + str(elapse_refine) + " seconds...\n")
    
    disp_psmnet = cv2.imread("../data/prediction/" + filename + ".png", -1)/256.0
    disp_gt = cv2.imread("../data/gt/disp_occ_0/" + filename + ".png", -1)/256.0
    disp_refined = replace_boundary(disp_psmnet, disp_bf)
    
    error1, error_map1 = compute_error(disp_gt, disp_refined)    
    error2, error_map2 = compute_error(disp_gt, disp_psmnet)
    error3, error_map3 = compute_error(disp_gt, disp_lidar)
#     print("LiDAR points upsampling..." + str(error3))
#     print("before refinement..." + str(error2))
#     print("after refinement..." + str(error1))

    f = plt.figure()

    ax1 = f.add_subplot(4,2, 1)
    plt.imshow(error_map2, 'rainbow', vmin=-5, vmax=20)
    plt.axis('off')
    ax1.set_title("Error predicted: " + str(100* error2)[:4] + "%", fontsize=10)
    
    ax2 = f.add_subplot(4,2, 2)
    plt.imshow(disp_psmnet, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax2.set_title("Disparity predicted", fontsize=8)
    
    ax3 = f.add_subplot(4,2, 3)
    plt.imshow(error_map1, 'rainbow', vmin=-5, vmax=20)
    plt.axis('off')
    ax3.set_title("Error refined:   " + str(100* error1)[:4] + "%", fontsize=8)
    
    ax4 = f.add_subplot(4,2, 4)
    plt.imshow(disp_refined, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax4.set_title("Disparity refined", fontsize=8)
    
    ax5 = f.add_subplot(4,2, 5)
    plt.imshow(error_map3, 'rainbow', vmin=-5, vmax=20, fontsize=8)
    plt.axis('off')
    ax5.set_title("Error upsampled: " + str(100* error3)[:4] + "%", fontsize=8)
    
    ax6 = f.add_subplot(4,2, 6)
    plt.imshow(disp_lidar, 'rainbow', vmin=10, vmax=80)
    plt.axis('off')
    ax6.set_title("Disparity upsampled", fontsize=8)
    
    ax7 = f.add_subplot(4,2, 7)
    plt.imshow(edge_map)
    plt.axis('off')
    ax7.set_title("Edges", fontsize=10)

    ax8 = f.add_subplot(4,2, 8)
    plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    ax8.set_title("Image", fontsize=10)

#     plt.show(block=True)
    plt.tight_layout()
    plt.savefig("../output/" + "compare_" + filename + ".png", dpi=600)

#     points, colors = reproject_to_3D(disp_lidar, imgL)
#     save_ply("../output/" + filename + "_upsampled.ply", points, colors)
#     points, colors = reproject_to_3D(disp_psmnet, imgL)
#     save_ply("../output/" + filename + "_predicted.ply", points, colors)
#     points, colors = reproject_to_3D(disp_refined, imgL)
#     save_ply("../output/" + filename + "_refined.ply", points, colors)
#     points, colors = reproject_to_3D(disp_gt, imgL)
#     save_ply("../output/" + filename + "_gt.ply", points, colors)
    plt.close()
    return error1, error2, error3

def evaluate_whole_files():
    import glob, os
    paths = glob.glob("../data/image_02/*.png")[20:]
    count = 0
    error_refined = 0
    error_predicted = 0
    error_upsampled = 0
    for path in paths:
        fn = os.path.basename(path)[:-4]
        err1, err2, err3 = evaluate_one_file(fn)
        count += 1
        error_refined += err1
        error_predicted += err2
        error_upsampled += err3
        
        if count % 10 == 0:
            print("\n\n")
            print(str(count) + " files processed...")
            print("Upsampled error average: " + str(error_upsampled / count))
            print("Predicted error average: " + str(error_predicted / count))
            print("Refined   error average: " + str(error_refined / count))
            print("\n\n")
            
if __name__ == "__main__":
    """
    Upsampled error average: 0.0622560780720797
    Predicted error average: 0.04163520386390926
    Refined   error average: 0.042493429159431724
    """
    evaluate_whole_files()


    