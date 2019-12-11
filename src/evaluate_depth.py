import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import dataLoader, reproject_to_3D, save_ply, compute_error
from lidar_map import measure_dispersion, replace_boundary, bf_vanilla_accelerated
import time
import argparse
import os
import tqdm

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--mode', type=int, default=0, help='specify evaluation mode: 0:one file | 1: whole files')
parser.add_argument('--fn', type=str, default="000035", help='specify file name, skip this if evaluating on whole files')
args = parser.parse_args()

def evaluate_one_file(filename):
    # evaluate on one file pair
    data = dataLoader(filename)
    imgL = data.imgL
    pc = data.pc
    print("Processing data " + filename + "...\n")
    
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
    obj_map = cv2.imread("../data/gt/obj_map/" + filename + ".png", -1)/256.0
    disp_refined = replace_boundary(disp_psmnet, disp_bf)
    
    rtn = []
    error1, error1_fg, error1_bg, error_map1, count1_above_15 = compute_error(disp_gt, disp_refined, obj_map)    
    rtn.append((error1, error1_fg, error1_bg, error_map1, count1_above_15))
    error2, error2_fg, error2_bg, error_map2, count2_above_15 = compute_error(disp_gt, disp_psmnet, obj_map)
    rtn.append((error2, error2_fg, error2_bg, error_map2, count2_above_15))
    error3, error3_fg, error3_bg, error_map3, count3_above_15 = compute_error(disp_gt, disp_lidar, obj_map)
    rtn.append((error3, error3_fg, error3_bg, error_map3, count3_above_15))
    print("All: LiDAR points upsampling... " + str(error3))
    print("All: before refinement... " + str(error2))
    print("All: after refinement... " + str(error1))
    print("FG: LiDAR points upsampling... " + str(error3_fg))
    print("FG: before refinement... " + str(error2_fg))
    print("FG: after refinement... " + str(error1_fg))
    print("BG: LiDAR points upsampling... " + str(error3_bg))
    print("BG: before refinement... " + str(error2_bg))
    print("BG: after refinement... " + str(error1_bg))
    print("BIG ERROR COUNT: LiDAR points upsampling... " + str(count3_above_15))
    print("BIG ERROR COUNT: before refinement... " + str(count2_above_15))
    print("BIG ERROR COUNT: after refinement... " + str(count1_above_15))

    f = plt.figure()

    ax1 = f.add_subplot(4,2, 1)
    plt.imshow(error_map2, 'rainbow', vmin=-5, vmax=20)
    plt.axis('off')
    ax1.set_title("Error predicted: " + str(100* error2)[:4] + "%", fontsize=8)
    
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
    plt.imshow(error_map3, 'rainbow', vmin=-5, vmax=20)
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

    plt.tight_layout()
    plt.savefig("../output/" + "compare_" + filename + ".png", dpi=600)
    plt.close()
    
    points, colors = reproject_to_3D(disp_lidar, imgL)
    save_ply("../output/" + filename + "_upsampled.ply", points, colors)
    points, colors = reproject_to_3D(disp_psmnet, imgL)
    save_ply("../output/" + filename + "_predicted.ply", points, colors)
    points, colors = reproject_to_3D(disp_refined, imgL)
    save_ply("../output/" + filename + "_refined.ply", points, colors)
    points, colors = reproject_to_3D(disp_gt, imgL)
    save_ply("../output/" + filename + "_gt.ply", points, colors)
    
    return rtn

def evaluate_whole_files():
    import glob, os
    paths = glob.glob("../data/image_02/*.png")[20:]
    count = 0
    
    error_refined_all = 0
    error_refined_fg = 0
    error_refined_bg = 0
    error_refined_above_15 = 0
    
    error_predicted_all = 0
    error_predicted_fg = 0
    error_predicted_bg = 0
    error_predicted_above_15 = 0
    
    error_upsampled_all = 0
    error_upsampled_fg = 0
    error_upsampled_bg = 0
    error_upsampled_above_15 = 0
    
    progress_bar = tqdm.tqdm(total=len(paths), leave=True, desc='eval')
    for path in paths:
        fn = os.path.basename(path)[:-4]
        err_pair1, err_pair2, err_pair3 = evaluate_one_file(fn)
        progress_bar.update()
        count += 1
        
        error1_all, error1_fg, error1_bg, _, count1_above_15 = err_pair1
        error2_all, error2_fg, error2_bg, _, count2_above_15 = err_pair2
        error3_all, error3_fg, error3_bg, _, count3_above_15 = err_pair3
        
        error_refined_all += error1_all 
        error_refined_fg += error1_fg
        error_refined_bg += error1_bg
        error_refined_above_15 += count1_above_15
        
        error_predicted_all += error2_all 
        error_predicted_fg += error2_fg
        error_predicted_bg += error2_bg
        error_predicted_above_15 += count2_above_15
    
        error_upsampled_all += error3_all 
        error_upsampled_fg += error3_fg
        error_upsampled_bg += error3_bg
        error_upsampled_above_15 += count3_above_15
        
        if count % 10 == 0:
            print("\n\n")
            print(str(count) + " files processed...")
            
            print("ERROR ALL:")
            print("Upsampled error average: " + str(error_upsampled_all / count))
            print("Predicted error average: " + str(error_predicted_all / count))
            print("Refined   error average: " + str(error_refined_all / count))
            
            print("ERROR FG:")
            print("Upsampled error average: " + str(error_upsampled_fg / count))
            print("Predicted error average: " + str(error_predicted_fg / count))
            print("Refined   error average: " + str(error_refined_fg / count))
            
            print("ERROR BG:")
            print("Upsampled error average: " + str(error_upsampled_bg / count))
            print("Predicted error average: " + str(error_predicted_bg / count))
            print("Refined   error average: " + str(error_refined_bg / count))
            
            print("NUMBER OF PIXELS WITH ERROR ABOVE 15...")
            print("Upsampled error average: " + str(error_upsampled_above_15 / count))
            print("Predicted error average: " + str(error_predicted_above_15 / count))
            print("Refined   error average: " + str(error_refined_above_15 / count))
            print("\n\n")
            
if __name__ == "__main__":
    if not os.path.exists("../output/"):
        os.makedirs("../output/")
    if args.mode == 1:
        print("Evaluating on all files")
        evaluate_whole_files()
        print("finished, please check output folder for results...")
    elif args.mode == 0:
        fn = args.fn
        print("Evaluating on " + fn + "...")
        try:
            evaluate_one_file(fn)
            print("finished, please check output folder for results...")
        except:
            print("file not found, please check data folder, or did you forget leading zeros in filename?")
    else:
        print("mode has to be 0 or 1!")


    