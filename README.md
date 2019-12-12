## Introduction
This repo contains implementation of **refining boundary areas** (where long tails always exist) of depth map predicted by [PSMNET](https://arxiv.org/abs/1803.08669), using the guidance of LiDAR points projection. 3D object detection is also tested on the reprojected pseudo points. We also compare the pros and cons of different approaches of generating high quality depth map. Details can be found here [poster](https://github.com/ziliHarvey/LiDAR-Guided-Boundary-Bleeding-Refinement-for-Depth-Map/blob/master/demo/poster.pdf).  

## Installation
### Dependencies
Conda environment is encouraged. Libraries includes sklearn, tqdm, numpy, skimage, opencv, matplotlib and numba.
### Data preparation
Data is extracted from [KITTI Stereo Benchmark](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), the corresponded velodyne points are found using mapping.txt in the calibration file, predictions are inferenced using PSMNET's provided pretrained model.

### Quick demo
```
git clone https://github.com/ziliHarvey/LiDAR-Guided-Boundary-Bleeding-Refinement-for-Depth-Map.git
cd src
python evaluate_depth.py --mode 0 --fn 000002
```

## Results and Analysis
<img src="https://github.com/ziliHarvey/LiDAR-Guided-Boundary-Bleeding-Refinement-for-Depth-Map/blob/master/demo/result.png" width=100% height=100%>

## Contributions
The idea is slightly modified and implemented following [High-resolution LIDAR-based Depth Mapping using Bilateral Filter](https://arxiv.org/abs/1606.05614) and all experiemntations are conducted using pre-trained models given by [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1812.07179). **However, our method successfully reduced the variance of depth prediction error (especially large error pixels number by 30%) while maintaing the overall mean error, the detection on the refined pseudo points also improved compared with the baseline.**

## Acknowledgement
The source code is released under MIT Licence.  
This research is led by [Zi Li](https://github.com/ziliHarvey) and [Fei Lu](https://github.com/fei123ilike) as the course project of geometry vision at CMU.  
Please cite properly when refer to our contents if helpful.  
Welcome to contact me at zili@andrew.cmu.edu for any question or suggestion.
