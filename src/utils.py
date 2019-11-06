import numpy as np
import cv2 as cv
from config import path_image_02, path_image_03, path_velodyne_points

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
            img = cv.imread(file_path, -1)
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

if __name__ == "__main__":
    data = dataLoader("0000000000")
    cv.imshow("stereo_left", data.imgL)
    cv.waitKey(0)
    print(data.pc)