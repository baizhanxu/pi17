# coding:utf-8
# 加入摄像头模块，让小车实现自动循迹行驶
# 思路为:摄像头读取图像，进行二值化，将白色的赛道凸显出来
# 选择下方的一行像素，红色为 0
# 找到0值的中点
# 目标中点与标准中点(320)进行比较得出偏移量
# 根据偏移量来控制小车左右轮的转速
# 考虑了偏移过多失控->停止;偏移量在一定范围内->高速直行(这样会速度不稳定，已删)

# import pandas as pd
# from scipy import linalg
# import tflite_runtime.interpreter as tflite
# import threading  
# import threading  # 导入 threading 库

import cv2
import numpy as np
from new_driver import driver
import time
from threading import Thread
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

STRAIGHT_POWER = 70

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=20):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

car = driver()
# 定义看到crosswalk的次数
crosswalk_num = 0
# 初始化输入网络的图像尺寸
image_size=(28,28)
# 打开摄像头，图像尺寸 640*480(长*高)，opencv 存储值为 480*640(行*列) 
videostream = VideoStream(resolution=(480,640),framerate=10).start()
time.sleep(1)

# upload calibration matrix
# data = np.load('calibration.npz')
# cameraMatrix = data['cameraMatrix']
# distCoeffs = data['distCoeffs']

try:
    while True:

        frame = videostream.read()

#         frame = cv2.undistort(frame, cameraMatrix, distCoeffs, None, cameraMatrix)

        frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_NEAREST) #采样 160*120
        
        # 按q键可以退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        cv2.imshow("original img", frame)
        
        # Processing frame from RGB to HSV
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Set threshold fro binary_processing
        lower_white = np.array([0, 0, 250])
        upper_white = np.array([180, 55, 255])
        
        # binary_processing
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        
        cv2.imshow("after processing", mask)
        
        width = 160
        height = 120
        center = (height // 2, width // 2)

        # 前进状态区间
        move_width_offset = range(-20, 20)
        move_height_offset = range(-30 + center[0], 30 + center[0])

        i = 0
        cnt = 1
        sum = 0

        
        for i in move_height_offset:
            for j in range(width):
                if mask[i, j] == 255:
                    cnt += 1
                    sum += i
        
        mid_of_road = sum // cnt

        dif = mid_of_road - 80
        print(dif)

        print(dif in move_width_offset)

        if dif in move_width_offset:
            print("move ahead")
            car.set_speed(STRAIGHT_POWER, 0, 0)
        else:
            print("stop")
            car.set_speed(0, 0, 0)


finally:
    # 确保在程序退出前停止小车
    print("Stopping the vehicle...")
    car.set_speed(0, 0, 0)
    videostream.stop()
    cv2.destroyAllWindows()


        
