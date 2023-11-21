import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
class Camera:
    def __init__(self) -> None:
        self.counter = 0
        self.width = 640
        self.height = 480

        self.pc = rs.pointcloud()
        self.points = rs.points()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
 
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.visual_preset, 2)
        print("Depth Scale is: ", self.depth_scale)
        self.depth_sensor.set_option(rs.option.laser_power, 100)
        self.laser_pwr = self.depth_sensor.get_option(rs.option.laser_power)
        print("laser power = ", self.laser_pwr)

        self.laser_range = self.depth_sensor.get_option_range(rs.option.laser_power)
        print("laser power range = " , self.laser_range.min , "~", self.laser_range.max)
        

    def get_pic(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        #print("The size of depth_image is:", depth_image.shape)
        color_image = np.asanyarray(color_frame.get_data())
        
        return (color_image, depth_frame, depth_image) 

if __name__ == "__main__":
    cam = Camera()
    start = True
    try:
        while True:
            if start:
                time.sleep(5)
                start = False
            path1, path2, arr3 = cam.get_pic()
            rgb = cv2.imread(path1, cv2.IMREAD_COLOR)
            depthcolor = cv2.imread(path2, cv2.IMREAD_COLOR)
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depthcolor)
            if cv2.waitKey(1) == ord("q"):
                break  
    except KeyboardInterrupt:
        cam.pipeline.stop()
        exit()


