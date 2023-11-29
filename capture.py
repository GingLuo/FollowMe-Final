import numpy as np
import cv2
IMG_H = 480
IMG_W = 640

# Usage:
# from capture import *
# cap = Capture()
# cap.set_scale(cam.depth_scale)

# when want to capture:
# cap.capture_depth(depth_image)
# will receive a txt file containing the depth


class Capture:
    def __init__(self) -> None:
        self.depth_fd = open('depth2.npy', 'wb')
        self.rgb_fd = cv2.VideoWriter('rgb2.avi', cv2.VideoWriter_fourcc(*'MPEG'), 10, (IMG_W, IMG_H)) 
        self.scale = None
        self.counter_depth = 0
        self.counter_rgb = 0

    def set_scale(scale):
        self.scale = scale

    def capture_depth(depth_frame):
        np.save(self.depth_fd, np.array(depth_scale))
        with open(f"{self.counter_depth}_{self.scale}.txt", "w") as f:
            f.write(str(depth_frame))
        self.counter_depth += 1

    def capture_rgb(rgb_frame):
        cv2.imwrite(+str(self.counter_rgb)+".jpg", rgb_frame)
        self.counter_rgb += 1

    def done(self):
        rgb_fd.release()
        depth_fd.close()