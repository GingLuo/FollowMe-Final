import cv2
import torch
from pathlib import Path
import pyrealsense2 as rs

import numpy as np

# from yolov5.models.experimental import attempt_load

IMG_H = 480
IMG_W = 640

def detection(org_img, boxs,depth_frame):
    img = org_img.copy()

    for box in boxs:
        cv2.putText(img,str(box[-1])+str(box[4]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

    x, y = 320, 240
    distance = depth_frame.get_distance(x, y)

    print(f"Distance at pixel ({x}, {y}): {distance} meters")

    return img
    

def realsense_setup():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    pipeline.start(config)

    return pipeline

def yolo_model_load():
    # Load YOLOv5 model
    model = torch.hub.load("./yolov5", 'custom', path='./best_m1.pt', source='local')
    # model = attempt_load('/home/cc/yolov5/weights')
    model.eval()

    return model


# Set up camera
# cap = cv2.VideoCapture(0)  # Use the default camera (change to a different camera index if needed)
# ret, frame = cap.read()

def runner_realsense():

    pipeline = realsense_setup()
    model = yolo_model_load()
    print("Is cuda avaiable:")
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())

        depth_image = depth_image.reshape((depth_image.shape[0],depth_image.shape[1],1))

        color_image = np.asanyarray(color_frame.get_data())
        # img0 = frame
        # img = letterbox(img0, new_shape=640)[0]

        # # Convert BGR image to RGB
        # color_image = np.asanyarray(img0)
        results = model(color_image)
        boxs = results.pandas().xyxy[0].values

        color_img = detection(color_image, boxs,depth_frame)

        height, width, channels = color_img.shape

        combined_image = np.zeros((height, 2 * width, channels), dtype=np.uint8)

        combined_image[:, :width, :] =color_img

        # Copy the second image to the right half of the combined image
        combined_image[:, width:, :] = depth_image

        cv2.imshow('img', combined_image)

        # cv2.imshow('img', depth_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV window
    # cap.release()
    cv2.destroyAllWindows()

runner_realsense()
