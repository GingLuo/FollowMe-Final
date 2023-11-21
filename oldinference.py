print("Importing Realsense Library...")

from get_frame import get_frame
print("Importing OpenCV...")
import cv2
print("Importing PyTorch...")
import torch
print("Importing Pathlib...")
from pathlib import Path
print("Importing Numpy...")
import numpy as np
from get_frame import get_instruction
import os
import multiprocessing
IMG_H = 480
IMG_W = 640

def detection(org_img, boxs,depth_frame):
    img = org_img.copy()

    for box in boxs:
        cv2.putText(img,str(box[-1])+str(box[4]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

    x, y = 320, 240
    distance = depth_frame.get_distance(x, y)

   #print(f"Distance at pixel ({x}, {y}): {distance} meters")

    return img
    

def yolo_model_load():
    # Load YOLOv5 model
    model = torch.hub.load("/home/followme/Desktop/FollowMe-Final/yolo-model/yolov5", 'custom', path='/home/followme/Desktop/FollowMe-Final/yolo-model/yolov5n.pt', source='local')
    # model = attempt_load('/home/cc/yolov5/weights')
    model.eval()

    return model

def give_instruction(labels, depth_image, depth_scale):
    itemsFound = dict()
    plurality = set()
    for label in labels:
        depth = get_instruction.object_depth_measurement_linear(depth_image, label, depth_scale)
        if label[2] in itemsFound:
            itemsFound[label[2]] = min(itemsFound[label[2]], depth)
            plurality.add(label[2])
        else:
            itemsFound[label[2]] = depth
            	        
    for item in itemsFound:
        distance = itemsFound[item]
        if distance != 0:
        ##sif item in plurality:
            text = item + " " + str(round(distance,1)) + "meters ahead"
            p = multiprocessing.Process(target=get_instruction.textToSpeaker, args=(text,))
            p.start()
            p.join()
            print(text)


# Set up camera
# cap = cv2.VideoCapture(0)  # Use the default camera (change to a different camera index if needed)
# ret, frame = cap.read()

def runner_realsense():

    
    print("Loading YOLO Model...")
    model = yolo_model_load()
    print("Is cuda available:")
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    cam = get_frame.Camera()
    count = 0
    p = None
    while True:
        
        color_image, depth_frame, depth_image = cam.get_pic()
        depth_scale = cam.depth_scale
        depth_image = depth_image.reshape((depth_image.shape[0],depth_image.shape[1],1))

        results = model(color_image) 
        boxs = results.pandas().xyxy[0].values

        color_img = detection(color_image, boxs, depth_frame)
        labels = get_instruction.receive_labels_map(results)
        count += 1
        if count == 10:
            if p != None:
                p.join()
            p = multiprocessing.Process(target=give_instruction, args=(labels, depth_image, depth_scale))
            p.start()
            #p.join()
            count = 0	
        else:
            height, width, channels = color_img.shape

            combined_image = np.zeros((height, 2 * width, channels), dtype=np.uint8)

            combined_image[:, :width, :] =color_img

            # Copy the second image to the right half of the combined image
            combined_image[:, width:, :] = depth_image

           # cv2.imshow('img', combined_image)
 
        # cv2.imshow('img', depth_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    # Release the camera and close the OpenCV window
    # cap.release()
    cv2.destroyAllWindows()

runner_realsense()
