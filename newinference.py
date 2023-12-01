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
import collections
import time
import logging
import pandas as pd
import os
import pyttsx3
import threading
IMG_H = 480
IMG_W = 640

plurality = collections.defaultdict(int)
logging.basicConfig(filename="latencyLog5.log", level=logging.INFO)
whiteList = set(["person", "door", "desk", "bicycle", "bench", "backpack", "umbrella", "suitcase", "chair", "couch", "dining table", "tv", "laptop", "sink"])

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
    model = torch.hub.load("/home/followme/Desktop/FollowMe-Final/yolo-model/yolov5", 'custom', path='/home/followme/Desktop/FollowMe-Final/yolo-model/best_m1.pt', source='local')
    # model = attempt_load('/home/cc/yolov5/weights')
    model2 = torch.hub.load("/home/followme/Desktop/FollowMe-Final/yolo-model/yolov5", 'custom', path='/home/followme/Desktop/FollowMe-Final/yolo-model/yolov5n.pt', source='local')
    model.eval()
    model2.eval()

    return model, model2

def give_instruction(labels, depth_image, depth_scale):
    startTime = time.time()
    itemsFound = dict()
    for label in labels:
        print("label is:", label)
        if label[2] not in whiteList:
            continue
        minx, miny = label[0]
        maxx, maxy = label[1]
        avey, avex = (miny + maxy) // 2, (minx + maxx) // 2
        print("ave x, y is", str(avex), " and ", str(avey))
        direction = "ahead"
        if avex > 520:
           direction = "on your right"
        if avex < 120:
            direction = "on your left"
        if avex > 590  or avex < 50:
            print("the label is on side so omited")
            continue 
        timea = time.time()

        depth = get_instruction.object_detection_fast(depth_image, label, depth_scale)
        timeb = time.time()
        logging.info("Latency for depth calculation: "+ str(timeb-timea))
        print("the depth we received is: ", str(depth))
        if plurality[label[2]] == 0:
            itemsFound[label[2]] = depth
            plurality[label[2]] += 1
        else:
            itemsFound[label[2]] = min(itemsFound[label[2]], depth)
            plurality[label[2]] += 1
    textSpoken = []
    newItemsFound = dict()
    for item in itemsFound:
        distance = itemsFound[item]
        if distance != 0 and plurality[item] > 0:
            plurality[item] = 0
            text = item + " " + str(round(distance,1)) + "meters" + direction
            textSpoken.append(text)
            #p = multiprocessing.Process(target=get_instruction.textToSpeaker, args=(text,))
            #p.start()
        else:
            newItemsFound[item] = itemsFound[item]
    t = ' '.join(textSpoken)
    time1 = time.time()
    get_instruction.textToSpeaker(t)
    itemsFound = newItemsFound
    finalTime = time.time()
    logging.info("Latency for speaker: " + str(finalTime - time1))
    logging.info("Latency for instruction process: " + str(finalTime - startTime))

# Set up camera
# cap = cv2.VideoCapture(0)  # Use the default camera (change to a different camera index if needed)
# ret, frame = cap.read()

def runner_realsense():

    
    print("Loading YOLO Model...")
    model, model2 = yolo_model_load()
    print("Is cuda available:")
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model2 = model2.to(device)
    cam = get_frame.Camera()
    count = 0
    depth_count = 1
    startAnnounce = "Welcome, System is ready"
    p = multiprocessing.Process(target=get_instruction.textToSpeaker,args=(startAnnounce, ))
    p.start()
    while True:
        startTime = time.time()
        color_image, depth_frame, depth_image = cam.get_pic()
        depth_scale = cam.depth_scale
        depth_image = depth_image.reshape((depth_image.shape[0],depth_image.shape[1],1))
        currTime = time.time()
        logging.info("Latency for Camera Processing: " + str(currTime - startTime))
        results1 = model(color_image)
        currTime2 = time.time()
        logging.info("Latency for first model: " + str(currTime2 - currTime))
        results2 = model2(color_image)
        currTime3 = time.time()
        logging.info("Latency for second model: " + str(currTime3 - currTime2))
        results = pd.concat([results1.pandas().xyxy[0], results2.pandas().xyxy[0]], ignore_index=True)
        
        boxs = results.values
        
        #color_img = detection(color_image, boxs, depth_frame)
        labels = get_instruction.receive_labels_map(results)
        currTime4 = time.time()
        logging.info("Latency for label processing: " + str(currTime4 - currTime3))
        count += 1
        if count == 5:
            if p != None:
               p.join()
            # Show images
            results1.pred[0] = torch.cat((results1.pred[0], results2.pred[0]), dim=0)
            res = []
            for i in range(results1.pred[0].size(0)):
                index = int(results1.pred[0][i, 5])
                name = results1.names[index]
                if name not in whiteList:
                    res.append(i)
            results1.pred[0] = results1.pred[0][res]
            results1.save(save_dir='./yolo_images/image', exist_ok=False)
            #if depth_count <= 1:
                #with open(f"yolo_images/image/depth_{depth_scale}.txt", "w+") as f:
                    #f.write(str(depth_image))
            #else:
                #with open(f"yolo_images/image{depth_count}/depth_{depth_scale}.txt", "w+") as f:
                    #f.write(str(depth_image))
            currTime5 = time.time()
            p = multiprocessing.Process(target=give_instruction, args=(labels, depth_image, depth_scale))
            p.start()
            currTime6 = time.time()
            print("Latency for instruction thread start: ", str(currTime6 - currTime5))
            count = 0
            depth_count += 1
        else:
            #height, width, channels = color_img.shape

            #combined_image = np.zeros((height, 2 * width, channels), dtype=np.uint8)

            #combined_image[:, :width, :] =color_img

            # Copy the second image to the right half of the combined image
            #combined_image[:, width:, :] = depth_image

           # cv2.imshow('img', combined_image)
 
        # cv2.imshow('img', depth_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        lastTime = time.time()
        if count % 5 == 0:
            logging.info('Latency for hard process: '+ str(lastTime - startTime))
        else:
            logging.info('Latency for easy process: ' + str(lastTime - startTime))
    # Release the camera and close the OpenCV window
    # cap.release()
    cv2.destroyAllWindows()

runner_realsense()
