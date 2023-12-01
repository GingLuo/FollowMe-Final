# get_instruction.py 
# Author: George Chang
# This file will calculate the distance from each object 
# labeled by yolo and output instruction to the speaker. 
# pixel indices inspection tool: https://pixspy.com
import numpy as np 
import cv2 
import os
from collections import defaultdict
import pyrealsense2.pyrealsense2 as rs
from get_frame import *
import pyttsx3
import math
minimum_detection_distance = 0.5 
maximum_detection_distance = 10
selection_width = 20

# @brief Function that would get lidar map 
# @return The return value of function should be 640 x 480 2d array with float value indicating distance.
def receive_lidar_map():
    global depth_scale
    depth_image = None
    try:
        depth_image = get_frame()[1]
        depth = depth_image[320,240].astype(float)*depth_scale
        cv2.imshow('depth', depth_image)
        print(f'Depth: {depth} m')
        #if cv2.waitKey(1) == ord("q"):
            #return [] 
    finally:
        return depth_image

# @brief Function that should give us object labels.
# @return List of list contains top left, bottom right corner indices and label. 
def receive_labels_map(yolo_output):    
    # reference: https://github.com/ultralytics/yolov5/issues/5304
    # Example output:
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    #return [[(300, 300), (400, 400), "Person"]] #example output
    #yolo_output = [] # source of output
    #result = yolo_output.pandas().xyxy[0]
    result = yolo_output
    top_left = list(zip(result['xmin'].values.tolist(), result['ymin'].values.tolist()))
    #print("top_left:", top_left)
    bottom_right = list(zip(result['xmax'].values.tolist(), result['ymax'].values.tolist()))
    #print("bottom_right:",bottom_right)
    names = result['name'].values.tolist()
    preprocess_list = list(zip(top_left, bottom_right, names))
    #print("preprocess list:",preprocess_list)

    #labels = list(map(preprocess_list, list()))
    #print("labels:", labels)
    return preprocess_list
    

# @brief a selective depth calculation algorithm which only consider the small "+" section 
#        to the middle of the picture  
# @return the average width to the closest 0.1m that we predicted. 
def object_depth_measurement_linear(depth_image, label, depth_scale):
    top_left, bottom_right, l = label 
    (minx, miny) = top_left
    (maxx, maxy) = bottom_right
    minx = math.floor(minx)
    miny = math.floor(miny)
    maxx = math.floor(maxx)
    maxy = math.floor(maxy)
    depth_list = defaultdict(int)
    depth_list[0] = 1
    midx, midy = (minx + maxx) // 2, (maxy + miny) // 2
    # consider horizontal line with width selection_width and height being half of the label box
    for y in range(max(midx - selection_width, minx), min(midx + selection_width, maxx)):
        for x in range((miny+midy)//2, (midy + maxy)//2):
            pixel_depth = depth_image[x][y][0]
            if pixel_depth > minimum_detection_distance/depth_scale and pixel_depth < maximum_detection_distance/depth_scale:
                #round to nearest tenth
                depth_list[round(pixel_depth,1)] += 1
    # consider horizontal line with height selection_width and width being half of the label box
    for y in range((minx + midx)//2, (midx + maxx)//2):
        for x in range(max(midy - selection_width, miny), min(midy + selection_width, maxy)):
            pixel_depth = depth_image[x][y][0]
            if pixel_depth > minimum_detection_distance/depth_scale and pixel_depth < maximum_detection_distance/depth_scale:
                #round to nearest tenth
                depth_list[round(pixel_depth,1)] += 1
    return max(depth_list, key=lambda x: depth_list[x]) * depth_scale

# @brief a brute force distance measuring algorithm that considers all pixels in box
# @return the depth of the object from the camera
def object_depth_measurement_square(depth_image, label, depth_scale):
    #print("the label is:", label)
    top_left, bottom_right, l = label 
    #print("at distance function:",top_left, bottom_right, l)
    (minx, miny) = top_left
    minx = math.floor(minx)
    miny = math.floor(miny)
    (maxx, maxy) = bottom_right
    maxx = math.floor(maxx)
    maxy = math.floor(maxy)
    depth_list = defaultdict(int)
    depth_list[0] = 1
    for x in range(miny, maxy):
        for y in range(minx, maxx):
            pixel_depth = depth_image[x][y][0]
            if pixel_depth > minimum_detection_distance/depth_scale and pixel_depth < maximum_detection_distance/depth_scale:
                #round to nearest tenth
                #print(x, y, pixel_depth)
                depth_list[round(pixel_depth,1)] += 1
    #print(depth_list)
    return max(depth_list, key=lambda x: depth_list[x]) * depth_scale

def textToSpeaker(text):
    engine = pyttsx3.init('espeak')
    engine.setProperty('volume', 0.5)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def object_detection_fast(depth_image, label, depth_scale):
    npArray = np.array(depth_image)
    topLeft, bottomRight, l = label
    centerX = (int)(topLeft[1] + bottomRight[1]) // 2
    centerY = (int)(topLeft[0] + bottomRight[0]) // 2
    newArray = np.round(npArray[centerX - selection_width : centerX + selection_width, centerY - selection_width: centerY + selection_width], 4)
    return np.mean(newArray)*depth_scale

# This is done for midterm demo testing.
def testing_Object():
    return
    depth_scale = 0.0002500000118743628
    labels = [[(198, 230), (479, 385), "person"], [(131, 195), (479, 434), "person"], [(50, 148), (479, 493), "person"], [(0, 92), (479, 639), "person"]]
    
    for i in range(4):
        #print("index" + str(i))
        filename = "test/test_depth" + str(depth_scale) + "_image" + str(i+1) + ".txt"
        depth_image = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                column_depth = line.split(' ')
                new = list(map(float, column_depth))
                if len(new) != 640:
                    print("error!" + str(len(new)))
                depth_image.append(new)
        # cv2.imshow('depth', depth_image)
        # #cv2.waitKey(0) 
        label = labels[i]
        result = object_depth_measurement_square(depth_image, label, depth_scale)
        object_label = label[2]
        text = "Hey George, there is a " + object_label + " " + str(round(result,1)) + " meters in front of you"
        print(text)
        
        textToSpeaker(text)
    

if __name__ == "__main__":
    #testing_Object()
    
    engine = pyttsx3.init('espeak')
    engine.say("Hey George How are you doing!")
    engine.runAndWait()
    # setup_camera()
    # try: 
    #     while True:
    #         depth_image = get_frame()[1]
    #         if len(depth_image) == 0: 
    #             print("Depth image is none!")
    #             continue
    #         print(depth_scale)
    #         depth = depth_image[320,240].astype(float)*depth_scale
    #         cv2.imshow('depth', depth_image)
    #         print(f'Depth: {depth} m')
    #         if cv2.waitKey(1) == ord("q"):
    #             break 
            
    #         labels = receive_labels_map()
    #         for elem in labels:
    #           result = object_depth_measurement_square(depth_image, elem)
    #           object_label = elem[2]
    #           text = "There is a " + object_label + " " + str(result) + " in front of you"
    #           textToSpeaker(text)
    # finally:
    #     textToSpeaker("End of service, thanks!")
    #     print("end")      
