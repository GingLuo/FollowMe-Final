# FollowMe - A Blind Aid for Object Detection inHallways
The problem we identified is the challenge of walking for blind people. Even on flat sidewalks or in indoor environments, they have to use walking canes or guide dogs to be very careful of obstacles while recognizing the intersections that lead to the destination. 
To help the blind detect obstacles and navigate the road, we want to develop a model that can perform real-time analysis of the surrounding environment. It can detect obstacles in the hallway and voice a warning based on distance.
The product uses an Intel Realsense sensor with camera and lidar.
Our aim is to promote public welfare of the blind by using the recognition system to replace guide dogs with better eyesight and is cheaper and easier to carry around than dogs.

# How to set up
1. Install yolo5:
    1. cd into yolo-model
    2. run: git clone https://github.com/ultralytics/yolov5.git
    3. Install requirements: cd into yolo-model/yolov5, run pip install -r requirements.txt. It requires Python >= 3.8
2. Intel RealSense
