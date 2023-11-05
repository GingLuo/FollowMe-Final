We explored this FRCNN-based model with Resnet50 as backbone and uses FPN, SSH and Anchors.

It is trained from scratch with randomly initialized weights.

The result checkpoints are in ./weights, and example visualizations are in ./test_images_visualization.

We decided to discard this model and use YOLOv5 instead because the model fail to capture ground truth bounding boxes and labels very accurately. Because of limited datapoints (manually collected and labelled in CMU), and limited computing resource and time, the model has limited performance.