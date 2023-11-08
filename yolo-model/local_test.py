import torch
from PIL import ImageDraw

def yolo_model_load():
    # Load YOLOv5 model
    model = torch.hub.load("./yolov5", 'custom', path='./yolov5n.pt', source='local')
    model.eval()

    return model

def visualize():
    model = yolo_model_load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    im = "./dataset/images/color_image8700.jpg"
    results = model(im)
    results.show()

    print(results.pandas().xyxy[0])

if __name__ == "__main__":
    visualize()