import torch
import os

from PIL import ImageDraw

def yolo_model_load():
    # Load YOLOv5 model
    model1 = torch.hub.load("./yolov5", 'custom', path='./yolov5n.pt', source='local')
    # model = torch.hub.load("./yolov5", 'custom', path='./yolo/yolov5n.pt', source='local')
    model = torch.hub.load("./yolov5", 'custom', path='./yolo/best_m1.pt', source='local')
    model1.eval()
    model.eval()

    return model, model1

def visualize():
    model1, model2 = yolo_model_load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model2 = model2.to(device)
    # im = "./hamerschlag_color_image1350.jpg"
    im = "./test.jpg"

    whitelist = set(["person"])
    results1 = model1(im)
    results2 = model2(im)
    # Merge outputs
    results1.pred[0] = torch.cat((results1.pred[0], results2.pred[0]), dim=0)
    res = []
    for i in range(results1.pred[0].size(0)):
        index = int(results1.pred[0][i, 5])
        name = results1.names[index]
        if name not in whitelist:
            res.append(i)
        print(index)
        print(results1.names[index])
    print(results1.pred[0])
    print(res)
    results1.pred[0] = results1.pred[0][res]
    results1.save(save_dir='./visualizations',exist_ok=False)
    # print(results1.names)
    # print(int(results1.pred[0][0, 5]))
    # print(results1.names[int(results1.pred[0][5])])
    input()

    for filename in os.listdir("./testing"):
        im = "./testing/" + filename
        results1 = model1(im)
        results2 = model2(im)
        # Merge outputs
        results1.pred[0] = torch.cat((results1.pred[0], results2.pred[0]), dim=0)

        results1.save(save_dir='./visualizations',exist_ok=True)
        print(results1.pred[0].shape)
    # results2.show()

    # print(results1.pandas().xyxy[0])
    # print(results2.pandas().xyxy[0])

if __name__ == "__main__":
    visualize()
