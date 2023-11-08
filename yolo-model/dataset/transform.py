import os
from PIL import Image, ImageDraw, ImageFont


if __name__ == '__main__':
    directory = "labels"
    data = [0] * 10
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            image_label = open(f, "r")
        # with open("./labels/"+filename, 'a') as file:
        #     for line in image_label:
        #         attributes = line.split(" ")
        #         class_label = int(attributes[0])
        #         # [121, 2, 0, 0, 52, 58, 25, 2, 10, 7]
        #         if class_label == 0 or class_label == 1:
        #             new_label = "80"
        #         elif class_label == 4:
        #             new_label = "81"
        #         elif class_label == 5:
        #             new_label = "56"
        #         elif class_label == 6:
        #             new_label == "82"
        #         elif class_label == 7:
        #             new_label == "82"
        #         elif class_label == 8:
        #             new_label = "80"
        #         elif class_label == 9:
        #             new_label = "12"
        #         else:
        #             print(filename)
        #         attributes[0] = new_label
        #         new_line = " ".join(attributes)
        #     file.write(new_line)

        path = filename[:-4] + ".jpg"
        real_path = os.path.join("images", path)
        img = Image.open(real_path).convert('RGB')
        # print(real_path)

        image_width = img.width
        image_height = img.height
        for line in open(f, "r"):
            attributes = line.split(" ")
            class_label = int(attributes[0])
            x_center = float(attributes[1]) * image_width
            y_center = float(attributes[2]) * image_height
            w = float(attributes[3]) * image_width
            h = float(attributes[4]) * image_height
            x1 = x_center - w/2
            y1 = y_center - h/2

            curr_x = (x1, x1 + w, x1 + w, x1, x1)
            curr_y = (y1, y1, y1 + h, y1 + h, y1)
            draw = ImageDraw.Draw(img)
            draw.line(list(zip(curr_x,curr_y)), fill="green", width=2) 
            font = ImageFont.load_default()
            draw.text((x1,y1), f"Class={class_label}", fill="black", anchor="ms", font=font)
        img.save(f"./visualizations/{path}")

    print(data)


