from PIL import Image, ImageDraw
import os

if __name__ == '__main__':
    # image_directory = "archive/train/images/"
    # for file in os.listdir(image_directory): 
    #     if not file.endswith(".jpg"): 
    #         img = Image.open(image_directory+file).convert('RGB')
    #         file_name, file_ext = os.path.splitext(file)
    #         img.save(f'{image_directory}{file_name}.jpg')
    # print("Finished")
    # input()

    with open('label_dataold.txt', 'a') as file:

        # directory = "archive/train/labels"
        # image_directory = "archive/train/images"
        directory = "dataold"
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                continue
            f = os.path.join(directory, filename)
            
            if os.path.isfile(f):
                image_label = open(f, "r")

                path = filename[:-4] + ".jpg"
                real_path = os.path.join(directory, path)
                file.write(f"# {path}\n")
                img = Image.open(real_path).convert('RGBA')
                print(real_path)

                image_width = img.width
                image_height = img.height

                for line in image_label:
                    attributes = line.split(" ")
                    x_center = float(attributes[1]) * image_width
                    y_center = float(attributes[2]) * image_height
                    w = float(attributes[3]) * image_width
                    h = float(attributes[4]) * image_height
                    x1 = x_center - w/2
                    y1 = y_center - h/2
                    class_label = int(attributes[0])+1
                    file.write(f"{class_label} {x1} {y1} {w} {h}\n")

                    curr_x = (x1, x1 + w, x1 + w, x1, x1)
                    curr_y = (y1, y1, y1 + h, y1 + h, y1)
                    draw = ImageDraw.Draw(img)
                    draw.line(list(zip(curr_x,curr_y)), fill="green", width=2) 
                # img.show()
                # input()
                count += 1
        print(f"Finished {count} images in directory {directory}")


