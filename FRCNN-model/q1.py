from PIL import Image
from PIL import ImageDraw


def IOU(A, B):  # returns None if rectangles don't intersect
    (x1, y1, w1, h1), (x2, y2, w2, h2) = (A, B)
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    # print(dx, dy)
    if (dx >= 0) and (dy >= 0):
        area = dx*dy    
        IOU = area / (w1 * h1 + w2 * h2 - area)
    else:
        IOU = 0 
    return IOU

if __name__ == '__main__':

    # q1
    bbox_count = 0
    # q2
    total_area = 0
    total_h = 0
    total_w = 0
    # q3
    total_x = 0
    total_y = 0
    total_width = 0
    total_height = 0

    # q4
    # If an bounding box exceeds the range of the image, 
    # we should crop the bounding box to fit the borders
    has_exceeded = False
    # q5
    overlapped_count = 0


    file = open("./widerface_homework/train/label.txt", "r")
    path = ""
    image_width = 0
    image_height = 0
    faces = []
    IOU_check = set()
    for line in file:
        if line.startswith("#"):
            path = "./widerface_homework/train/images/" + line[2:-1]
            img = Image.open(path)
            image_width = img.width
            image_height = img.height
            faces = []
            # Aggregate previous image's IOU count
            # # q5
            overlapped_count += len(IOU_check)
            IOU_check = set()
        else:
            # q1
            bbox_count += 1
            # q2
            attributes = line.split(" ")
            x1 = float(attributes[0])
            y1 = float(attributes[1])
            w = float(attributes[2])
            h = float(attributes[3])
            curr_bbox = (x1, y1, w, h)
            curr_index = len(faces)
            total_w += w
            total_h += h
            total_x += x1
            total_y += y1
            total_width += image_width
            total_height += image_height
            total_area += h * w
            # q4
            if (x1 + w > image_width) or (y1 + h > image_height):
                print("GT exceeds image borders", path)
                has_exceeded = True
            # q5
            for i in range(len(faces)):
                try:
                    if IOU(faces[i], curr_bbox) > 0:
                        # print("overlap ", path)
                        IOU_check.add(i)
                        IOU_check.add(curr_index)
                except:
                    print(path)
                    print(faces[i], curr_bbox)
                    # ./widerface_homework/train/images/36--Football/36_Football_americanfootball_ball_36_184.jpg
                    # This image has several bboxs with width=0, which leads to the error
            faces.append(curr_bbox)


    print("q1 - Training Set Total Count: ", bbox_count)
    print("q2 - Average Area: ", total_area/bbox_count)
    avg_w = total_w/bbox_count
    avg_h = total_h/bbox_count
    print("q2 - Average Width: ", avg_w)
    print("q2 - Average Height: ", avg_h)
    print("q3 - Average center is at: ", total_x/bbox_count + avg_w/2, total_y/bbox_count + avg_h/2)
    print("q3 - Real center is at: ", total_width/(bbox_count*2), total_height/(bbox_count*2))
    print("q3 - So the faces are not uniformly distributed on images. If faces were, then the average center should be close to the real average image center")
    print("q3 - The faces are skewed to the upper part of the image, which is reasonable because pictures tend to capture faces in the upper part.")
    if has_exceeded:
        print("q4 - There exist images exceeding the border")
    else:
        print("q4 - No image exceeds the border")
    print("q4 - If an bounding box exceeds the range of the image, we should crop the bounding box to fit the borders")
    print("q5 - Percentage of Overlapped GT: ", overlapped_count/bbox_count)

    # Output:
    # q1 - Training Set Total Count:  159424
    # q2 - Average Area:  3851.222206192292
    # q2 - Average Width:  28.947887394620633
    # q2 - Average Height:  37.3986539040546
    # q3 - Average center is at:  508.9530120935367 304.4233459203131
    # q3 - Real center is at:  2048.0 1536.0214271376958
    # q3 - So the faces are not uniformly distributed on images. If faces were, then the average center should be close to the real average image center
    # q3 - The faces are skewed to the upper part of the image, which is reasonable because pictures tend to capture faces in the upper part.
    # q4 - No image exceeds the border
    # q4 - If an bounding box exceeds the range of the image, we should crop the bounding box to fit the borders
    # q5 - Percentage of Overlapped GT:  0.12396502408671216

    # Validation
    file = open("./widerface_homework/val/label.txt", "r")
    bbox_count = 0
    for line in file:
        if not line.startswith("#"):
            bbox_count += 1
    print("q1 - Validation Set Total Count: ", bbox_count)

    # Output
    # q1 - Validation Set Total Count:  39708


    # q1.2
    file = open("./widerface_homework/train/label.txt", "r")
    count = 0
    img = ""
    for line in file:
        if line.startswith("#"):
            if count > 0:
                img.save(f"./q1images/{count}-test.png")
            if count >= 4:
                break

            path = "./widerface_homework/train/images/" + line[2:-1]
            img = Image.open(path).convert('RGBA')
            count += 1
        else:
            attributes = line.split(" ")
            x1 = float(attributes[0])
            y1 = float(attributes[1])
            w = float(attributes[2])
            h = float(attributes[3])
            curr_x = (x1, x1 + w, x1 + w, x1, x1)
            curr_y = (y1, y1, y1 + h, y1 + h, y1)
            draw = ImageDraw.Draw(img)
            draw.line(list(zip(curr_x,curr_y)), fill="green", width=2) 
            border = w * 0.025 
            for i in range(5):
                if attributes[4+3*i] == "-1.0":
                    continue
                coord = (float(attributes[4+3*i])-border, float(attributes[4+3*i+1])-border,
                        float(attributes[4+3*i])+border, float(attributes[4+3*i+1])+border)
                draw.ellipse(coord, fill = 'red', outline ='red')

