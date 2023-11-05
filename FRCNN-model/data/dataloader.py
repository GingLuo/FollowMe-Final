import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class mydataloader(data.Dataset):

    def __init__(self, txt_path, preproc=None):

        file = open(txt_path, "r")
        img_path = txt_path[:-9] + "images/"
        self.preproc = preproc
        self.imgs_path = []
        self.words = []

        annos = []
        anno = []
        for line in file:
            if line.startswith("#"):
                path = img_path + line[2:-1]
                self.imgs_path.append(path)
                self.words.append(annos)
                annos = []

            else:
                attributes = line.split(" ")
                class_label = float(attributes[0])
                x1 = float(attributes[1])
                y1 = float(attributes[2])
                w = float(attributes[3])
                h = float(attributes[4])
                anno.append(class_label)
                anno.append(x1)
                anno.append(y1)
                anno.append(x1+w)
                anno.append(y1+h)
                anno.extend([1]*10)
                # assert(len(anno) == 15)
                # anno.append(float(attributes[4]))
                # anno.append(float(attributes[5]))
                # anno.append(float(attributes[7]))
                # anno.append(float(attributes[8]))
                # anno.append(float(attributes[10]))
                # anno.append(float(attributes[11]))
                # anno.append(float(attributes[13]))
                # anno.append(float(attributes[14]))
                # anno.append(float(attributes[16]))
                # anno.append(float(attributes[17]))
                # if (attributes[4] == "-1.0"):
                #     anno.append(float(-1))
                # else:
                #     anno.append(float(1))

                annos.append(anno)
                anno = []
        self.words.append(annos)
        self.words = self.words[1:]



    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, annotation in enumerate(labels):
            annotation = np.array(annotation)
            annotation = np.reshape(annotation, (-1, 15))

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """


    targets = []
    imgs = []

    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    if len(batch) > 1 and (isinstance(batch[0], type(np.empty(0))) and isinstance(batch[1], type(np.empty(0)))):
        return (None, None)
    if len(batch) == 1 and isinstance(batch[0], type(np.empty(0))):
        return (None, None)

    # print("Normal")
    # print(len(batch))
    # print(type(batch[0]))
    # print(type(batch[1]))
    # print(batch[0])
    # print(batch[1])
    # print("--------------------------------")
    # try:
    #     torch.stack(imgs, 0)
    # except:
    #     print("dataloader")
    #     print(f"img: {len(imgs)}")
    #     print(f"target: {len(targets)}")
    #     print(f"batch: {len(batch)} {len(batch[0])} {len(batch[1])}")
    #     print(f"batch: {type(batch[0])} {type(batch[0][0])} {len(batch[0][1])}")
    #     print(type(batch))
    #     print(batch)



    # print(batch)
    # print(torch.stack(imgs, 0), targets)
    return (torch.stack(imgs, 0), targets)


# m = mydataloader('/afs/ece.cmu.edu/usr/yichuanl/Private/18794/hw/capstone/my_dataset/label.txt')
# print(m[0])
# print(m[2])