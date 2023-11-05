import numpy as np
from .box_utils import IOU


def nms(dets, thresh):
    # Descending order, indices
    remains = dets[:, 4].argsort()[::-1]
    keep = []
    while len(remains) > 0:
        index = remains[0]
        reference = dets[index][:-1]
        remains = remains[1:]
        keep.append(index)

        iou = np.zeros(remains.shape)

        for j in range(len(remains)):
            victim = remains[j]
            iou[j] = (IOU(reference, dets[victim][:-1]))

        remains = remains[iou < thresh]

    return keep
