from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import cv2

from data import cfg
from loss_and_anchor import anchor
from detector.mydetector import mydetector
from utils.nms import nms
from utils.box_utils import decode, decode_landm
from utils.misc import load_model, Timer

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description='18794 detection')
parser.add_argument('--ckpt', default='./weights/Resnet50_Final.pth',
                    type=str, help='trained checkpoint')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='/afs/ece.cmu.edu/usr/yichuanl/Private/18794/hw/FollowMe-Final/FRCNN-model/dataset/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.65, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.1, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # net and model
    net = mydetector(cfg=cfg, phase='test')
    net = load_model(net, args.ckpt, args.cpu)
    net.eval()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "label.txt"

    test_dataset = open(testset_list, "r")

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, line in enumerate(test_dataset):
        if i >= 10:
            break
        if not line.startswith("#"):
            continue
        img_name = line[2:-1]
        image_path = testset_folder + img_name
        print(image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1200
        max_size = 1600
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = anchor(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        predicted_conf, predicted_class = torch.max(conf.squeeze(0).data.cpu(), 1)
        scores = predicted_conf.numpy()
        predicted_class = predicted_class.numpy()

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]]).to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        predicted_class = predicted_class[inds]


        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        predicted_class = predicted_class[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        predicted_class = predicted_class[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        predicted_class = predicted_class[:args.keep_top_k]

        # dets = np.concatenate((dets, predicted_class), axis=1)
        
        print(dets[:, -1])
        print(predicted_class)
        # input()

        img = Image.open(image_path).convert('RGBA')
        draw = ImageDraw.Draw(img)
        for index in range(len(keep)):
            conf = scores[keep[index]]
            attributes = dets[index]
            x1 = float(attributes[0])
            y1 = float(attributes[1])
            w = float(attributes[2]) - x1
            h = float(attributes[3]) - y1
            curr_x = (x1, x1 + w, x1 + w, x1, x1)
            curr_y = (y1, y1, y1 + h, y1 + h, y1)
            draw.line(list(zip(curr_x,curr_y)), fill="green", width=2)
            font = ImageFont.load_default() 
            draw.text((x1+10, y1+10), f"Conf = {conf}", fill="black", anchor="ms", font=font)
        img.save(f"./test_images/q8-{i}-test.png")

        _t['misc'].toc()



