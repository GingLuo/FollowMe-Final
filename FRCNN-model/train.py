import os
import time
import math
import torch
import argparse
import datetime
import matplotlib.pyplot as plt
import datetime


from data import mydataloader, collate, preproc, cfg
from loss_and_anchor import Loss, anchor
from detector.mydetector import mydetector

parser = argparse.ArgumentParser(description='18794 detection')
parser.add_argument('--data_path', default='/afs/ece.cmu.edu/usr/yichuanl/Private/18794/hw/FollowMe-Final/FRCNN-model/dataset/label.txt')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--save_path', default='./weights/')

args = parser.parse_args()


def train():
    net = mydetector(cfg=cfg)
    if cfg['ngpu'] > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    net.train()
    torch.backends.cudnn.benchmark = True



    epoch = 0
    dataset = mydataloader(args.data_path, preproc(img_dim=cfg['image_size'],
                                                         rgb_means=(104, 117, 123)))

    traindata = torch.utils.data.DataLoader(dataset,
                                            cfg['batch_size'],
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate)

    epoch_size = math.ceil(len(dataset) / cfg['batch_size'])
    max_iter = cfg['epoch'] * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = Loss(10, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = anchor(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    with torch.no_grad():
        priors = priorbox.forward().cuda()

    loss_total = []
    loss_localization = []
    loss_classification = []
    loss_landmark = []

    for iteration in range(0, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(traindata)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        if (images == None):
            # print("Meet irregular data")
            continue
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)

        loss = cfg['loc_weight'] * loss_l + loss_c

        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if iteration % 100 == 0:
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                .format(epoch, cfg['epoch'], (iteration % epoch_size) + 1,
                epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(),
                        0, lr, batch_time, str(datetime.timedelta(seconds=eta))))

        if iteration % 20 == 0:
            loss_total.append(loss)
            loss_localization.append(loss_l)
            loss_classification.append(loss_c)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    torch.save(net.state_dict(), args.save_path + cfg['name'] + '_Final.pth')

    return loss_total, loss_localization, loss_classification, loss_landmark


def adjust_learning_rate(optimizer, epoch, step_index, iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = args.lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    
    loss_total, loss_localization, loss_classification, loss_landmark = train()
    print("All is over")
    # input()

    # Plot loss curves
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    owner = "yichuanl"
    plt.figure(figsize=(6, 4))
    plt.plot(loss_total, label='Total Loss')
    plt.plot(loss_localization, label='Localization Loss')
    plt.plot(loss_classification, label='Classification Loss')
    plt.plot(loss_landmark, label='Landmark Loss')
    plt.xlabel('Per 20 Iterations')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves\n {owner}: {timestamp_str}', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("./result_curves/test-loss-curve-not-20.png")


    # x = [20 * i for i in range(len(loss_total))]
    # plt.figure(figsize=(6, 4))
    # plt.plot(x, loss_total, label='Total Loss')
    # plt.plot(x, loss_localization, label='Localization Loss')
    # plt.plot(x, loss_classification, label='Classification Loss')
    # plt.plot(x, loss_landmark, label='Landmark Loss')
    # plt.xlabel('Num of Iterations')
    # plt.ylabel('Loss')
    # plt.title(f'Loss Curves\n {owner}: {timestamp_str}', fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.savefig("./result_curves/test-loss-curve.png")

    # x = [20 * i for i in range(len(loss_total))]
    # plt.figure(figsize=(16, 8))
    # plt.plot(x, loss_total, label='Total Loss')
    # plt.plot(x, loss_localization, label='Localization Loss')
    # plt.plot(x, loss_classification, label='Classification Loss')
    # plt.plot(x, loss_landmark, label='Landmark Loss')
    # plt.xlabel('Num of Iterations')
    # plt.ylabel('Loss')
    # plt.title(f'Loss Curves\n {owner}: {timestamp_str}', fontsize=16)
    # plt.legend()
    # plt.setp(plt.legend().get_lines(), linewidth=2)
    # plt.setp(plt.legend().get_texts(), fontsize='x-large')
    # plt.grid()
    # plt.savefig("./result_curves/test-loss-curve-wide.png")


# Loss output
# Epoch:1/1 || Epochiter: 4801/6440 || Iter: 4801/6440 || Loc: 0.9353 Cla: 1.5471 Landm: 2.1233 || LR: 0.00100000 || Batchtime: 0.6921 s || ETA: 0:18:55
# Epoch:1/1 || Epochiter: 5101/6440 || Iter: 5101/6440 || Loc: 3.4078 Cla: 3.8303 Landm: 18.6551 || LR: 0.00100000 || Batchtime: 0.6789 s || ETA: 0:15:09
# Epoch:1/1 || Epochiter: 5401/6440 || Iter: 5401/6440 || Loc: 0.6965 Cla: 1.4381 Landm: 1.0676 || LR: 0.00100000 || Batchtime: 0.6735 s || ETA: 0:11:40
# Epoch:1/1 || Epochiter: 5701/6440 || Iter: 5701/6440 || Loc: 0.7189 Cla: 1.0520 Landm: 1.5581 || LR: 0.00100000 || Batchtime: 0.6454 s || ETA: 0:07:57
# Epoch:1/1 || Epochiter: 6001/6440 || Iter: 6001/6440 || Loc: 2.2662 Cla: 2.8337 Landm: 7.2278 || LR: 0.00100000 || Batchtime: 0.8076 s || ETA: 0:05:55
# Epoch:1/1 || Epochiter: 6301/6440 || Iter: 6301/6440 || Loc: 2.2961 Cla: 3.4013 Landm: 3.2449 || LR: 0.00100000 || Batchtime: 0.6540 s || ETA: 0:01:31