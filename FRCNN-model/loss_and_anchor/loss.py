import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg
GPU = cfg['gpu_train']

class Loss(nn.Module):

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, landm_data = predictions
        # Ignore landm_data for this

        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, 1:5].data
            labels = targets[idx][:, 0].data
            landms = targets[idx][:, 5:15].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()

        # # landm Loss (Smooth L1)
        # # Ignore
        # # Shape: [batch,num_priors,10]
        # pos1 = conf_t > zeros
        # num_pos_landm = pos1.long().sum(1, keepdim=True)
        # N1 = max(num_pos_landm.data.sum().float(), 1)
        # pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # landm_p = landm_data[pos_idx1].view(-1, 10)
        # landm_t = landm_t[pos_idx1].view(-1, 10)
        # loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != zeros

        # conf_t[pos] = 1
        conf_t = torch.sub(conf_t, 1)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Ignore Hard Negative Mining 
        # # Compute max conf across batch for hard negative mining
        # batch_conf = conf_data.view(-1, self.num_classes)
        # loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # # Hard Negative Mining
        # loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        # loss_c = loss_c.view(num, -1)
        # _, loss_idx = loss_c.sort(1, descending=True)
        # _, idx_rank = loss_idx.sort(1)
        # num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        # print("target weight")
        # print(labels)
        # print(conf_p.shape)
        # print(targets_weighted)
        # input()
        num_pos = pos.long().sum(1, keepdim=True)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        # loss_landm /= N1

        return loss_l, loss_c
