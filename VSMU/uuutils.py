import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from medpy.metric import binary


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    dice1(ET):label4
    dice2(TC):label1 + label4
    dice3(WT): label1 + label2 + label4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output,dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3


def cal_hd(output, target):
    '''
    Calculate the 95th percentile Hausdorff Distance (HD95).
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    hd1(ET): label 4
    hd2(TC): label 1 + label 4
    hd3(WT): label 1 + label 2 + label 4
    '''
    # Convert softmax output to discrete predictions
    output = torch.argmax(output, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    # Initialize lists to store HD for each class
    hd1_list, hd2_list, hd3_list = [], [], []

    # Loop over the batch to compute HD for each sample
    for i in range(output.shape[0]):
        # Compute HD for ET (label 4)
        output_et = (output[i] == 3).astype(np.bool_)
        target_et = (target[i] == 3).astype(np.bool_)
        if np.any(output_et) and np.any(target_et):
            hd1 = binary.hd95(output_et, target_et)
        else:
            hd1 = 0
        hd1_list.append(hd1)

        # Compute HD for TC (label 1 or label 4)
        output_tc = ((output[i] == 1) | (output[i] == 3)).astype(np.bool_)
        target_tc = ((target[i] == 1) | (target[i] == 3)).astype(np.bool_)
        if np.any(output_tc) and np.any(target_tc):
            hd2 = binary.hd95(output_tc, target_tc)
        else:
            hd2 = 0
        hd2_list.append(hd2)

        # Compute HD for WT (label 1, label 2, or label 4)
        output_wt = (output[i] > 0).astype(np.bool_)
        target_wt = (target[i] > 0).astype(np.bool_)
        if np.any(output_wt) and np.any(target_wt):
            hd3 = binary.hd95(output_wt, target_wt)
        else:
            hd3 = 0
        hd3_list.append(hd3)

    # Compute mean HD across the batch
    hd1_mean = np.mean(hd1_list)
    hd2_mean = np.mean(hd2_list)
    hd3_mean = np.mean(hd3_list)

    return hd1_mean, hd2_mean, hd3_mean


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        smooth = 0.01

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target,self.n_classes)
        input1 = rearrange(input1,'b n h w s -> b n (h w s)')
        target1 = rearrange(target1,'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试

        target1 = F.interpolate(target1, size=input1.shape[2])
        # print(f"input1 shape: {input1.shape}")
        # print(f"target1 shape: {target1.shape}")
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input,target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


class Loss_2(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss_2, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        smooth = 0.01

        # input1 = F.softmax(input, dim=1)
        # target1 = F.one_hot(target,self.n_classes)
        # input1 = rearrange(input1,'b n h w s -> b n (h w s)')
        # target1 = rearrange(target1,'b h w s n -> b n (h w s)')

        # input1 = input1[:, 1:, :]
        # target1 = target1[:, 1:, :].float()

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试
        # inter = torch.sum(input1 * target1)
        # union = torch.sum(input1) + torch.sum(target1) + smooth
        # dice = 2.0 * inter / union

        loss = F.cross_entropy(input,target, weight=self.weight)

        # total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return loss


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return np.array(hd95)
    else:
        return np.array(50)


class Loss_1(nn.Module):
    def __init__(self, n_classes, weight=None):
        super(Loss_1, self).__init__()
        self.weight = weight.cuda()
        self.n_classes = n_classes

    def forward(self, input, target):
        # 使用 PyTorch 的交叉熵损失函数 F.cross_entropy
        # 注意：不需要手动进行 softmax 操作，F.cross_entropy 会在内部处理
        # 参数 weight 可以用于设置不同类别的权重
        loss = F.cross_entropy(input, target, weight=self.weight)
        return loss



class HD95_Val(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=False):
        super(HD95_Val, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"


    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
        a = inputs.clone().detach().cpu().numpy()
        b = targets.clone().detach().cpu().numpy()
        if np.sum(a) > 0 and np.sum(b) > 0:
            hd95 = binary.hd95(a, b)
        else:
            hd95 = 0

        # inputs = a.clone().detach().numpy()
        # targets = b.clone().detach().numpy()
            # print(type(b), "inputs")
        # print(inputs)
        # if metric_mode:
        #     dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        # else:
        #     dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        # if metric_mode:
        return hd95
        # return 1 - dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(3)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, False))
            dices.append(dice)
        return dices


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    losser = Loss_1(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice(x, y))
