import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from BraTS import *
from model_ultra.mamba_seg import VSMU
from uuutils import Loss, cal_dice, cosine_scheduler, HD95_Val, cal_hd
from thop import profile
import SimpleITK as sitk

def cal_params_flops(model, size):
    input = torch.randn(1, 4, size, size, size).cuda()
    flops, params = profile(model, inputs=(input, ))
    print("flops", flops/1e9)
    print('params', params/1e6)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.4fM" % (total/1e6))

def train_loop(model,optimizer,scheduler,criterion,train_loader,device,epoch):
    model.train()
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    hd95_1_train = 0
    hd95_2_train = 0
    hd95_3_train = 0
    hd95_metrics = []
    pbar = tqdm(train_loader)
    for it, (images,masks) in enumerate(pbar):
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it
        param_group = optimizer.param_groups[0]
        param_group['lr'] = scheduler[it]
        # print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device),masks.to(device)
        # [b,4,128,128,128], 4分割
        # outputs,outputs2 = model(images)
        outputs = model(images)
        torch.cuda.empty_cache()
        # outputs = torch.softmax(outputs,dim=1)
        # loss = criterion(outputs, masks)+criterion(outputs2, masks)
        loss = criterion(outputs, masks)
        dice1, dice2, dice3 = cal_dice(outputs, masks)
        hd95_1, hd95_2, hd95_3 = cal_hd(outputs, masks)
        pbar.desc = "loss: {:.3f} ".format(loss.item())

        running_loss += loss.item()
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()
        hd95_1_train += hd95_1.item()
        hd95_2_train += hd95_2.item()
        hd95_3_train += hd95_3.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = running_loss / len(train_loader)
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    hd95_1 = hd95_1_train / len(train_loader)
    hd95_2 = hd95_2_train / len(train_loader)
    hd95_3 = hd95_3_train / len(train_loader)
    # hd95_metrics_1 = list(zip(*hd95_metrics))
    # hd95_metrics_1 = [torch.tensor(hd95, device='cpu').numpy() for hd95 in hd95_metrics_1]
    # hd95_metrics_2 = (np.mean(hd95_metrics_1[0]), np.mean(hd95_metrics_1[1]), np.mean(hd95_metrics_1[2]))
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3,'hd95_1':hd95_1,'hd95_2':hd95_2,'hd95_3':hd95_3}


def val_loop(model, criterion, val_loader, device, save_pth, save_YN):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    hd95_1_val = 0
    hd95_2_val = 0
    hd95_3_val = 0
    count = 0

    pbar = tqdm(val_loader)
    with torch.no_grad():
        for images, masks in pbar:
            count += 1
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)
            hd95_1, hd95_2, hd95_3 = cal_hd(outputs, masks)

            if save_YN:
                outputs = outputs.cpu().detach().numpy()
                out_doc = sitk.GetImageFromArray(outputs)
                out_path = os.path.join(save_pth, f'outputs_epoch{count}', f'out_{count}')
                os.makedirs(out_path, exist_ok=True)
                sitk.WriteImage(out_doc, f'{out_path}.nii')

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            hd95_1_val += hd95_1.item()
            hd95_2_val += hd95_2.item()
            hd95_3_val += hd95_3.item()
        count = 0

            # pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)
    loss = running_loss / len(val_loader)
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    hd95_1 = hd95_1_val / len(val_loader)
    hd95_2 = hd95_2_val / len(val_loader)
    hd95_3 = hd95_3_val / len(val_loader)

    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3,'hd95_1':hd95_1,'hd95_2':hd95_2,'hd95_3':hd95_3}


def train(model,optimizer,scheduler,criterion,train_loader,
          val_loader,epochs,device,train_log,valid_loss_min=999.0):
    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model, optimizer, scheduler, criterion, train_loader, device, e)
        # eval for epoch
        val_metrics = val_loop(model, criterion, val_loader, device, args.save_nii, save_YN=False)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,train_metrics["loss"],val_metrics["loss"])
        info2 = ("Train--ET_dice: {:.3f} TC_dice: {:.3f} WT_dice: {:.3f} ET_hd95: {:.3f} TC_hd95: {:.3f} WT_hd95: {:.3f}"
                 .format(train_metrics['dice1'], train_metrics['dice2'], train_metrics['dice3'], train_metrics['hd95_1'],
                         train_metrics['hd95_2'], train_metrics['hd95_3']))
        info3 = ("Valid--ET_dice: {:.3f} TC_dice: {:.3f} WT_dice: {:.3f} ET_hd95: {:.3f} TC_hd95: {:.3f} WT_hd95: {:.3f}"
                 .format(val_metrics['dice1'],val_metrics['dice2'],val_metrics['dice3'],val_metrics['hd95_1'],
                         val_metrics['hd95_2'],val_metrics['hd95_3']))

        print(info1)
        print(info2)
        print(info3)
        with open(train_log,'a') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, 'results/UNet_seven21.pth')
        else:
            torch.save(save_file,os.path.join(args.save_path,'checkpoint{}.pth'.format(e+1)))
    print("Finished Training!")


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info
    patch_size = (160, 160, 128)
    train_dataset = BraTS(args.data_path, args.train_txt, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    val_dataset = BraTS(args.data_path,args.valid_txt, transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    test_dataset = BraTS(args.data_path,args.test_txt,transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=12,   # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                             pin_memory=True)
    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    model = VSMU(in_chans=4, out_chans=4, depths=[2,2,2,2], feat_size=[48, 96, 192, 384]).to(device)
    cal_params_flops(model, 128)
    criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=1e-4, weight_decay=5e-4)
    scheduler = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs,
                                 niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs, start_warmup_value=5e-4)


    train(model,optimizer,scheduler,criterion,train_loader,val_loader,args.epochs,device,train_log=args.train_log)

    metrics2 = val_loop(model, criterion, val_loader, device, args.save_nii, save_YN=True)
    metrics3 = val_loop(model, criterion, test_loader, device, args.save_nii, save_YN=False)

    print("Valid -- loss: {:.3f} ET_dice: {:.3f} TC_dice: {:.3f} WT_dice: {:.3f} ET_hd95: {:.3f} TC_hd95: {:.3f} WT_hd95: {:.3f}"
          .format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3'], metrics2['hd95_1'],
                  metrics2['hd95_2'], metrics2['hd95_3']))
    print("Test -- loss: {:.3f} ET_dice: {:.3f} TC_dice: {:.3f} WT_dice: {:.3f} ET_hd95: {:.3f} TC_hd95: {:.3f} WT_hd95: {:.3f}"
          .format(metrics3['loss'], metrics3['dice1'], metrics3['dice2'], metrics3['dice3'], metrics3['hd95_1'],
                  metrics3['hd95_2'], metrics3['hd95_3']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--min_lr', type=float, default=0.004)
    parser.add_argument('--data_path', type=str, default=r'/dataset20')
    parser.add_argument('--train_txt', type=str, default=r'/dataset20/train.txt')
    parser.add_argument('--valid_txt', type=str, default=r'/dataset20/valid.txt')
    parser.add_argument('--test_txt', type=str, default=r'/dataset20/test.txt')
    parser.add_argument('--train_log', type=str, default='results/UNet_seven21.txt')
    parser.add_argument('--weights', type=str, default='results/UNet_seven21.pth')
    parser.add_argument('--save_path', type=str, default='checkpoint/UNet_seven21')
    parser.add_argument('--save_nii', type=str, default='/outputs')

    args = parser.parse_args()

    main(args)
