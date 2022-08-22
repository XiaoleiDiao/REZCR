import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import wmi

from nets.RIE import RIEBody
from nets.radical_loss import RadicalLoss
from utils.config import Config
from utils.dataloader import RadicalDataset, radical_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net, R_losses, epoch, epoch_size, epoch_size_val, dataloader, dataloader_val, Epoch):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataloader):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]

            with torch.no_grad():
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            outputs = net(images)

            loss = R_losses(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataloader_val):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]

            optimizer.zero_grad()
            outputs = net(images_val)

            loss = R_losses(outputs, targets_val)
            loss.backward()
            optimizer.step()

            val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    if epoch % 20 == 0:
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
            (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


##### Monitor the temperature of the device to avoid overheating #####
def avg(value_list):
    num = 0
    length = len(value_list)
    for val in value_list:
        num += val
    return num / length


def temp_monitor():
    w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
    sensors = w.Sensor()
    cpu_temps = []
    gpu_temp = 0
    for sensor in sensors:
        if sensor.SensorType == u'Temperature' and not 'GPU' in sensor.Name:
            cpu_temps += [float(sensor.Value)]
        elif sensor.SensorType == u'Temperature' and 'GPU' in sensor.Name:
            gpu_temp = sensor.Value

    # print("Avg CPU: {}".format(avg(cpu_temps)))
    # print("GPU: {}".format(gpu_temp))
    if avg(cpu_temps) > 75 or gpu_temp > 75:
        print("Avg CPU: {}".format(avg(cpu_temps)))
        print("GPU: {}".format(gpu_temp))
        print("sleeping 60s")
        time.sleep(60)
    return


if __name__ == "__main__":

    Cuda = True
    normalize = False

    model = RIEBody(Config)
    model_path = "model_data/pretrained_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()

    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    print('Finished Model Loading')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # set loss
    R_losses = RadicalLoss(np.reshape(Config["RIE"]["anchors"], [-1, 2]), Config["RIE"]["classes"], (Config["img_w"], Config["img_h"]), Cuda, normalize)

    annotation_path = '2007_train.txt'

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    lr = 1e-3
    Batch_size = 8
    Init_Epoch = 0
    Freeze_Epoch = 50

    optimizer = optim.Adam(net.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    train_dataset = RadicalDataset(lines[:num_train], (Config["img_h"], Config["img_w"]), True)
    val_dataset = RadicalDataset(lines[num_train:], (Config["img_h"], Config["img_w"]), False)

    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=radical_dataset_collate)
    dataloader_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=radical_dataset_collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size
    # ------------------------------------#
    # Start training
    # ------------------------------------#
    for param in model.backbone.parameters():
        param.requires_grad = True   # Do not freeze training

    for epoch in range(Init_Epoch, Freeze_Epoch):
        fit_ont_epoch(net, R_losses, epoch, epoch_size, epoch_size_val, dataloader, dataloader_val, Freeze_Epoch)
        lr_scheduler.step()
        temp_monitor()

    # if True:
    #     lr = 1e-4
    #     Batch_size = 4
    #     Freeze_Epoch = 50
    #     Unfreeze_Epoch = 102
    #     optimizer = optim.Adam(net.parameters(),lr)
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
    #
    #     train_dataset = RadicalDataset(lines[:num_train], (Config["img_h"], Config["img_w"]), True)
    #     val_dataset = RadicalDataset(lines[num_train:], (Config["img_h"], Config["img_w"]), False)
    #     gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
    #                                 drop_last=True, collate_fn=radical_dataset_collate)
    #     gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True,
    #                                 drop_last=True, collate_fn=radical_dataset_collate)
    #
    #
    #     epoch_size = num_train//Batch_size
    #     epoch_size_val = num_val//Batch_size

    #     for param in model.backbone.parameters():
    #         param.requires_grad = True
    #
    #     for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
    #         fit_ont_epoch(net,R_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
    #         lr_scheduler.step()
    #         # temp_monitor()
