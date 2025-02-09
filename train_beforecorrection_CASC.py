import argparse
import os, sys
import pandas as pd

sys.path.append("..")
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.nn as nn

import imgaug.augmenters as iaa
import kornia

from torchvision import transforms

from PIL import Image, ImageOps

import os.path as osp

from matplotlib import cm

from MOTSDataset_2D_Patch_supervise_csv_512_Cell import MOTSDataSet as MOTSDataSet
from MOTSDataset_2D_Patch_supervise_csv_512_Cell import MOTSValDataSet as MOTSValDataSet

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util_a.image_pool import ImagePool
from unet2D_Dodnet_scale_token_corrective import UNet2D as UNet2D_scale


def one_hot_3D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


confident_flag = 1
confident_e = 1
simattention_flag = 1
simattention_e = 1
simloss_flag = 1
simloss_e = 1


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")
    # parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_trainingset_patch/data_list.csv')
    parser.add_argument("--trainset_dir", type=str, default='/Data4/CASC/dataset_patch/Glo_Cell_beforecorrection/Train/data_list.csv')

    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/data_list.csv')
    parser.add_argument("--valset_dir", type=str, default='/Data4/CASC/dataset_patch/Glo_Cell_beforecorrection/Val/data_list.csv')


    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.0)

    parser.add_argument("--scale", type=str2bool, default=False)
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/PrPSeg_beforecorrection_CASC_e/')
    parser.add_argument("--reload_path", type=str, default='')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=101)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser




def count_score(preds, labels, rmin, rmax, cmin, cmax, simscore):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin:rmax,cmin:cmax]
        label = labels[ki,:,rmin:rmax,cmin:cmax]
        weight = simscore[ki,rmin:rmax,cmin:cmax].unsqueeze(0)
        Val_DICE += dice_score(pred, label, weight)

        preds1 = pred[1, ...].flatten().detach().cpu().numpy()
        labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

        weight1 = weight.detach().flatten().detach().cpu().numpy()
        cnf_matrix = confusion_matrix(preds1, labels1, sample_weight = weight1)

        try:
            FP = cnf_matrix[1, 0]
            FN = cnf_matrix[0, 1]
            TP = cnf_matrix[1, 1]
            TN = cnf_matrix[0, 0]

        except:
            FP = np.array(1)
            FN = np.array(1)
            TP = np.array(1)
            TN = np.array(1)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        Val_TPR += TP / (TP + FN)
        Val_PPV += TP / (TP + FP)

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, Val_TPR/cnt, Val_PPV/cnt



def dice_score(preds, labels, weights):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    weight = weights.contiguous().view(1, -1)

    num = torch.sum(torch.mul(predict, target) * weight, dim=1)
    den = torch.sum(predict * weight, dim=1) + torch.sum(target * weight, dim=1) + 1

    dice = (2 * num / den)

    return dice.mean()


def get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE):

    term_seg_Dice = 0
    term_seg_BCE = 0
    term_all = 0

    term_seg_Dice += loss_seg_DICE.forward(preds, labels, weight)
    term_seg_BCE += loss_seg_CE.forward(preds, labels, weight)
    term_all += (term_seg_Dice + term_seg_BCE)

    return term_seg_Dice, term_seg_BCE, term_all


def supervise_learning_correct(images, labels, batch_size, scales, model, now_task, weight,
                                                   loss_seg_DICE, loss_seg_CE, confident_flag,
                                                   confident_e, simattention_flag, simattention_e, simloss_flag,
                                                   simloss_e, loss_KL, loss_MSE):
    a = labels.clone().detach()
    preds, simscore, simscore_noise, conscore, top_feature, noise_feature = model(images, torch.ones(batch_size).cuda() * now_task, scales, labels)
    labels = one_hot_3D(labels.long())

    term_seg_Dice = 0
    term_seg_BCE = 0
    term_seg_all = 0

    #term_seg_Dice, term_seg_BCE, term_all = get_loss(preds, labels, 1.5 ** simscore, loss_seg_DICE, loss_seg_CE)

    cnt = 0
    if confident_flag > 0:
        if confident_e:
            #ori weight = conscore * a  #only increase TP
            now_weight = 1.5 ** ((conscore * a) + (1 - conscore) * (1 - a)) # increase TP and TN
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(images, preds, labels, now_weight, loss_seg_DICE, loss_seg_CE)
        else:
            print('a')
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(images, preds, labels, conscore * a, loss_seg_DICE, loss_seg_CE)

        term_seg_Dice += confident_flag * now_term_seg_Dice
        term_seg_BCE += confident_flag * now_term_seg_BCE
        term_seg_all += confident_flag * now_term_seg_all
        cnt += confident_flag

    if simattention_flag > 0:
        if simattention_e:
            # ori weight = simscore * a # only increase TP
            now_weight = 1.5 ** (simscore - simscore_noise) # increase TP and TN
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(images, preds, labels, now_weight, loss_seg_DICE, loss_seg_CE)
        else:
            print('a')
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(images, preds, labels, simscore * a, loss_seg_DICE, loss_seg_CE)

        term_seg_Dice += simattention_flag * now_term_seg_Dice
        term_seg_BCE += simattention_flag * now_term_seg_BCE
        term_seg_all += simattention_flag * now_term_seg_all
        cnt += simattention_flag


    if simloss_flag > 0:
        features_map1 = top_feature.cuda()
        features_map2 = noise_feature.cuda()

        norm_features_map1 = (features_map1 - torch.min(features_map1, dim=1, keepdim=True)[0] * torch.ones((features_map1.shape)).cuda()) / ((torch.max(features_map1, dim=1, keepdim=True)[0]-torch.min(features_map1, dim=1, keepdim=True)[0]) * torch.ones((features_map1.shape)).cuda())
        norm_features_map2 = (features_map2 - torch.min(features_map2, dim=1, keepdim=True)[0] * torch.ones((features_map2.shape)).cuda()) / ((torch.max(features_map2, dim=1, keepdim=True)[0]-torch.min(features_map2, dim=1, keepdim=True)[0]) * torch.ones((features_map2.shape)).cuda())

        term_KL = loss_KL(norm_features_map1, norm_features_map2)
        term_MSE = loss_MSE(features_map1 + 0.001, features_map2 + 0.001)
        term_all_semi = term_KL + term_MSE

    return term_seg_Dice / cnt, term_seg_BCE / cnt, term_seg_all / cnt - term_all_semi


def supervise_learning(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE):

    preds = model(images, torch.ones(batch_size).cuda() * now_task, scales)
    # print(now_task, scales)

    labels = one_hot_3D(labels.long())

    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE)

    return term_seg_Dice, term_seg_BCE, term_all


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        task_num = 4
        # Create model
        criterion = None
        model = UNet2D_scale(num_classes=task_num, weight_std=False)
        check_wo_gpu = 0

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not check_wo_gpu:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)                     #only for multi-head
            loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=args.num_classes, ignore_index=255).to(device)
            loss_KL = nn.KLDivLoss().to(device)
            loss_MSE = nn.MSELoss().to(device)


        else:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
            loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=args.num_classes, ignore_index=255)
            loss_KL = nn.KLDivLoss()
            loss_MSE = nn.MSELoss()


        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        trainloader = DataLoader(
            MOTSDataSet(args.trainset_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                        edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=num_worker)

        valloader = DataLoader(
            MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=4,shuffle=False,num_workers=num_worker)


        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999

        layer_num = [0]

        for epoch in range(0,args.num_epochs):
            model.train()

            # Dynamically create pools and task-specific variables
            for i in range(task_num):
                globals()[f'task{i}_pool_image'] = ImagePool(8)
                globals()[f'task{i}_pool_mask'] = ImagePool(8)
                globals()[f'task{i}_pool_weight'] = ImagePool(8)
                globals()[f'task{i}_scale'] = []
                globals()[f'task{i}_layer'] = []

            if epoch < args.start_epoch:
                continue

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []

            batch_size = args.batch_size
            each_loss = torch.zeros((task_num)).cuda()
            count_batch = torch.zeros((task_num)).cuda()
            loss_weight = torch.ones((task_num)).cuda()


            for iter, batch in enumerate(trainloader):

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3]
                l_ids = batch[4].cuda()
                t_ids = batch[5].cuda()
                s_ids = batch[6].cuda()

                sum_loss = 0

                for ki in range(len(imgs)):
                    now_task = layer_num[l_ids[ki]] + t_ids[ki]

                    # Dynamically access the corresponding pools and lists using `globals()`
                    globals()[f'task{now_task}_pool_image'].add(imgs[ki].unsqueeze(0))
                    globals()[f'task{now_task}_pool_mask'].add(lbls[ki].unsqueeze(0))
                    globals()[f'task{now_task}_pool_weight'].add(wt[ki].unsqueeze(0))
                    globals()[f'task{now_task}_scale'].append(s_ids[ki])
                    globals()[f'task{now_task}_layer'].append(l_ids[ki])

                for now_task in range(task_num):  # Loop through tasks 0 to 22
                    task_pool_image = globals()[f'task{now_task}_pool_image']
                    task_pool_mask = globals()[f'task{now_task}_pool_mask']
                    task_pool_weight = globals()[f'task{now_task}_pool_weight']
                    task_scale = globals()[f'task{now_task}_scale']
                    task_layer = globals()[f'task{now_task}_layer']

                    if task_pool_image.num_imgs >= batch_size:
                        images = task_pool_image.query(batch_size)
                        labels = task_pool_mask.query(batch_size)
                        wts = task_pool_weight.query(batch_size)

                        scales = torch.ones(batch_size).cuda()
                        layers = torch.ones(batch_size).cuda()

                        for bi in range(len(scales)):
                            scales[bi] = task_scale.pop(0)
                            layers[bi] = task_layer.pop(0)

                        weight = edge_weight ** wts

                        # Call supervise_learning function
                        # term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(
                        #     images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE,
                        #     loss_seg_CE)
                        'corrective learning'
                        term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning_correct(images, labels, batch_size, scales, model, now_task, weight,
                                                   loss_seg_DICE, loss_seg_CE, confident_flag,
                                                   confident_e, simattention_flag, simattention_e, simloss_flag,
                                                   simloss_e, loss_KL, loss_MSE)

                        reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                        reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                        reduce_all = engine.all_reduce_tensor(Sup_term_all)

                        optimizer.zero_grad()
                        reduce_all.backward()

                        optimizer.step()

                        print(
                            f'Epoch {epoch}: {iter}/{len(trainloader)}, lr = {optimizer.param_groups[0]["lr"]:.4}, '
                            f'Dice = {reduce_Dice.item():.4}, BCE = {reduce_BCE.item():.4}, loss_Sum = {reduce_all.item():.4}'
                        )

                        # Update loss tracking
                        each_loss[now_task] += reduce_all
                        count_batch[now_task] += 1
                        epoch_loss.append(float(reduce_all))

            'last round pop'
            for task_id in range(task_num):  # Loop from task 8 to task 22
                task_pool_image = globals()[f'task{task_id}_pool_image']
                task_pool_mask = globals()[f'task{task_id}_pool_mask']
                task_pool_weight = globals()[f'task{task_id}_pool_weight']
                task_scale = globals()[f'task{task_id}_scale']
                task_layer = globals()[f'task{task_id}_layer']

                if task_pool_image.num_imgs > 0:
                    batch_size = task_pool_image.num_imgs
                    images = task_pool_image.query(batch_size)
                    labels = task_pool_mask.query(batch_size)
                    wts = task_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task_scale.pop(0)
                        layers[bi] = task_layer.pop(0)

                    now_task = task_id
                    weight = edge_weight ** wts

                    'supervise_learning'
                    # term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
                    #                                                                          scales,
                    #                                                                          model, now_task, weight,
                    #                                                                          loss_seg_DICE, loss_seg_CE
                    #                                                                          )
                    'corrective learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning_correct(images, labels, batch_size,
                                                                                           scales, model, now_task,
                                                                                           weight,
                                                                                           loss_seg_DICE, loss_seg_CE,
                                                                                           confident_flag,
                                                                                           confident_e,
                                                                                           simattention_flag,
                                                                                           simattention_e, simloss_flag,
                                                                                           simloss_e, loss_KL, loss_MSE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()

                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))

                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss.item()))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', epoch_loss.item(), epoch)

            if (epoch >= 0) and (args.local_rank == 0) and (((epoch % 10 == 0) and (epoch >= 800)) or (epoch % 1 == 0)):
                print('save validation image ...')

                model.eval()

                for i in range(task_num):
                    globals()[f'task{i}_pool_image'] = ImagePool(8)
                    globals()[f'task{i}_pool_mask'] = ImagePool(8)
                    globals()[f'task{i}_pool_weight'] = ImagePool(8)
                    globals()[f'task{i}_scale'] = []
                    globals()[f'task{i}_layer'] = []
                    globals()[f'task{i}_filename'] = []
                    globals()[f'task{i}_single_df'] = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

                val_loss = np.zeros((task_num))
                val_F1 = np.zeros((task_num))
                val_Dice = np.zeros((task_num))
                val_TPR = np.zeros((task_num))
                val_PPV = np.zeros((task_num))
                cnt = np.zeros((task_num))

                size_512 = [0, 1, 2, 3]

                with torch.no_grad():
                    for iter, batch1 in enumerate(valloader):

                        'dataloader'
                        imgs = batch1[0].cuda()
                        lbls = batch1[1].cuda()
                        wt = batch1[2].cuda().float()
                        volumeName = batch1[3]
                        l_ids = batch1[4].cuda()
                        t_ids = batch1[5].cuda()
                        s_ids = batch1[6].cuda()


                        for ki in range(len(imgs)):
                            now_task = layer_num[l_ids[ki]] + t_ids[ki]
                            # Dynamically access the corresponding pools and lists using `globals()`
                            globals()[f'task{now_task}_pool_image'].add(imgs[ki].unsqueeze(0))
                            globals()[f'task{now_task}_pool_mask'].add(lbls[ki].unsqueeze(0))
                            globals()[f'task{now_task}_pool_weight'].add(wt[ki].unsqueeze(0))
                            globals()[f'task{now_task}_scale'].append(s_ids[ki])
                            globals()[f'task{now_task}_layer'].append(l_ids[ki])
                            globals()[f'task{now_task}_filename'].append(volumeName[ki])

                        output_folder = os.path.join(
                            args.snapshot_dir.replace('snapshots_2D/', '/Data4/CASC/validation_'),
                            str(epoch))

                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        optimizer.zero_grad()

                        for now_task in range(task_num):  # Loop through tasks 0 to 22
                            task_pool_image = globals()[f'task{now_task}_pool_image']
                            task_pool_mask = globals()[f'task{now_task}_pool_mask']
                            task_pool_weight = globals()[f'task{now_task}_pool_weight']
                            task_scale = globals()[f'task{now_task}_scale']
                            task_layer = globals()[f'task{now_task}_layer']
                            task_filename = globals()[f'task{now_task}_filename']
                            task_single_df = globals()[f'task{now_task}_single_df']

                            if task_pool_image.num_imgs >= batch_size:
                                if now_task in size_512:
                                    images = task_pool_image.query(batch_size)
                                    labels = task_pool_mask.query(batch_size)
                                    scales = torch.ones(batch_size).cuda()
                                    layers = torch.ones(batch_size).cuda()
                                    now_task = torch.tensor(now_task)
                                    filename = []
                                    for bi in range(len(scales)):
                                        scales[bi] = task_scale.pop(0)
                                        layers[bi] = task_layer.pop(0)
                                        filename.append(task_filename.pop(0))
                                    preds, simscore, simscore_noise, conscore, top_feature, noise_feature = model(images, torch.ones(batch_size).cuda() * now_task, scales, labels)

                                    now_preds = torch.argmax(preds, 1) == 1
                                    now_preds_onehot = one_hot_3D(now_preds.long())

                                    labels_onehot = one_hot_3D(labels.long())
                                    rmin, rmax, cmin, cmax = 0, 512, 0, 512

                                    F1 = 0
                                    DICE = 0
                                    TPR = 0
                                    PPV = 0

                                    cntt = 0
                                    if confident_flag > 0:
                                        if confident_e:
                                            now_weight = 1.5 ** ((conscore * labels) + (1 - conscore) * (1 - labels))
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                             labels_onehot,
                                                                                             rmin, rmax, cmin,
                                                                                             cmax, now_weight)
                                        else:
                                            print('a')
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                             labels_onehot,
                                                                                             rmin, rmax, cmin,
                                                                                             cmax, conscore * labels)

                                        F1 += confident_flag * now_F1
                                        DICE += confident_flag * now_DICE
                                        TPR += confident_flag * now_TPR
                                        PPV += confident_flag * now_PPV
                                        cntt += confident_flag

                                    if simattention_flag > 0:
                                        if simattention_e:
                                            now_weight = 1.5 ** (simscore - simscore_noise)
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                             labels_onehot, rmin, rmax,
                                                                                             cmin, cmax, now_weight)
                                        else:
                                            print('a')
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                             labels_onehot,
                                                                                             rmin, rmax, cmin,
                                                                                             cmax, simscore * labels)

                                        F1 += simattention_flag * now_F1
                                        DICE += simattention_flag * now_DICE
                                        TPR += simattention_flag * now_TPR
                                        PPV += simattention_flag * now_PPV
                                        cntt += simattention_flag

                                    val_F1[now_task] += F1 / cntt
                                    val_Dice[now_task] += DICE / cntt
                                    val_TPR[now_task] += TPR / cntt
                                    val_PPV[now_task] += PPV / cntt
                                    cnt[now_task] += 1

                                    for pi in range(len(images)):
                                        prediction = now_preds[pi]
                                        num = len(glob.glob(os.path.join(output_folder, '*')))
                                        out_image = images[pi].permute([1, 2, 0]).detach().cpu().numpy()
                                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                                   labels[pi].detach().cpu().numpy(), cmap=cm.gray)
                                        plt.imsave(os.path.join(output_folder,filename[pi] + '_preds_%s.png' % (now_task.item())),
                                                   prediction.detach().cpu().numpy(), cmap=cm.gray)
                                        plt.imsave(os.path.join(output_folder,filename[pi] + '_sim_%s.png' % (now_task.item())),
                                                   simscore[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                        plt.imsave(os.path.join(output_folder,filename[pi] + '_sim_noise_%s.png' % (now_task.item())),
                                                   simscore_noise[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                        plt.imsave(os.path.join(output_folder,filename[pi] + '_sim*label_%s.png' % (now_task.item())),
                                            simscore[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                        plt.imsave(os.path.join(output_folder,filename[pi] + '_sim_noise*label_%s.png' % (now_task.item())),
                                            simscore_noise[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                        plt.imsave(os.path.join(output_folder, filename[pi] + '_con_%s.png' % (now_task.item())),
                                            conscore[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                        plt.imsave(os.path.join(output_folder,filename[pi] + '_con*label_%s.png' % (now_task.item())),
                                            conscore[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy(), cmap='seismic')

                                        F1 = 0
                                        DICE = 0
                                        TPR = 0
                                        PPV = 0

                                        cntt = 0
                                        if confident_flag > 0:
                                            if confident_e:
                                                now_weight = 1.5 ** (
                                                        (conscore * labels) + (1 - conscore) * (1 - labels))
                                                now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                                        labels_onehot[pi].unsqueeze(0),
                                                                                                 rmin, rmax, cmin,
                                                                                                 cmax, now_weight)
                                            else:
                                                print('a')
                                                now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                                        labels_onehot[pi].unsqueeze(0),
                                                                                                 rmin, rmax, cmin,
                                                                                                 cmax,
                                                                                                 conscore * labels)

                                            F1 += confident_flag * now_F1
                                            DICE += confident_flag * now_DICE
                                            TPR += confident_flag * now_TPR
                                            PPV += confident_flag * now_PPV

                                            cntt += confident_flag

                                        if simattention_flag > 0:
                                            if simattention_e:
                                                now_weight = 1.5 ** (simscore - simscore_noise)
                                                now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                                        labels_onehot[pi].unsqueeze(0), rmin,
                                                                                                 rmax, cmin, cmax,
                                                                                                 now_weight)
                                            else:
                                                print('a')
                                                now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                                        labels_onehot[pi].unsqueeze(0),
                                                                                                 rmin, rmax, cmin,
                                                                                                 cmax,
                                                                                                 simscore * labels)
                                            F1 += simattention_flag * now_F1
                                            DICE += simattention_flag * now_DICE
                                            TPR += simattention_flag * now_TPR
                                            PPV += simattention_flag * now_PPV
                                            cntt += simattention_flag

                                        F1, DICE, TPR, PPV = F1 / cntt, DICE / cntt, TPR / cntt, PPV / cntt

                                        row = len(task_single_df)
                                        task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), TPR, PPV]

                    'last round pop'
                    for now_task in range(task_num):  # Loop through tasks 0 to 22
                        task_pool_image = globals()[f'task{now_task}_pool_image']
                        task_pool_mask = globals()[f'task{now_task}_pool_mask']
                        task_pool_weight = globals()[f'task{now_task}_pool_weight']
                        task_scale = globals()[f'task{now_task}_scale']
                        task_layer = globals()[f'task{now_task}_layer']

                        if task_pool_image.num_imgs > 0:
                            batch_size = task_pool_image.num_imgs

                            if now_task in size_512:
                                images = task_pool_image.query(batch_size)
                                labels = task_pool_mask.query(batch_size)
                                scales = torch.ones(batch_size).cuda()
                                layers = torch.ones(batch_size).cuda()
                                now_task = torch.tensor(now_task)
                                filename = []

                                for bi in range(len(scales)):
                                    scales[bi] = task_scale.pop(0)
                                    layers[bi] = task_layer.pop(0)
                                    filename.append(task_filename.pop(0))

                                preds, simscore, simscore_noise, conscore, top_feature, noise_feature = model(images, torch.ones(batch_size).cuda() * now_task, scales, labels)

                                now_preds = torch.argmax(preds, 1) == 1
                                now_preds_onehot = one_hot_3D(now_preds.long())

                                labels_onehot = one_hot_3D(labels.long())
                                rmin, rmax, cmin, cmax = 0, 512, 0, 512

                                F1 = 0
                                DICE = 0
                                TPR = 0
                                PPV = 0

                                cntt = 0
                                if confident_flag > 0:
                                    if confident_e:
                                        now_weight = 1.5 ** ((conscore * labels) + (1 - conscore) * (1 - labels))
                                        now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                         labels_onehot,
                                                                                         rmin, rmax, cmin,
                                                                                         cmax, now_weight)
                                    else:
                                        print('a')
                                        now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                         labels_onehot,
                                                                                         rmin, rmax, cmin,
                                                                                         cmax, conscore * labels)

                                    F1 += confident_flag * now_F1
                                    DICE += confident_flag * now_DICE
                                    TPR += confident_flag * now_TPR
                                    PPV += confident_flag * now_PPV
                                    cntt += confident_flag

                                if simattention_flag > 0:
                                    if simattention_e:
                                        now_weight = 1.5 ** (simscore - simscore_noise)
                                        now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                         labels_onehot, rmin, rmax,
                                                                                         cmin, cmax, now_weight)
                                    else:
                                        print('a')
                                        now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot,
                                                                                         labels_onehot,
                                                                                         rmin, rmax, cmin,
                                                                                         cmax, simscore * labels)

                                    F1 += simattention_flag * now_F1
                                    DICE += simattention_flag * now_DICE
                                    TPR += simattention_flag * now_TPR
                                    PPV += simattention_flag * now_PPV
                                    cntt += simattention_flag

                                val_F1[now_task] += F1 / cntt
                                val_Dice[now_task] += DICE / cntt
                                val_TPR[now_task] += TPR / cntt
                                val_PPV[now_task] += PPV / cntt
                                cnt[now_task] += 1

                                for pi in range(len(images)):
                                    prediction = now_preds[pi]
                                    num = len(glob.glob(os.path.join(output_folder, '*')))
                                    out_image = images[pi].permute([1, 2, 0]).detach().cpu().numpy()
                                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                               labels[pi].detach().cpu().numpy(), cmap=cm.gray)
                                    plt.imsave(
                                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                        prediction.detach().cpu().numpy(), cmap=cm.gray)
                                    plt.imsave(
                                        os.path.join(output_folder, filename[pi] + '_sim_%s.png' % (now_task.item())),
                                        simscore[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                    plt.imsave(
                                        os.path.join(output_folder, filename[pi] + '_sim_noise_%s.png' % (now_task.item())),
                                        simscore_noise[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                    plt.imsave(os.path.join(output_folder,
                                                            filename[pi] + '_sim*label_%s.png' % (now_task.item())),
                                               simscore[pi, ...].detach().cpu().numpy() * labels[
                                                   pi, ...].detach().cpu().numpy(), cmap='seismic')
                                    plt.imsave(os.path.join(output_folder, filename[pi] + '_sim_noise*label_%s.png' % (
                                        now_task.item())),
                                               simscore_noise[pi, ...].detach().cpu().numpy() * labels[
                                                   pi, ...].detach().cpu().numpy(), cmap='seismic')
                                    plt.imsave(
                                        os.path.join(output_folder, filename[pi] + '_con_%s.png' % (now_task.item())),
                                        conscore[pi, ...].detach().cpu().numpy(), cmap='seismic')
                                    plt.imsave(os.path.join(output_folder,
                                                            filename[pi] + '_con*label_%s.png' % (now_task.item())),
                                               conscore[pi, ...].detach().cpu().numpy() * labels[
                                                   pi, ...].detach().cpu().numpy(), cmap='seismic')

                                    F1 = 0
                                    DICE = 0
                                    TPR = 0
                                    PPV = 0

                                    cntt = 0
                                    if confident_flag > 0:
                                        if confident_e:
                                            now_weight = 1.5 ** (
                                                (conscore * labels) + (1 - conscore) * (1 - labels))
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(
                                                now_preds_onehot[pi].unsqueeze(0),
                                                labels_onehot[pi].unsqueeze(0),
                                                rmin, rmax, cmin,
                                                cmax, now_weight)
                                        else:
                                            print('a')
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(
                                                now_preds_onehot[pi].unsqueeze(0),
                                                labels_onehot[pi].unsqueeze(0),
                                                rmin, rmax, cmin,
                                                cmax,
                                                conscore * labels)

                                        F1 += confident_flag * now_F1
                                        DICE += confident_flag * now_DICE
                                        TPR += confident_flag * now_TPR
                                        PPV += confident_flag * now_PPV

                                        cntt += confident_flag

                                    if simattention_flag > 0:
                                        if simattention_e:
                                            now_weight = 1.5 ** (simscore - simscore_noise)
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(
                                                now_preds_onehot[pi].unsqueeze(0),
                                                labels_onehot[pi].unsqueeze(0), rmin,
                                                rmax, cmin, cmax,
                                                now_weight)
                                        else:
                                            print('a')
                                            now_F1, now_DICE, now_TPR, now_PPV = count_score(
                                                now_preds_onehot[pi].unsqueeze(0),
                                                labels_onehot[pi].unsqueeze(0),
                                                rmin, rmax, cmin,
                                                cmax,
                                                simscore * labels)
                                        F1 += simattention_flag * now_F1
                                        DICE += simattention_flag * now_DICE
                                        TPR += simattention_flag * now_TPR
                                        PPV += simattention_flag * now_PPV
                                        cntt += simattention_flag

                                    F1, DICE, TPR, PPV = F1 / cntt, DICE / cntt, TPR / cntt, PPV / cntt

                                    row = len(task_single_df)
                                    task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), TPR, PPV]

                    avg_val_F1 = val_F1 / cnt
                    avg_val_Dice = val_Dice / cnt
                    avg_val_TPR = val_TPR / cnt
                    avg_val_PPV = val_PPV / cnt

                    class_name = ['0_podocytes', '1_mesangial', '2_endo', '3_pecs']
                    output_str = 'Validate\n'

                    for i in range(task_num):
                        output_str += ' {}_f1={{:.4}} dsc={{:.4}} tpr={{:.4}} ppv={{:.4}}\n'.format(class_name[i])
                        task_single_df = globals()[f'task{i}_single_df']
                        task_single_df.to_csv(os.path.join(output_folder, 'testing_result_%d.csv' % (i)))

                    # Print the formatted string with the metrics for each task
                    print(output_str.format(
                        *[val.item() for task_metrics in zip(avg_val_F1, avg_val_Dice, avg_val_TPR, avg_val_PPV) for val
                          in task_metrics]
                    ))

                    # Create a DataFrame with the appropriate columns
                    df = pd.DataFrame(columns=['task', 'F1', 'Dice', 'TPR', 'PPV'])

                    # Populate the DataFrame for each task using a loop
                    for i in range(task_num):
                        df.loc[i] = [
                            class_name[i],
                            avg_val_F1[i].item(),
                            avg_val_Dice[i].item(),
                            avg_val_TPR[i].item(),
                            avg_val_PPV[i].item()
                        ]

                    # Save the DataFrame to a CSV file
                    df.to_csv(os.path.join(output_folder, 'validation_result.csv'), index=False)


                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                break

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
