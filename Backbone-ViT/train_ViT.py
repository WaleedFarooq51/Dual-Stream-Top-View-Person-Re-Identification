from dataloader import TrainData,TestData, GenIdx, IdentitySampler
from datamanager import process_gallery, process_query
from transforms import transform_rgb, transform_rgb2gray, transform_depth, transform_test
from model.make_model import build_vision_transformer
from scheduler import create_scheduler
from loss.Triplet import TripletLoss
from optimizer import make_optimizer
from eval_metrics import evaluation
from utils import AverageMeter, set_seed
from config.config import cfg
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda import amp
import torch.utils.data as data
import torch
import torch.nn as nn
import optimizer
import numpy as np
import os.path as osp
import argparse
import os
import time


parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--config_file', default='config.yml',
                    help='path to config file', type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--model_path', default='save_model/',
                    help='model save path', type=str)
parser.add_argument('--num_workers', default=0,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=128,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None,nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != '':
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

set_seed(cfg.SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

data_path = cfg.DATA_PATH

trainset_gray = TrainData(data_path, transform1=transform_rgb2gray, transform2=transform_depth)
rgb_pos_gray, depth_pos_gray = GenIdx(trainset_gray.train_rgb_label, trainset_gray.train_depth_label)

trainset_rgb = TrainData(data_path, transform1=transform_rgb, transform2=transform_depth)
rgb_pos, depth_pos = GenIdx(trainset_rgb.train_rgb_label, trainset_rgb.train_depth_label)

num_classes = len(np.unique(trainset_rgb.train_rgb_label))

model = build_vision_transformer(num_classes=num_classes,cfg = cfg)
model.to(device)

# load checkpoint
if len(args.resume) > 0:
    model_path = args.resume
    print("ModelPath:",model_path)

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        model.load_param(model_path)
        print('==> loaded checkpoint {}'.format(args.resume))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# Loss
criterion_ID = nn.CrossEntropyLoss()
criterion_Tri = TripletLoss(margin=cfg.MARGIN, feat_norm='no')

optimizer = make_optimizer(cfg, model)
scheduler = create_scheduler(cfg, optimizer)

scaler = amp.GradScaler()

# Test query data
query_img, query_label, query_folder = process_query(data_path)
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

# Test query loader
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)

loss_meter = AverageMeter()
loss_ce_meter = AverageMeter()
loss_tri_meter = AverageMeter()
acc_meter = AverageMeter()

def train(epoch):
    start_time = time.time()

    loss_meter.reset()
    loss_ce_meter.reset()
    loss_tri_meter.reset()
    acc_meter.reset()

    scheduler.step(epoch)
    model.train()

    for idx, (input1, input2, label1, label2) in enumerate(trainloader):

        optimizer.zero_grad()
        input1 = input1.to(device)
        input2 = input2.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        labels = torch.cat((label1,label2),0)

        with amp.autocast(enabled=True):
        
            fused_stream_features= torch.cat([input1,input2])
     
            scores, fused_features = model(fused_stream_features)            

            loss_id = criterion_ID(scores, labels.long())

            loss_tri = criterion_Tri(fused_features, fused_features, labels)

            loss = loss_id + loss_tri

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = (scores.max(1)[1] == labels).float().mean()

        loss_tri_meter.update(loss_tri.item())
        loss_ce_meter.update(loss_id.item())
        loss_meter.update(loss.item())
        acc_meter.update(acc, 1)

        torch.cuda.synchronize()

        if (idx + 1) % 32 == 0 :
            print('Epoch[{}] Iteration[{}/{}]'
                  ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                  'Acc: {:.3f}, '
                  'Base Lr: {:.2e} '.format(epoch, (idx+1),
                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                loss_ce_meter.avg, acc_meter.avg,
                optimizer.state_dict()['param_groups'][0]['lr']))

    end_time = time.time()
    time_per_batch = end_time - start_time
    print(' Epoch {} done. Time per batch: {:.1f}[min] '.format(epoch, time_per_batch/60))

trial_seeds = list(range(10))

def test(query_loader):
    all_cmc = 0
    all_mAP = 0
    all_mINP = 0
    
    model.eval()
    print('Testing...')

    ptr = 0
    nquery = len(query_label)
    query_feat = np.zeros((nquery, 768))
    
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    for i in tqdm(trial_seeds):

        gall_img, gall_label, gall_folder = process_gallery(data_path, trial=i, gall_mode='single')
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)

        ngall = len(gall_label)
        ptr = 0
        gall_feat = np.zeros((ngall, 768))

        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat = model(input)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))

        cmc, mAP, mInp = evaluation(distmat, query_label, gall_label, query_folder, gall_folder)

        print('\n mAP: {:.2%} | mInp: {:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-8: {:.2%}'.format(mAP, mInp, cmc[0], cmc[4], cmc[7]))

        all_cmc += cmc
        all_mAP += mAP
        all_mINP += mInp

    all_cmc /= 10.0
    all_mAP /= 10.0
    all_mINP /= 10.0

    return all_cmc, all_mAP, all_mINP

# Training
best_mAP = 0
print('==> Start Training...')

for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH + 1):

    print('==> Preparing Data Loader...')

    sampler_rgb = IdentitySampler(trainset_rgb.train_rgb_label, trainset_rgb.train_depth_label,
                                  rgb_pos,depth_pos, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)

    trainset_rgb.rgbIndex = sampler_rgb.index1            # rgb index
    trainset_rgb.depthIndex = sampler_rgb.index2          # depth index

    trainloader = data.DataLoader(trainset_rgb, batch_size=cfg.BATCH_SIZE, sampler=sampler_rgb,
                                    num_workers=args.num_workers, drop_last=True, pin_memory=True)

    train(epoch)
    
    if epoch > args.start_test and epoch % args.test_epoch == 0:
        cmc, mAP, mINP = test(query_loader)
        
        print('\n Average:')
        print('\n mAP: {:.2%} | mInp: {:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-8: {:.2%}'.format(mAP, mINP, cmc[0], cmc[4], cmc[7]))

        if mAP > best_mAP:
            best_mAP = mAP
          
            torch.save(model.state_dict(), osp.join('my_trained_model', os.path.basename(args.config_file)[:-4] + '_best{}.pth'.format(epoch)))  # maybe not the best

    if epoch > 0 and epoch % args.save_epoch == 0:

        torch.save(model.state_dict(), osp.join('my_trained_model', os.path.basename(args.config_file)[:-4]  + '_epoch{}.pth'.format(epoch)))






