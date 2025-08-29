from __future__ import print_function
from data_loader import TrainData, TestData
from data_manager import *
from eval_metrics import evaluation
from scheduler import create_scheduler
from make_dual_stream_resnet_model import dual_stream_net
from config import cfg
from train_utils import *
from loss import OriTripletLoss
from tqdm import tqdm
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
import sys
import time
import os


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=2, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=128, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=128, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

test_mode = [1, 2]

data_path = cfg.DATA_PATH

log_path = args.log_path + 'training_log/'

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

suffix = "checkpoint" + 'p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, cfg.BASE_LR, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

sys.stdout = Logger(log_path + suffix + '_os.txt')
print("==========\nArgs:{}\n==========".format(args))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 1

print('==> Loading data..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([                                    # Random erasing trasnf to be added
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),         
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),    
    transforms.ToTensor(),
    normalize,
])

end = time.time()

# training set
trainset = TrainData(data_path, transform=transform_train)

# generate the idx of each person identity
rgb_pos, depth_pos = GenIdx(trainset.train_rgb_label, trainset.train_depth_label)

# testing set
query_img, query_label, query_folder = process_query(data_path)

queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_rgb_label))
nquery = len(query_label)

print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building Models..')
network = dual_stream_net(n_class, arch=args.arch)
network.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        network.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss functions
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)

if args.optim == 'sgd':

    optimizer = optim.SGD([
        # RGB stream layers 
        {'params': network.rgb_stream_stem_block_resnet.parameters(), 'lr': 0.1 * cfg.BASE_LR},
      
        # Depth stream layers 
        {'params': network.depth_stream_stem_block_resnet.parameters(), 'lr': 0.1 * cfg.BASE_LR},
       
        # Residual block layers
        {'params': network.residual_block_resnet.parameters(), 'lr': 0.1 * cfg.BASE_LR},
         
        # Bottleneck and Classifier layers 
        {'params': network.bottleneck.parameters(), 'lr': cfg.BASE_LR},
        {'params': network.classifier.parameters(), 'lr': cfg.BASE_LR},],

        weight_decay=7e-3, momentum=0.9, nesterov=True)

scheduler = create_scheduler(cfg, optimizer)

def train(epoch):

    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    scheduler.step(epoch)

    # switch to train mode
    network.train()

    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        optimizer.zero_grad()

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        label1 = Variable(label1.cuda())
        label2 = Variable(label2.cuda())
        labels = torch.cat((label1, label2), 0)
        labels = Variable(labels.cuda())

        data_time.update(time.time() - end)

        fused_features, scores = network(input1, input2)

        loss_id = criterion_id(scores, labels.long())
        loss_tri, batch_acc = criterion_tri(fused_features, labels)

        correct += (batch_acc / 2)
        _, predicted = scores.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_id + loss_tri
        
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += label1.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % 32 == 0 :
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.2e} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, (batch_idx+1), len(trainloader), optimizer.state_dict()['param_groups'][4]['lr'],
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

trial_seeds = list(range(10))

def test(epoch):

    all_cmc = 0
    all_mAP = 0
    all_mINP = 0

    # switch to evaluation
    network.eval()

    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_pool_feat = np.zeros((nquery, 2048))
    query_bn_feat = np.zeros((nquery, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
           
            pool_feat, bn_feat = network(input, input)

            query_pool_feat[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            query_bn_feat[ptr:ptr + batch_num, :] = bn_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    #print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    for i in tqdm(trial_seeds):

        gall_img, gall_label, gall_folder = process_gallery(data_path, trial=i)
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        ngall = len(gall_label)
        
        # switch to evaluation mode
        network.eval()

        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0

        gall_pool_feat = np.zeros((ngall, 2048))
        gall_bn_feat = np.zeros((ngall, 2048))

        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
          
                pool_feat, bn_feat = network(input, input)

                gall_pool_feat[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
                gall_bn_feat[ptr:ptr + batch_num, :] = bn_feat.detach().cpu().numpy()
                ptr = ptr + batch_num
        #print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        start = time.time()

        # compute the similarity
        distmat = np.matmul(query_bn_feat, np.transpose(gall_bn_feat))

        # evaluation
        cmc, mAP, mINP = evaluation(-distmat, query_label, gall_label, query_folder, gall_folder)
        print('\n mAP: {:.2%} | mInp: {:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-8: {:.2%}'.format(mAP, mINP, cmc[0], cmc[4], cmc[7]))

        #print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

        all_cmc += cmc
        all_mAP += mAP
        all_mINP += mINP
    
    all_cmc /= 10.0
    all_mAP /= 10.0
    all_mINP /= 10.0

    return all_cmc, all_mAP, all_mINP

# Training Model
print('==> Start Training...')

for epoch in range(start_epoch, cfg.MAX_EPOCH + 1):

    print('==> Preparing Data Loader...')
    
    # identity sampler
    sampler = IdentitySampler(trainset.train_rgb_label, trainset.train_depth_label, rgb_pos, depth_pos, args.num_pos, args.batch_size, epoch)
   
    trainset.cIndex = sampler.index1  # rgb index
    trainset.tIndex = sampler.index2  # depth index
    
    print("Epoch: ",epoch)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, num_workers=args.workers, drop_last=True)

    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP = test(epoch)

        # save model
        if cmc[0] > best_acc: 
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': network.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_bestepoch_{}.t'.format(epoch))

            print('Best Epoch [{}]'.format(best_epoch))

        print('\n Average:')
        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-8: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(cmc[0], cmc[4], cmc[7], mAP, mINP))

    