from __future__ import print_function
from eval_metrics import evaluation
from dataloader import TestData
from datamanager import *
from model.make_model import build_vision_transformer
from config.config import cfg
from transforms import transform_test
from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import argparse


parser = argparse.ArgumentParser(description='PMT Training')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--test-batch', default=128, type=int,
                    help='testing batch size')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--gall_mode', default='single', type=str,
                    help='single or multi')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = cfg.DATA_PATH
n_class = 130

print('==> Building model..')
model = build_vision_transformer(num_classes = n_class, cfg = cfg)
model.to(device)
cudnn.benchmark = True
model.eval()

def extract_gall_feat(gall_loader):
    model.eval()
    ptr = 0
    gall_feat = np.zeros((ngall, 768))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    return gall_feat

def extract_query_feat(query_loader):
    model.eval()
    ptr = 0
    query_feat = np.zeros((nquery, 768))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    return query_feat

all_cmc = 0
all_mAP = 0
all_mINP = 0

# load checkpoint
print('==> Resuming from checkpoint..')
if len(args.resume) > 0:

    model_path = args.resume
    print("ModelPath:",model_path)

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        model.load_param(model_path)
        print('==> loaded checkpoint {}'.format(args.resume))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))
else:
    print('==> no checkpoint provided')

query_img, query_label, query_folder = process_query(data_path)
gall_img, gall_label, gall_folder = process_gallery(data_path, trial=0, gall_mode=args.gall_mode)

nquery = len(query_label)
ngall = len(gall_label)

print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")
print("Dataset Loaded")

queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
query_loader = data.DataLoader(queryset, batch_size=128, shuffle=False, num_workers=args.workers)

query_feat = extract_query_feat(query_loader)

trial_seeds = list(range(10))

for i in tqdm(trial_seeds):

    gall_img, gall_label, gall_folder = process_gallery(data_path, trial=i, gall_mode=args.gall_mode) 

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    gall_feat = extract_gall_feat(trial_gall_loader)

    distmat = -np.matmul(query_feat, np.transpose(gall_feat))

    cmc, mAP, mInp = evaluation(distmat, query_label, gall_label, query_folder, gall_folder)

    print('\n mAP: {:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-8: {:.2%} | top-10: {:.2%}'.format(mAP, cmc[0], cmc[4], cmc[7], cmc[9]))

    all_cmc += cmc
    all_mAP += mAP
    all_mINP += mInp

all_cmc /= 10.0
all_mAP /= 10.0
all_mINP /= 10.0

print('\n Average:')
print('mAP: {:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-8: {:.2%} | top-10: {:.2%}'.format(all_mAP, all_cmc[0], all_cmc[4], all_cmc[7], all_cmc[9]))
