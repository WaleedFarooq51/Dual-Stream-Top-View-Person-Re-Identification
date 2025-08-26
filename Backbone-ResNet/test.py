from __future__ import print_function
from data_loader import TestData
from data_manager import *
from eval_metrics import evaluation
#from make_single_stream_resnet_model import single_stream_net            # Single-Stream network
from make_dual_stream_resnet_model import dual_stream_net                 # Dual-Stream network
from test_utils import *
from config import cfg
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
import time


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
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
parser.add_argument('--method', default='awg', type=str,
                    metavar='m', help='method type: base or awg')
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = cfg.DATA_PATH

n_class = 130
test_mode = [1, 2]

best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 2048

print('==> Building model..')
network = dual_stream_net(n_class, arch=args.arch)
network.to(device)    
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion.to(device)

print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

print('==> Resuming from checkpoint..')
if len(args.resume) > 0:
    model_path =  args.resume

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        network.load_state_dict(checkpoint['net'])
        print('==> successfully loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    else:
        print('==> no checkpoint found at {}'.format(args.resume))


def extract_gall_feat(gall_loader):

    network.eval()

    print ('Extracting Gallery Feature...')

    torch.cuda.synchronize()
    start = time.time()
    ptr = 0

    gall_pool_feat = np.zeros((ngall, 2048))
    gall_bn_feat = np.zeros((ngall, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
        
            pool_feat, bn_feat = network(input, input)                            # Dual-Stream network

            #pool_feat, bn_feat = network(input)                                   # Single-Stream network

            gall_pool_feat[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            gall_bn_feat[ptr:ptr + batch_num, :] = bn_feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    torch.cuda.synchronize()
    print('Extracting Time for Gallery features:\t {:.3f}'.format(time.time()-start))

    return gall_bn_feat
    
def extract_query_feat(query_loader):
  
    network.eval()

    print ('Extracting Query Feature...')

    torch.cuda.synchronize()
    start = time.time()
    ptr = 0

    query_pool_feat = np.zeros((nquery, 2048))
    query_bn_feat = np.zeros((nquery, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
           
            pool_feat, bn_feat = network(input, input)                            # Dual-Stream network

            #pool_feat, bn_feat = network(input)                                  # Single-Stream network

            query_pool_feat[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            query_bn_feat[ptr:ptr + batch_num, :] = bn_feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    torch.cuda.synchronize()        
    print('Extracting Time for Query Features:\t {:.3f}'.format(time.time()-start))

    return query_bn_feat

query_img, query_label, query_folder = process_query(data_path)
gall_img, gall_label, gall_folder = process_gallery(data_path, trial=0)

nquery = len(query_label)
ngall = len(gall_label)

print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=0)
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

#query_feat_bn= extract_query_feat(query_loader)

all_cmc = 0
all_mAP = 0
all_mINP = 0

trial_seeds = list(range(10))
latencies=[]

for i in tqdm(trial_seeds):

    torch.cuda.synchronize()
    start_time = time.time()

    query_feat_bn = extract_query_feat(query_loader)

    gall_img, gall_label, gall_folder = process_gallery(data_path, trial= i)

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=0)

    gall_feat_bn = extract_gall_feat(trial_gall_loader)

    torch.cuda.synchronize()
    matching_start_time = time.time()

    # compute the similarity
    distmat = np.matmul(query_feat_bn, np.transpose(gall_feat_bn))

    # evaluation
    cmc, mAP, mINP = evaluation(-distmat, query_label, gall_label, query_folder, gall_folder)
    
    #For retrieval case; Qualitative analysis
    #cmc, mAP, mINP = evaluation(-distmat, query_label, gall_label, query_folder, gall_folder, q_paths=query_img, g_paths=gall_img, visualize=True, vis_dir= 'path to retrieval_folder')

    torch.cuda.synchronize()
    matching_end_time = time.time()
    print('Matching Time:\t {:.3f}'.format(matching_end_time - matching_start_time)) 

    all_cmc += cmc
    all_mAP += mAP
    all_mINP += mINP

    print('Test Trial: {}'.format(i))
    print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-8: {:.2%}| Rank-10: {:.2%}| mAP: {:.2%}'.format(cmc[0], cmc[4], cmc[7], cmc[9], mAP))
    
    torch.cuda.synchronize()
    end_time = time.time()
    print('Inference Time:\t {:.3f}'.format(end_time - start_time))

    latencies.append(end_time - start_time)
   
all_cmc /= 10.0
all_mAP /= 10.0
all_mINP /= 10.0

print('Average:')
print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-8: {:.2%} | Rank-10: {:.2%} | mAP: {:.2%}'.format(all_cmc[0], all_cmc[4], all_cmc[7], all_cmc[9], all_mAP))

average_latency = sum(latencies) / len(latencies)
print(f"Average Inference Latency: {average_latency:.3f} seconds")
