from torch.utils.data.sampler import Sampler
import os
import numpy as np
import sys
import os.path as osp
import torch

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    
def GenIdx(train_rgb_label):
    rgb_pos = []
    unique_label_color = np.unique(train_rgb_label)

    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_rgb_label) if v==unique_label_color[i]]
        rgb_pos.append(tmp_pos)

    return rgb_pos

# def GenIdx(train_depth_label):        
#     depth_pos = []
#     unique_label_thermal = np.unique(train_depth_label)

#     for i in range(len(unique_label_thermal)):
#         tmp_pos = [k for k,v in enumerate(train_depth_label) if v==unique_label_thermal[i]]
#         depth_pos.append(tmp_pos)

#     return depth_pos
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_rgb_label, train_depth_label: labels of two modalities
            rgb_pos, depth_pos: positions of each identity
            batchSize: batch size
    """
    def __init__(self, train_rgb_label, rgb_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_rgb_label)
        self.n_classes = len(uni_label)
        
        N = len(train_rgb_label)

        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  

            for i in range(batchSize):
                sample_rgb  = np.random.choice(rgb_pos[batch_idx[i]], num_pos)
                #sample_depth = np.random.choice(depth_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_rgb
                    #index2= sample_depth
                else:
                    index1 = np.hstack((index1, sample_rgb))
                    #index2 = np.hstack((index2, sample_depth))
        
        self.index1 = index1
        #self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N       

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad