from PIL import Image
from torch.utils.data.sampler import Sampler
import torch.utils.data as data
import numpy as np

class TrainData(data.Dataset):
    def __init__(self, data_dir, transform1=None,transform2 = None, rgbIndex=None, depthIndex=None):
        
        # Load training images (path) and labels
        train_rgb_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_rgb_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_depth_image = np.load(data_dir + 'train_colored_depth_resized_img.npy')
        self.train_depth_label = np.load(data_dir + 'train_colored_depth_resized_label.npy')

        # RGB format
        self.train_rgb_image = train_rgb_image
        self.train_depth_image = train_depth_image
        self.transform1 = transform1
        self.transform2 = transform2
        self.rgbIndex = rgbIndex
        self.depthIndex = depthIndex

    def __getitem__(self, index):

        img1, target1 = self.train_rgb_image[self.rgbIndex[index]], self.train_rgb_label[self.rgbIndex[index]]
        img2, target2  = self.train_depth_image[self.depthIndex[index]], self.train_depth_label[self.depthIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)
        
        #print("IMAGE 1 IS :", img1)
        #print("TARGET 1 IS :", target1)
        #print("IMAGE 2 IS :", img2)
        #print("TARGET 2 IS :", target2)
        #print("-------------------------------------------------")

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(224, 224)):
        test_image = []
        
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
            
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

def GenIdx(train_rgb_label, train_depth_label):
    rgb_pos = []
    unique_label_color = np.unique(train_rgb_label)
   
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_rgb_label) if v == unique_label_color[i]]
        rgb_pos.append(tmp_pos)

    depth_pos = []
    unique_label_thermal = np.unique(train_depth_label)
    
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_depth_label) if v == unique_label_thermal[i]]
        depth_pos.append(tmp_pos)

    return rgb_pos, depth_pos

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_rgb_label, train_depth_label: labels of two modalities
            rgb_pos, depth_pos: positions of each identity
            batchSize: batch size
    """
    def  __init__(self, train_rgb_label, train_depth_label, rgb_pos, depth_pos, batchSize, per_img):
        uni_label = np.unique(train_rgb_label)
        self.n_classes = len(uni_label)

        sample_rgb = np.arange(batchSize)
        sample_depth = np.arange(batchSize)
        N = np.maximum(len(train_rgb_label), len(train_depth_label))

        # per_img = 4
        per_id = batchSize / per_img
        
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_rgb[i:i + per_img] = np.random.choice(rgb_pos[batch_idx[s]], per_img, replace=False)
                sample_depth[i:i + per_img] = np.random.choice(depth_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index1 = sample_rgb
                index2 = sample_depth
            else:
                index1 = np.hstack((index1, sample_rgb))
                index2 = np.hstack((index2, sample_depth))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
