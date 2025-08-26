from PIL import Image
from config import cfg
import numpy as np
import torch.utils.data as data


class TrainData(data.Dataset):
    def __init__(self, data_dir, transform=None, rgbIndex = None, depthIndex = None):
         
        data_dir = cfg.DATA_PATH

        # Load training images (path) and labels
        train_rgb_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_rgb_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_depth_image = np.load(data_dir + 'train_depth_resized_img.npy')
        self.train_depth_label = np.load(data_dir + 'train_depth_resized_label.npy')
        
        # BGR to RGB
        self.train_rgb_image   = train_rgb_image
        self.train_depth_image = train_depth_image
        self.transform = transform
        self.cIndex = rgbIndex
        self.tIndex = depthIndex

    def __getitem__(self, index):

        img1, target1 = self.train_rgb_image[self.cIndex[index]],  self.train_rgb_label[self.cIndex[index]]
        img2, target2 = self.train_depth_image[self.tIndex[index]], self.train_depth_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)
                
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

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
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
                  

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()

        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label