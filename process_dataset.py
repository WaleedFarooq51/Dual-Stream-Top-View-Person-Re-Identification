from PIL import Image
import os
import numpy as np


data_path = "DATASET_PATH"
rgb_data_path= "RGB DATASET_PATH"
depth_data_path= "DEPTH DATASET_PATH"

rgb_imgs = ['clean_instance_1', 'clean_instance_2', 'clean_instance_3', 'clean_instance_4', 'clean_instance_5']
depth_imgs  = ['clean_instance_1', 'clean_instance_2', 'clean_instance_3', 'clean_instance_4', 'clean_instance_5']

# Load id info
file_path_train = os.path.join(data_path, 'training_ids.txt')
with open(file_path_train, 'r') as file:
	ids = file.read().splitlines()
	ids = [int(y) for y in ids[0].split(',')]
	id_train = ["%03d" % x for x in ids]
print("All Training IDs:",id_train)

#Saving all image paths into lists
img_paths_rgb = []
img_paths_depth  = []

for pid in sorted(id_train):
	for cam in rgb_imgs:
		img_dir = os.path.join(rgb_data_path, cam, pid)
		if os.path.isdir(img_dir):
			new_files = sorted([img_dir+'\\'+i for i in os.listdir(img_dir)])
			img_paths_rgb.extend(new_files)

	for cam in depth_imgs:
		img_dir = os.path.join(depth_data_path, cam, pid)
		if os.path.isdir(img_dir):
			new_files = sorted([img_dir+'\\'+i for i in os.listdir(img_dir)])
			img_paths_depth.extend(new_files)
			
#print("Path of all RGB imgs:",img_paths_rgb)
#print("Path of all Depth imgs",img_paths_depth)

# Assigning labels
pid_container = set()

for img_path in img_paths_depth:
	pid = os.path.basename(os.path.dirname(img_path))
	pid_container.add(pid)
print("Unique IDs",pid_container)

#Assigning labels to each unique IDs
pid2label = {pid:label for label, pid in enumerate(pid_container)}  
print("Lables to each IDs",pid2label)

#Converting imgs to numpy array as .npy files
def read_imgs(img_paths, img_w, img_h):     
	train_img = []
	train_label = []
	for img_path in img_paths:
		# img
		img = Image.open(img_path)
		img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
		pix_array = np.array(img)
		train_img.append(pix_array)
		
		# label
		pid = os.path.basename(os.path.dirname(img_path))
		#print("ID:",pid)
		label = pid2label[pid]
		#print("Label of ID:",label)
		#print("----------------------------------------------------------")
		
		train_label.append(label)
	
	return np.array(train_img), np.array(train_label)         #Returns Numpy arrays


train_img, train_label = read_imgs(img_paths_rgb, img_w=128, img_h=256)
np.save(os.path.join(data_path, 'train_rgb_resized_img.npy'), train_img)
np.save(os.path.join(data_path, 'train_rgb_resized_label.npy'), train_label)
print("RGB imgs to numpy array converted")

train_img, train_label = read_imgs(img_paths_depth, img_w=128, img_h=256)
np.save(os.path.join(data_path, 'train_depth_resized_img.npy'), train_img)
np.save(os.path.join(data_path, 'train_depth_resized_label.npy'), train_label)
print("Depth imgs to numpy array converted")
