from __future__ import print_function, absolute_import
from PIL import Image
import os
import numpy as np
import random

def process_query(data_path, relabel=False):
    
    probe_images = ['pr6']

    file_path = os.path.join(data_path,'testing_ids.txt')
    img_files = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%03d" % x for x in ids]

    for id in sorted(ids):
        for img in probe_images:
            img_dir = os.path.join(data_path,img,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'\\'+i for i in os.listdir(img_dir)])
                img_files.extend(new_files)

    query_img = []
    query_id = []
    query_folder = []

    for img_path in img_files:
        folder = int(''.join(filter(str.isdigit, img_path.split(os.sep)[-3])))  
        pid = int(img_path.split(os.sep)[-2])                                 

        query_img.append(img_path)
        query_id.append(pid)
        query_folder.append(folder)
        
    return query_img, np.array(query_id), np.array(query_folder)

generated_seeds_values=[]

def process_gallery(data_path, trial, relabel=False):
    
    random.seed(trial)
    generated_seed_value = int(random.random() * 10000)
    generated_seeds_values.append(generated_seed_value)
    #print("Generated seed values: ",generated_seeds_values)
    
    gallery_images = ['gal5']

    file_path = os.path.join(data_path,'testing_ids.txt')
    img_files = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%03d" % x for x in ids]

    for id in sorted(ids):
        for img in gallery_images:
            img_dir = os.path.join(data_path,img,id)

            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'\\'+i for i in os.listdir(img_dir)])
                img_files.append(random.choice(new_files))

                ## For Multi-Shot ##
                #sample_size = min(10, len(new_files))
                #img_files.append(np.random.choice(new_files, sample_size, replace=False))

    gall_img = []
    gall_id = []
    gall_folder = []

    for img_path in img_files:
        folder = int(''.join(filter(str.isdigit, img_path.split(os.sep)[-3])))  
        pid = int(img_path.split(os.sep)[-2])

        gall_img.append(img_path)
        gall_id.append(pid)
        gall_folder.append(folder)
        
        ## For Multi-Shot ##
        # for i in img_path:
        #     camid_str = i.split(os.sep)[-3] 
        #     camid = int(''.join(filter(str.isdigit, camid_str)))
        #     pid = int(i.split(os.sep)[-2])

        #     gall_img.append(i)
        #     gall_id.append(pid)
        #     gall_folder.append(camid)

    # Save images in a unique folder for each trial inside the function
    # output_folder= "..\\Test_Imgs"
    # unique_folder = os.path.join(output_folder, f'trial_{trial}')
    # if not os.path.exists(unique_folder):
    #     os.makedirs(unique_folder)

    # for img_path in gall_img:
    #     img = Image.open(img_path)
    #     img_name = os.path.basename(img_path)
    #     img_save_path = os.path.join(unique_folder, img_name)
    #     img.save(img_save_path)
    #     print(f"Saved: {img_save_path}")

    return gall_img, np.array(gall_id), np.array(gall_folder)
    