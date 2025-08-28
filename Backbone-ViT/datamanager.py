from __future__ import print_function, absolute_import
from PIL import Image
import os
import numpy as np
import random
import re

def process_query(data_path, relabel=False):
    
    probe_images = ['pr6']
    file_path = os.path.join(data_path, 'testing_ids.txt')

    files_rgb = []
    files_depth = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%03d" % x for x in ids]

    for id in sorted(ids):
        for folder in probe_images:
            img_dir = os.path.join(data_path, folder, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '\\' + i for i in os.listdir(img_dir)])
                files_depth.extend(new_files)

    query_img = []
    query_id = []
    query_folder = []

    for img_path in files_depth:
        foldername = int(''.join(filter(str.isdigit, img_path.split(os.sep)[-3])))  
        pid = int(img_path.split(os.sep)[-2])                            

        query_img.append(img_path)
        query_id.append(pid)
        query_folder.append(foldername)

    return query_img, np.array(query_id), np.array(query_folder)

generated_seeds_values=[]

def process_gallery(data_path, trial, relabel=False, gall_mode='single'):

    random.seed(trial)
    generated_seed_value = int(random.random() * 10000)
    generated_seeds_values.append(generated_seed_value)
    #print("Generated seed values: ",generated_seeds_values)

    gall_images = ['gal5']

    file_path = os.path.join(data_path, 'testing_ids.txt')
    files_rgb = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%03d" % x for x in ids]
    
    for id in sorted(ids):
        for folder in gall_images:
            img_dir = os.path.join(data_path, folder, id)
            
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '\\' + i for i in os.listdir(img_dir)])
              
                if gall_mode == 'single':
                    files_rgb.append(random.choice(new_files))
                if gall_mode == 'multi':
                    files_rgb.append(np.random.choice(new_files, 10, replace=False))
    
    gall_img = []
    gall_id = []
    gall_folder = []

    for img_path in files_rgb:
        if gall_mode == 'single':
            foldername = int(''.join(filter(str.isdigit, img_path.split(os.sep)[-3])))  
            pid = int(img_path.split(os.sep)[-2])
            
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_folder.append(foldername)

        if gall_mode == 'multi':
            for i in img_path:
                foldername_str = i.split(os.sep)[-3] 
                foldername = int(''.join(filter(str.isdigit, foldername_str)))
                pid = int(i.split(os.sep)[-2])

                gall_img.append(i)
                gall_id.append(pid)
                gall_folder.append(foldername)
    
    # Save images in a unique folder for each trial inside the function
    # output_folder= "output_folder_path"
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