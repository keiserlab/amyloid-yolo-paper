import csv
import glob, os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
import random
import pickle
import shutil
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import argparse
import subprocess

np.random.seed(42)

def preProcess(weak_label=False):
    """
    From image_details_phase1.csv, will create a new df
    that also includes labels for cored, diffuse, and CAA
    Not all images have a human annotation however
    If a human annotation is available, then use it 
    else if weak_label=True, then run the consensus of 2 model to get a prediction, else empty
    Returns a dictionary with key: 1536 image name to value: list of (bbox coordinate, class label) tuples
    """
    ##first creat a mapp of phase 1 image name to 3 class annotation
    binary_consensus_labels = pd.read_csv("csvs/strict_agreed_by_2.csv")
    mapp = {} ##key: 256 image name to value: annotation tuple (cored, diffuse, CAA)
    for index,row in binary_consensus_labels.iterrows():
        full_path_img = row["imagename"]
        img_name = full_path_img[full_path_img.find("/") + 1:]
        mapp[img_name] = (int(row["cored"]), int(row["diffuse"]), int(row["CAA"]))
    ##add columns cored, diffuse, and CAA to image_details dataframe
    ##if human annotation exists, pull human annotation, else run prediction
    ##as iterating over df, save to dictionary 
    df = pd.read_csv("csvs/image_details_phase1.csv")
    df["cored"] = [""] * len(df) 
    df["diffuse"] = [""] * len(df) 
    df["CAA"] = [""] * len(df) 
    model = torch.load("pickles/model_all_fold_3_thresholding_2_l2.pkl")
    norm = np.load("pickles/normalization.npy", allow_pickle=True).item()
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
    final_mapp = {}
    for index, row in df.iterrows(): 
        img_path_1536 = "data/custom/images/" + row["source"] + "_0_" + str(row["tile_row"]) + "_" + str(row["tile_column"]) + ".jpg"
        bbox_coord = row["blob coordinates (xywh)"]
        bbox_coord = bbox_coord.replace("[", "").replace("]", "").split(" ")
        bbox_coord = [int(x) for x in bbox_coord if x != ""]      
        ##if human annotation exists
        if row["imagename"] in mapp:
            img_name = row["imagename"] 
            df.at[index, "cored"] = mapp[img_name][0]
            df.at[index, "diffuse"] = mapp[img_name][1]
            df.at[index, "CAA"] = mapp[img_name][2]
        else: ##predict the annotation and use this 
            ##center the bbox
            if weak_label:
                img = cv2.imread(img_path_1536)
                img_256 = get256Img(img, bbox_coord)
                bbox_class_preds = getClassPreds(img_256, model, data_transforms)
                df.at[index, "cored"] = bbox_class_preds[0]
                df.at[index, "diffuse"] = bbox_class_preds[1]
                df.at[index,"CAA"] = bbox_class_preds[2]
            else: ##if we didn't find the human annotation and weak label == False, then continue 
                continue
        if img_path_1536 not in final_mapp:
            final_mapp[img_path_1536] = [(bbox_coord, (df.at[index, "cored"], df.at[index, "diffuse"], df.at[index, "CAA"]))]
        else:
            final_mapp[img_path_1536].append((bbox_coord, (df.at[index, "cored"], df.at[index, "diffuse"], df.at[index, "CAA"])))
    return final_mapp

def get256Img(img, bbox_coord):
    """
    Given a cv2 img, and a bbox_coordinate (top left x, top left y, width (x), height (y)) , will return a 256 x 256 cropped image that is centered on the bbox
    """
    center_point = (int(bbox_coord[0] + (bbox_coord[2] / 2)), int(bbox_coord[1] + (bbox_coord[3] / 2)))
    x_left_valid = center_point[0] - 128 > 0 
    x_right_valid = center_point[0] + 128 < 1536
    y_top_valid = center_point[1] - 128 > 0
    y_bottom_valid = center_point[1] + 128 < 1536 
    x_valid = x_left_valid and x_right_valid
    y_valid = y_top_valid and y_bottom_valid
    x_crop_left, x_crop_right, y_crop_top, y_crop_bottom = -1,-1,-1,-1
    ##easy case
    if x_valid and y_valid:
        x_crop_left, x_crop_right = center_point[0] - 128, center_point[0] + 128
        y_crop_top, y_crop_bottom = center_point[1] - 128, center_point[1] + 128
    ##either x is valid, or y is valid 
    elif x_valid and not y_valid:
        if y_bottom_valid:
            x_crop_left, x_crop_right =center_point[0] - 128,center_point[0] + 128
            y_crop_top, y_crop_bottom = 0, 256       
        if y_top_valid:
            x_crop_left, x_crop_right = center_point[0] - 128, center_point[0] + 128
            y_crop_top, y_crop_bottom = 1280, 1536
    elif not x_valid and y_valid:
        if x_left_valid:
            x_crop_left, x_crop_right = 1280,1536 
            y_crop_top, y_crop_bottom = center_point[1] - 128, center_point[1] + 128 
        if x_right_valid:
            x_crop_left, x_crop_right = 0, 256
            y_crop_top, y_crop_bottom =center_point[1] - 128, center_point[1] + 128
    ##corner piece
    else:
        ##top left
        if x_right_valid and y_bottom_valid:
            x_crop_left, x_crop_right = 0,256
            y_crop_top, y_crop_bottom =0,256
        ##top right
        if x_left_valid and y_bottom_valid:
            x_crop_left, x_crop_right = 1280,1536
            y_crop_top, y_crop_bottom = 0,256
        ##bottom left
        if x_right_valid and y_top_valid:
            x_crop_left, x_crop_right = 0,256
            y_crop_top, y_crop_bottom = 1280,1536 
        ##bottom right
        if x_left_valid and y_top_valid:
            x_crop_left, x_crop_right = 1280,1536
            y_crop_top, y_crop_bottom = 1280,1536
    center_cropped = img[y_crop_top:y_crop_bottom, x_crop_left:x_crop_right]
    return center_cropped

class Net(nn.Module):
    """
    The CNN architecture used for filtering, first shown in: 
    https://www.biorxiv.org/content/10.1101/2021.03.12.435050v1
    """
    def __init__(self, fc_nodes=512, num_classes=3, dropout=0.5):
        super(Net, self).__init__()
        self.drop = 0.2
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(16, 32, 3, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(32, 48, 3, padding=1),
                                      nn.BatchNorm2d(48),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(48, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(64, 80, 3, padding=1),
                                      nn.BatchNorm2d(80),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(80, 96, 3, padding=1),
                                      nn.BatchNorm2d(96),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),)
        self.classifier = nn.Sequential(nn.Linear(96 * 4 * 4, num_classes))
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def getClassPreds(img_256, model, data_transforms):
    """
    Given a 256 x 256 pixel cv2 image, and a pytorch model, returns the 3 class prediction of the image
    """    
    img_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)##openCV imread loads as BGR! 
    img_as_img = Image.fromarray(img_256.astype('uint8'), 'RGB')
    pixels_2 = list(img_as_img.getdata())
    img_as_img = data_transforms(img_as_img)
    outputs = model(img_as_img.view(1, 3, 256, 256).cuda())
    predictions = torch.sigmoid(outputs).type(torch.cuda.FloatTensor).tolist()[0]
    predictions = tuple(predictions)
    return predictions

def filterMapToGetCoredOrCAA(mapp):
    """
    Will filter a mapp of  map of key: 1536 image name to value: list of (bbox coordinate, class label) tuples
    to only include 1536 images that have either a cored or CAA bbox prediction present 
    """
    new_mapp = {}
    for img in mapp:
        for bbox, class_preds in mapp[img]:
            if class_preds[0] >= .5 or class_preds[2] >= 0.5:
                new_mapp[img] = mapp[img]
                continue
    return new_mapp

def filterMapToGetCAATypes(mapp):
    """
    Will filter a mapp of key: 1536 image name to value: list of (bbox coordinate, CAA class labels: leptomeningeal,parenchymal,capillary,flag,notsure,notrecommended,vonsattel0,vonsattel1,vonsattel2,vonsattel3,vonsattel4) tuples
    to only include 1536 images that have either a CAA bbox prediction present 
    """
    new_mapp = {}
    for img in mapp:
        for bbox, class_preds in mapp[img]:
            if class_preds[0] >= .5 or class_preds[1] >= 0.5 or class_preds[2] >= 0.5:
                new_mapp[img] = mapp[img]
                continue
    return new_mapp

def seedSamplesFolder(sample_size=25):
    """
    Take SAMPLE_SIZE images from valid.txt file and copy them over to data/amyloid_samples/ directory
    """
    shutil.rmtree("data/amyloid_samples/")
    os.mkdir("data/amyloid_samples/")
    images = os.listdir("data/custom/images/")
    with open("data/custom/valid.txt","r") as f:
        lines = f.readlines()
        lines = lines[0:sample_size]
        for line in lines:
            line = line.replace("\n", "")
            img_name = line[line.rfind("/") + 1: ]
            shutil.copy(line, "data/amyloid_samples/" + img_name)

def seedTestFolder():
    """
    seeds data/amyloid_test/ which will contain all of the raw un-annotated test images
    """
    shutil.rmtree("data/amyloid_test/")
    os.mkdir("data/amyloid_test/")
    images = os.listdir("data/custom/images/")
    with open("data/custom/valid.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            img_name = line[line.rfind("/") + 1: ]
            shutil.copy(line, "data/amyloid_test/" + img_name)    

def seedTrainFolder():
    """
    seeds data/amyloid_train/ which will contain all of the raw un-annotated train images
    """
    shutil.rmtree("data/amyloid_train/")
    os.mkdir("data/amyloid_train/")
    images = os.listdir("data/custom/images/")
    with open("data/custom/train.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            img_name = line[line.rfind("/") + 1: ]
            shutil.copy(line, "data/amyloid_train/" + img_name)    

def combineOverlappingBboxes(pickle_name=None, mapp=None):
    """
    PICKLE_NAME specifies the name of the mapp to load, with key: imagename, value: list of tuples(bbox, class labels of bbox),
    alternatively we can pass in the mapp directly 
    if a bbox overlaps with any other bbox (and shares either a cored + or CAA + label),
    then combine them into a single bbox retaining the union of the two bbox labels 
    returns mapp
    """
    if pickle_name != None: ##if loading a pickle file
        mapp = pickle.load(open(pickle_name, "rb"))
        ##first convert lists to hashable types...
        for img_name in mapp.keys():
            new_list = []
            for bbox, label in mapp[img_name]:
                new_list.append((tuple(bbox), tuple(label)))
            mapp[img_name] = new_list
    z = 0
    for img_name in mapp.keys():
        z += 1 
        tuple_set = set(mapp[img_name]) 
        keep_going = True ##we need to iterate over our set pairwise multiple times because combining bboxes can result in new possible overlaps with other bboxes
        removed = set()
        while keep_going: ##will be marked False initially, and if still False by end of pairwise iteration, then we know no more combinations can be made
            keep_going = False
            tuple_list = list(tuple_set)
            for i in range(0, len(tuple_list)):
                for j in range(i + 1, len(tuple_list)): ##have i != j by default and don't look at redundancies
                    bbox_i, label_i = tuple_list[i]
                    bbox_j, label_j = tuple_list[j]
                    ##if not (both are cored or both are CAA)  then continue
                    if not ((label_i[0] == 1 == label_j[0]) or (label_i[2] == 1 == label_j[2])):
                        continue
                    if (bbox_i, label_i) in removed or (bbox_j, label_j) in removed:
                        continue 
                    combined = combineIfOverlapping(bbox_i, bbox_j)
                    combined_label = label_i or label_j ##want to always preserve a label if present, so use OR, i.e. (1,0,1) or (0,0,1) -> (1, 0, 1)    
                    ##if bboxes can be combined AND this wasn't already combined before
                    if combined[0] == True and (combined[1], combined_label) not in tuple_set:
                        tuple_set.add((combined[1], combined_label)) ##add combined bbox
                        tuple_set.remove((bbox_i, label_i)) ##remove two smaller bbboxes from set, discard does not raise error if DNE
                        tuple_set.remove((bbox_j, label_j))
                        removed.add((bbox_i, label_i))
                        removed.add((bbox_j, label_j))
                        keep_going = True ##with new combined bbox, there is potential for more combinations now, so iterate again 
            if keep_going == False: ##break out of while loop
                break
        mapp[img_name] = list(tuple_set) ##replace mapp entry with new list of combined bboxes
    return mapp

def combineIfOverlapping(bbox1, bbox2):
    """
    given bboxes bbox1 and bbox2 (format: [x,y,w,h] for each bbox)
    If there is overlap, will return (True, new bbox coordinate that encompasses both)
    else will return False, -1
    """
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    points_1 = set()
    for x_i in range(x1, x1 + w1):
        for y_i in range(y1, y1 + h1):
            points_1.add((x_i, y_i))
    points_2 = set()
    for x_i in range(x2, x2 + w2):
        for y_i in range(y2, y2 + h2):
            points_2.add((x_i, y_i))
    
    intersection = points_1.intersection(points_2)
    if len(points_1.intersection(points_2)) == 0:
        return False, -1 
    else:
        ##construct new bbox, find top left corner and bottom right corner, and create box from that
        ##top left:
        super_set = points_1.union(points_2)
        furthest_left = 1000000
        furthest_right = -1
        furthest_top = 1000000
        furthest_bottom = -1
        for point in super_set:
            if point[0] < furthest_left:
                furthest_left = point[0]
            if point[0] > furthest_right:
                furthest_right = point[0]
            if point[1] < furthest_top:
                furthest_top = point[1]
            if point[1] > furthest_bottom:
                furthest_bottom = point[1]
        new_bbox = furthest_left, furthest_top, furthest_right - furthest_left, furthest_bottom - furthest_top
        return True, new_bbox

def mergeDetections(detections):
    """
    Function to merge bboxes given DETECTIONS (a 2D tensor of [x1, y1, x2, y2, conf, cls_conf, cls_pred] entries),
    cls_pred = 1 for cored, 0 for CAA,
    e.g. detections tensor([[6.3632e+02, 5.6398e+02, 7.5309e+02, 6.7693e+02, 9.6941e-01, 9.9990e-01,
         1.0000e+00],
        [1.7516e+01, 1.2104e+03, 1.2656e+02, 1.3061e+03, 9.5140e-01, 9.9925e-01,
         1.0000e+00],
        [4.3764e+00, 4.7533e+00, 6.4293e+01, 4.3032e+01, 9.0409e-01, 9.8825e-01,
         1.0000e+00],
        [1.0779e+03, 1.5156e+03, 1.1419e+03, 1.5329e+03, 8.5024e-01, 9.8666e-01,
         1.0000e+00]])
    Will return a 2D tensor of entries, but joining any entries that have the same cls_pred
    """
    list_values = detections.tolist() ##convert to list 
    list_values = [tuple(entry) for entry in list_values] ##convert each sublist to tuple
    tuple_set = set(list_values) ##make a set
    keep_going = True ##we need to iterate over our set pairwise multiple times because combining bboxes can result in new possible overlaps with other bboxes
    removed = set()
    while keep_going: ##will be marked False initially, and if still False by end of pairwise iteration, then we know no more combinations can be made
        keep_going = False
        tuple_list = list(tuple_set)
        for i in range(0, len(tuple_list)):
            for j in range(i + 1, len(tuple_list)): ##have i != j by default and don't look at redundancies
                entry_i = tuple_list[i]
                width_i, height_i = entry_i[2] - entry_i[0], entry_i[3] - entry_i[1]
                bbox_i = (int(entry_i[0]), int(entry_i[1]), int(width_i), int(height_i))
                conf_i = entry_i[4]
                cls_conf_i = entry_i[5]
                label_i = entry_i[6]
                entry_j = tuple_list[j]
                width_j, height_j = entry_j[2] - entry_j[0], entry_j[3] - entry_j[1]
                bbox_j = (int(entry_j[0]), int(entry_j[1]), int(width_j), int(height_j))
                conf_j = entry_j[4]
                cls_conf_j = entry_j[5]
                label_j = entry_j[6]
                ##if not (both are cored or both are CAA)  then continue
                if not ((label_i == 1 == label_j) or (label_i == 0 == label_j)):
                    continue
                if entry_i in removed or entry_j in removed:
                    continue 
                is_combinable, new_bbox = combineIfOverlapping(bbox_i, bbox_j)
                if is_combinable:
                    new_entry = (new_bbox[0], new_bbox[1], new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3], min(conf_i, conf_j), min(cls_conf_i, cls_conf_j), label_i)
                    ##if bboxes can be combined AND this wasn't already combined before
                    if new_entry not in tuple_set:
                        tuple_set.add(new_entry) ##add combined bbox
                        tuple_set.remove(entry_i) ##remove two smaller bbboxes from set, discard does not raise error if DNE
                        tuple_set.remove(entry_j)
                        removed.add(entry_i)
                        removed.add(entry_j)
                        keep_going = True ##with new combined bbox, there is potential for more combinations now, so iterate again 
        if keep_going == False: ##break out of while loop
            break
    ##now put back to original format of detections
    l = [list(x) for x in tuple_set]
    l = torch.as_tensor(l)
    return l

def filterDetectionsByCAAModel(img_name, detections, classes):
    """
    Given the IMG_NAME of a 1536 x 1536 image, and the bbox DETECTIONS for that image,
    will filter these detections to get rid of any bbox that is CAA positive, but the consensus of 2 model predicts as CAA negative
    Returns the filtered detections
    """
    filtered_detections = []
    model = torch.load("pickles/model_all_fold_3_thresholding_2_l2.pkl")
    norm = np.load("pickles/normalization.npy", allow_pickle=True).item()

    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
    img = cv2.imread(img_name)

    list_values = detections.tolist() ##convert to list 
    list_values = [tuple(entry) for entry in list_values] ##convert each sublist to tuple
    tuple_set = set(list_values) ##make a set

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in tuple_set:
        img_256 = get256Img(img, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        cored_pred, diffuse_pred, CAA_pred = getClassPreds(img_256, model, data_transforms)
        if classes[int(cls_pred)] == "CAA" and CAA_pred <= 0.5:
            continue
        else:
            filtered_detections.append((x1, y1, x2, y2, conf, cls_conf, cls_pred))
        ##now put back to original format of detections
    l = [list(x) for x in filtered_detections]
    l = torch.as_tensor(l)
    return l

def writeCAADetectionsToPickle(img_name, detections):
    """
    Given DETECTIONS (a 2D tensor of [x1, y1, x2, y2, conf, cls_conf, cls_pred] entries),
    cls_pred = 1 for cored, 0 for CAA,
    Will write the CAA detections to a pickle called CAA_detections.pkl
    CAA_detections.pkl should already be instantiated, and this function simply writes to it / adds to it
    """
    CAA_detections = pickle.load(open("pickles/CAA_detections.pkl", "rb"))
    ##need image name
    img_name = img_name[img_name.rfind("/") + 1:]
    list_values = detections.tolist() ##convert to list 
    list_values = [tuple(entry) for entry in list_values] ##convert each sublist to tuple
    tuple_set = set(list_values) ##make a set
    new_bbox_labels = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in tuple_set:
        label_idx = 0 if classes[int(cls_pred)] == "CAA" else 1
        x_center = (int(x2) - int(x1)) / float(2)
        y_center = (int(y2) - int(y1)) / float(2)
        width = int(x2) - int(x1)
        height = int(y2) - int(y1)
        if classes[int(cls_pred)] == "CAA":
            new_bbox_labels.append((label_idx, x_center, y_center, width, height))
    if img_name not in CAA_detections:
        CAA_detections[img_name] = new_bbox_labels
    else:
        CAA_detections[img_name] += new_bbox_labels
    pickle.dump(CAA_detections, open("pickles/CAA_detections.pkl", "wb"))

def get_gpu_memory_map():
    """
    Get the current gpu usage.
    pulled from https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    Returns usage dict: keys are device ids as integers, values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


