"""
Analyses for:
    Model version 1 and version 2 benchmarking
    The prospective validation with four neuropathologists
"""
from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
import os
import sys
import time
import datetime
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import pickle
import json
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from core import *

def runModelOnValidationImages(val_type="prospective"):
    """
    if val_type == "phase1" will run checkpoints_modelv1/yolov3_ckpt_157.pth model over data/amyloid_test
    if val_type == "phase2" will run checkpoints/yolov3_ckpt_105.pth model over data/amyloid_test
    if val_type == "prospective" will run checkpoints/yolov3_ckpt_105.pth model over prospective_validation_images/
    Saves a dictionary called {val_type}_validation_predictions.pkl
    with key: image name, value: list of (bbox coordinate, class) tuples
    """
    validation_predictions_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
    if val_type == "phase2" or val_type == "prospective":
        model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_105.pth"))
    if val_type == "phase1":
        model.load_state_dict(torch.load("checkpoints_modelv1/yolov3_ckpt_157.pth"))
    model.eval() 
    if val_type == "prospective":
        image_folder = "prospective_validation_images/"
    else:
        image_folder = "data/amyloid_test/"
    dataloader = DataLoader(
    ImageFolder(image_folder, transform= \
        transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)])),
    batch_size=8,
    shuffle=False,
    num_workers=12,
    ) 
    classes = load_classes("data/custom/classes.names")  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 0.8, 0.4) ##use conf threshold of 0.8 for detection 
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        if val_type == "prospective":
            save_path = path.replace("prospective_validation_images/", "")   
        else:
            save_path = path.replace("data/amyloid_test/", "")   
        validation_predictions_dict[save_path] = []
        img = np.array(Image.open(path))
        if detections is None:
            continue
        detections = rescale_boxes(detections, 416, img.shape[:2])
        if val_type != "phase1": ##for phase 2 and prospective, we merge final detections and also apply the consensus CAA filter
            detections = mergeDetections(detections) 
            detections = filterDetectionsByCAAModel(path, detections, classes)
        if len(detections) == 0: ##it's possible that we removed all of the detections after filtering
            continue 
        else:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                validation_predictions_dict[save_path].append(({'x1': x1.item(), 'x2': x2.item(), 'y1': y1.item(), 'y2': y2.item(), 'conf': conf.item(), 'cls_conf': cls_conf.item(), 'cls_pred': cls_pred.item()}, classes[int(cls_pred)]))
    pickle.dump(validation_predictions_dict, open("pickles/{}_validation_predictions.pkl".format(val_type), "wb"))

def convertPreProspectiveAnnotationsToPickle(phase="phase1"):
    """
    If phase == "phase1":
    will take the labels found in original_data/labels/ and convert to pickle format with key: imagename, value: list of (coord dictionary x1, y1, x2, y2, class_label as string) tuples
    If phase == "phase2":
    will take the labels found in data/custom/labels/
    Saves dictionary called pickles/{phase}_annotations.pkl
    """  
    if phase == "phase1":
        labels_dir = "original_data/labels/"
    if phase == "phase2":
        labels_dir = "data/custom/labels/"
    annotation_dict = {} ##key: imagename, value: (coord dictionary x1, y1, x2, y2, class_label as string)
    for label_txt in os.listdir(labels_dir):
        img_name = label_txt.replace(".txt", ".jpg")  
        annotation_dict[img_name] = []          
        with open(labels_dir + label_txt) as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                entries = line.split(" ")
                idx, x_center, y_center, width, height = [float(x) for x in entries]
                x_center, y_center, width, height = x_center*1536, y_center*1536, width*1536, height*1536 ##convert to pixel space
                x1 = int(x_center - (width / float(2)))
                x2 = int(x_center + (width / float(2)))
                y1 = int(y_center - (height / float(2)))
                y2 = int(y_center + (height / float(2)))
                class_label = "Cored" if idx == 1 else "CAA"
                annotation_dict[img_name].append(({"x1": x1, "x2":x2, "y1":y1, "y2":y2}, class_label))
    pickle.dump(annotation_dict, open("pickles/{}_annotations.pkl".format(phase), "wb"))                
                
def compareAnnotationsToPredictions(iou_threshold=0.5, annotator="NP1", val_type="prospective"):
    """
    Loads annotations and predictions dictionaries with format key: image name, value:  value: list of (bbox coordinate, class) tuples
    Saves a table df PRC_tables/{}_PRC_table_{}_iou_{}_{}.csv.format(val_type, annotator, iou_threshold, amyloid_class)
    Also saves a dictionary called img_precision_maps/{}_precision_img_map_{}_{}.format(val_type, annotator, iou_threshold) mapping key: imagename, to value: precision  
    """
    if val_type == "prospective":
        if annotator == "consensus":
            annotations = pickle.load(open("prospective_annotations/consensus_annotations_iou_thresh_{}.pkl".format(round(iou_threshold, 2)), "rb"))
        else:
            annotations = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator), "rb"))
    if val_type == "phase1" or val_type == "phase2":
        annotations = pickle.load(open("pickles/{}_annotations.pkl".format(val_type), "rb"))
    predictions = pickle.load(open("pickles/{}_validation_predictions.pkl".format(val_type), "rb"))
    ##iterate over each image and compute success metrics
    table_cored = [] ##will be a table for each Cored model detection of confidence, TP, FP, cumTP, cumFP, Precision, Recall
    table_CAA = []
    num_CAA_labels = 0
    num_Cored_labels = 0
    precision_img_map_cored = {}
    precision_img_map_CAA = {}
    for img_name in predictions.keys(): ##iterate over prediction keys because this is a subset of annotation keys for phase1 and phase2
        outputs = [] ##list of model detections for this particular image
        labels = [] ## list of annotations for this particular image
        for entry in predictions[img_name]: ##can be multiple boxes
            if len(entry) == 0:
                continue
            dictionary, class_label = entry[0], entry[1]
            x1 = dictionary['x1']
            y1 = dictionary['y1']
            x2 = dictionary['x2']
            y2 = dictionary['y2']
            conf = dictionary['conf']
            cls_conf = dictionary['cls_conf']
            cls_pred = dictionary['cls_pred']
            outputs.append([x1, y1, x2, y2, conf, cls_conf, cls_pred])
        for entry in annotations[img_name]:
            dictionary, class_label  = entry[0], entry[1]
            annotation_class = 1 if class_label == "Cored" else 0
            x1 = dictionary['x1']
            y1 = dictionary['y1']
            x2 = dictionary['x2']
            y2 = dictionary['y2']
            labels.append([x1, y1, x2, y2, annotation_class])  
            if annotation_class == 1:
                num_Cored_labels += 1
            if annotation_class == 0:
                num_CAA_labels += 1
        
        ##from list of model detections, determine which are TP and which are FP, and add them to the table with headers conf, TP, FP
        TPs = getTPs(outputs, labels, iou_threshold, Pascal_VOC_scheme=True)
        cored_TP, cored_FP, CAA_TP, CAA_FP = 0,0,0,0 ##per image basis
        for i in range(0, len(TPs)):
            detection = outputs[i]
            conf = detection[4] #model confidence
            cls_pred = detection[6]
            if TPs[i] == 1: #TP
                if cls_pred == 1: ##TP and cored
                    table_cored.append((conf, 1, 0))
                    cored_TP += 1
                else: ##TP and CAA
                    table_CAA.append((conf, 1, 0))
                    CAA_TP += 1
            else: #FP 
                if cls_pred == 1: ##FP and cored
                    table_cored.append((conf, 0, 1))
                    cored_FP += 1
                else:##FP and CAA
                    table_CAA.append((conf, 0, 1))
                    CAA_FP += 1
        if cored_TP + cored_FP > 0: #must have a prediction to be valid
            precision_img_map_cored[img_name] = cored_TP / (float(cored_TP + cored_FP))
        else:
            precision_img_map_cored[img_name] = -1
        if CAA_TP + CAA_FP > 0:
            precision_img_map_CAA[img_name] = CAA_TP / (float(CAA_TP + CAA_FP))
        else:
            precision_img_map_CAA[img_name] = -1
   
    pickle.dump(precision_img_map_cored, open("pickles/img_precision_maps/{}_precision_img_map_Cored_{}_{}.pkl".format(val_type, annotator, round(iou_threshold,2)), "wb"))
    pickle.dump(precision_img_map_CAA, open("pickles/img_precision_maps/{}_precision_img_map_CAA_{}_{}.pkl".format(val_type, annotator, round(iou_threshold,2)), "wb"))

    ##make the table into a dataframe, and compute the relevant columns
    for amyloid_class in ["Cored", "CAA"]:
        if amyloid_class == "Cored":
            table = table_cored
            all_ground_truths = num_Cored_labels
        else:
            table = table_CAA
            all_ground_truths = num_CAA_labels

        table = sorted(table, key=lambda x:x[0]) ##sort table in descending order of model confidence
        table.reverse()
        table = pd.DataFrame.from_records(table, columns =['Conf', 'TP', 'FP'])

        ##add columns to the table cumTP, cumFP
        TPs = list(table['TP'])
        cumTP = []
        summ = 0
        for i in range(0, len(TPs)):
            if TPs[i] == 1:
                summ += 1            
            cumTP.append(summ)
        table['cumTP'] = cumTP

        FPs = list(table['FP'])
        cumFP = []
        summ = 0 
        for i in range(0, len(FPs)):
            if FPs[i] == 1:
                summ += 1            
            cumFP.append(summ)
        table['cumFP'] = cumFP

        ## now add columns precision, recall
        precision = []
        recall = []
        for i in range(0, len(cumTP)):
            precision.append(cumTP[i] / float(i+ 1))
            recall.append(cumTP[i] / float(all_ground_truths))
        table['Precision'] = precision
        table['Recall'] = recall

        table.to_csv("PRC_tables/{}_PRC_table_{}_iou_{}_{}.csv".format(val_type, annotator, round(iou_threshold, 1), amyloid_class))

def plotAPsForPhases(plot_phases_separately=True):
    """
    Plot APs vs IOU for model v1 and model v2
    """
    iou_thresholds = list(np.arange(0.1, 1.0, 0.1))
    amyloid_classes = ["Cored", "CAA"]
    phases = ["phase1", "phase2"]
    AP_map = {phase: {amyloid_class: {thresh: -1 for thresh in iou_thresholds} for amyloid_class in amyloid_classes} for phase in phases}
    for phase in phases:
        for amyloid_class in amyloid_classes:
            for iou_threshold in iou_thresholds:
                df = pd.read_csv("PRC_tables/{}_PRC_table_{}_iou_{}_{}.csv".format(phase, phase, round(iou_threshold, 2), amyloid_class))
                precision, recall, thresholds = precision_recall_curve(list(df['TP']),list(df['Conf']))
                AP = average_precision_score(list(df['TP']),list(df['Conf']))
                AP_map[phase][amyloid_class][iou_threshold] = AP
    for phase in phases:
        fig, ax = plt.subplots()
        for amyloid_class in amyloid_classes:
            x = iou_thresholds
            y = [AP_map[phase][amyloid_class][thresh] for thresh in x]
            ax.plot(x, y, linestyle='-', marker='o', label=amyloid_class)    
        plt.ylim([0.0, 1.0])
        modelv = "One" if phase == "phase1" else "Two"
        plt.title("Model Version {} Average Precisions".format(modelv))
        ax.set_xlabel("IOU Threshold", fontname="Times New Roman", fontsize=12)
        ax.set_ylabel("Average Precision", fontname="Times New Roman", fontsize=12)
        plt.xticks(np.arange(0.1, 1.0, .1))
        #Shrink current axis and place legend outside plot, top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
        ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.40))
        plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
        plt.savefig("figures/PRC_cumulative_plot_{}.png".format(phase), dpi=300)

def plotAPsForProspective(plotAvgOverlay=True):
    """
    Will plot average precisions for prospective validation at different IOU thresholds for each (model, annotator) pair
    if plotAvgOverlay, will plot the average and std fill of comparing each annotator relative to each other 
    """
    annotators = ["consensus"] + ["NP{}".format(i) for i in range(1, 5)]
    iou_thresholds = list(np.arange(0.1, 1.0, 0.1))
    amyloid_classes = ["Cored", "CAA"]
    AP_map = {annotator: {amyloid_class: {thresh: -1 for thresh in iou_thresholds} for amyloid_class in amyloid_classes} for annotator in annotators}
    for annotator in annotators:
        for amyloid_class in amyloid_classes:
            for iou_threshold in iou_thresholds:
                df = pd.read_csv("PRC_tables/prospective_PRC_table_{}_iou_{}_{}.csv".format(annotator, round(iou_threshold, 2), amyloid_class))
                precision, recall, thresholds = precision_recall_curve(list(df['TP']),list(df['Conf']))
                AP = average_precision_score(list(df['TP']),list(df['Conf']))
                AP_map[annotator][amyloid_class][iou_threshold] = AP
    pickle.dump(AP_map, open("pickles/APs_per_annotator.pkl", "wb"))

    color_dict = {"NP1": "#ff8800", "NP2": "#03ebfc", "NP3":"#fc039d", "NP4":"#23ba28", "merged": "#51169e", "consensus": "#000000"}
    for amyloid_class in amyloid_classes:
        fig, ax = plt.subplots()
        for annotator in annotators:
            x = iou_thresholds
            y = [AP_map[annotator][amyloid_class][thresh] for thresh in x]
            ax.plot(x, y, linestyle='-', marker='o', label=annotator, color=color_dict[annotator])    
           
        plt.ylim([0.0, 1.0])
        plt.title("{} Average Precisions\nper Annotator".format(amyloid_class))
        ax.set_xlabel("IOU Threshold", fontname="Times New Roman", fontsize=12)
        ax.set_ylabel("Average Precision", fontname="Times New Roman", fontsize=12)
        plt.xticks(np.arange(0.1, 1.0, .1))

        if plotAvgOverlay:
            summary = pickle.load(open("pickles/annotatorPrecisionRelativeToEachOtherSummary.pkl", "rb"))
            x, y_global_avg, y_global_std = summary[amyloid_class]["x"], summary[amyloid_class]["avg"], summary[amyloid_class]["std"]
            plt.plot(x,y_global_avg, linestyle='--', marker='.')#, label="Average of AnnotatorsRelative to Each Other")
            plt.fill_between(x, y_global_avg - y_global_std, y_global_avg + y_global_std, alpha=0.5)

        #Shrink current axis and place legend outside plot, top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
        ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.40))
        plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
        plt.savefig("figures/PRC_cumulative_plot_{}.png".format(amyloid_class), dpi=300)

def findLowPerformanceImages(amyloid_class, annotator, iou_threshold=0.5):
    """
    Will print a sorted list of images with worst precision to images with greatest precision
    for given amyloid_class, evaluated against annotator and an iou_threshold
    """
    precision_img_map = pickle.load(open("pickles/img_precision_maps/precision_img_map_{}_{}_{}.pkl".format(amyloid_class, annotator, round(iou_threshold,2)), "rb"))
    sorted_list = sorted(precision_img_map.items(), key=lambda item: item[1])
    sorted_list = [x for x in sorted_list if x[1] != -1] ##exclude images with no detections
    print(sorted_list)

def getAnnotationOverlaps(annotator="NP1", iou_threshold=0.5):
    """
    Check how many overlaps ANNOTATOR has with himself / herself
    i.e. how many of annotators bounding boxes overlap with another bbox from same annotator
    Returns number of overlaps
    """
    if annotator == "consensus":
        annotation = pickle.load(open("prospective_annotations/consensus_annotations_iou_thresh_0.5.pkl", "rb")) ##use the 0.5 iou consensus annotation
    else:
        annotation = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator), "rb"))
    overlaps_with_same_class_label = 0
    for img_name in annotation.keys():
        entries = annotation[img_name] ##list of dictionary, class_label tuples
        ##convert format to list of [x1, y1, x2, y2, class_label]
        entries = [[dictionary['x1'], dictionary['y1'], dictionary['x2'], dictionary['y2'], class_label] for dictionary, class_label in entries]
        for i in range(0, len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[i][4] == entries[j][4] and IOU(entries[i][0:4], entries[j][0:4]) >= iou_threshold: ##if class labels are  the same
                    overlaps_with_same_class_label += 1
    print("number of overlaps with the same class: ", overlaps_with_same_class_label)
    return overlaps_with_same_class_label
        
def getInterraterAgreement(iou_threshold=0.50):
    """
    Interrater agreement between all annotators 
    measured as the number of # of overlaps / (overlaps + everything else)
    We can't just treat one as pred and one as label and derive PRC because no confidence scores for preds!

    Method: from set of a1 annotations and a2 annotations, find any that overlap (by IOU threshold) and being sure to restrict any bbox to be a part of at most 1 overlap, 
    any that are left in a1's set only had a1 label it by definition
    any that are left in a2's set only had a2 label it by definition
    Build a list of annotations for each plaque considered by either a1 or a2, with 1 indicating annotator labeled or would label (by IOU) as (+), 0 for negative 

    E.g. 
    Suppose for a given image, a1 and a2 both gave a total of two labels
    suppose there are 3 plaques, A B C, from the union of a1 and a2
    a1 labeled A and B, but didn't label C
    a2 labeled B and C, but didn't label A
    both labeled B (because there exists a bbox label with IOU>threshold)
    then we will make annotations like this:
                     A B C   
    a1 annotations = 1 1 0
    a2 annotations = 0 1 1
    Agreement accuracy = 1/3
    """

    annotators = ["NP{}".format(i) for i in range(1, 5)]
    pairs = [(a1, a2) for a1 in annotators for a2 in annotators if a1 != a2]
    pairs = [] 
    for a1 in annotators:
        for a2 in annotators:
            if a1 != a2 and (a1, a2) not in pairs and (a2, a1) not in pairs:
                pairs.append((a1, a2))
    pair_map = {pair: {amyloid_class: -1 for amyloid_class in ["Cored", "CAA"]} for pair in pairs} 
    for annotator1, annotator2 in pairs:

        annotation1 = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator1), "rb"))
        annotation2 = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator2), "rb"))
        
        ##to be used for testing correctness of method: get annotation counts and set up counts for overlaps for each class
        a1_cored_count = sum([1 for img_name in annotation1 for dictionary, class_label in annotation1[img_name] if class_label == "Cored"])
        a1_CAA_count = sum([1 for img_name in annotation1 for dictionary, class_label in annotation1[img_name] if class_label == "CAA"])
        a2_cored_count = sum([1 for img_name in annotation2 for dictionary, class_label in annotation2[img_name] if class_label == "Cored"])
        a2_CAA_count = sum([1 for img_name in annotation2 for dictionary, class_label in annotation2[img_name] if class_label == "CAA"])
        overlaps_counts = {"Cored": 0, "CAA":0}
       
        a1_final_annotations = {"Cored": [], "CAA":[]} ##list of 1s and 0s, 1 indicating yes I label the plaque as (+) 
        a2_final_annotations = {"Cored": [], "CAA":[]}

        for img_name in annotation1.keys():
            entries1 = annotation1[img_name]
            ##reformat entries
            entries1 = [[dictionary['x1'], dictionary['y1'], dictionary['x2'], dictionary['y2'], class_label] for dictionary, class_label in entries1]
            entries2 = annotation2[img_name]
            entries2 = [[dictionary['x1'], dictionary['y1'], dictionary['x2'], dictionary['y2'], class_label] for dictionary, class_label in entries2]

            ##filter entries to only contain entries pertaining to specific class
            for amyloid_class in ["Cored", "CAA"]:
                class_entries_1 = [x for x in entries1 if x[4] == amyloid_class]
                class_entries_2 = [x for x in entries2 if x[4] == amyloid_class]
                ##count of overlaps between a1 and a2 for specific class
                overlapping_boxes = []
                overlaps = 0
                for entry1 in class_entries_1:
                    for entry2 in class_entries_2:
                        ##want to make sure that if a box was counted in an overlap before, we don't allow it to count again as an overlap
                        if IOU(entry1[0:4], entry2[0:4]) >= iou_threshold and entry1 not in overlapping_boxes and entry2 not in overlapping_boxes: 
                            overlapping_boxes.append(entry1)
                            overlapping_boxes.append(entry2)
                            overlaps += 1
                ##1 for overlaps, 1 for plaques that a1 labeled but a2 did not, 0 for plaques that a2 labeled but a1 did not
                a1_final_annotations[amyloid_class] += [1] * overlaps + [1] * (len(class_entries_1) - overlaps) + [0] * (len(class_entries_2) - overlaps)
                ##1 for overlaps, 0 for plaques that a1 labeled but a2 did not, 1 for plaques that a2 labeled but a1 did not
                a2_final_annotations[amyloid_class] += [1] * overlaps + [0] * (len(class_entries_1) - overlaps) + [1] * (len(class_entries_2) - overlaps)
                overlaps_counts[amyloid_class] += overlaps

        ##test to make sure everything is accounted for
        assert(a1_cored_count + a2_cored_count - overlaps_counts["Cored"] == len(a1_final_annotations["Cored"]) == len(a2_final_annotations["Cored"]))
        assert(a1_CAA_count + a2_CAA_count - overlaps_counts["CAA"] == len(a1_final_annotations["CAA"]) == len(a2_final_annotations["CAA"]) )
        for amyloid_class in ["Cored", "CAA"]:
            pair_map[(annotator1, annotator2)][amyloid_class] = getAccuracy(a1_final_annotations[amyloid_class], a2_final_annotations[amyloid_class])
    pickle.dump(pair_map, open("pickles/annotator_interrater_map_iou_{}.pkl".format(iou_threshold), "wb"))

def plotInterraterAgreement(iou_threshold=0.5):
    """
    Plots a heatmap from annotator_interrater_map.pkl
    """
    pair_map = pickle.load(open("pickles/annotator_interrater_map_iou_{}.pkl".format(iou_threshold), "rb"))
    annotators = ["NP{}".format(i) for i in range(1, 5)]
    for amyloid_class in ["Cored", "CAA"]:
        grid = []
        for a1 in annotators:
            l = []
            for a2 in annotators:
                if a1 == a2:
                    l.append(1.0)
                else:
                    try: 
                        l.append(pair_map[(a1, a2)][amyloid_class])
                    except:
                        l.append(pair_map[(a2, a1)][amyloid_class])
            grid.append(l)
        
        print("Average agreement for {}:{}, std:{}".format(amyloid_class, np.mean([x for x in l if x != 1.0]),np.std([x for x in l if x != 1.0])))
        fig, ax = plt.subplots()
        im = ax.imshow(grid,vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(annotators)))
        ax.set_yticks(np.arange(len(annotators)))
        plt_labels = annotators
        ax.set_xticklabels(plt_labels,fontsize=11)
        ax.set_yticklabels(plt_labels,fontsize=11)
        for i in range(len(annotators)):
            for j in range(len(annotators)):
                text = ax.text(j, i, str(round(grid[i][j], 2)), ha="center", va="center", color="black", fontsize=11)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=11)
        fig.tight_layout()
        ax.set_title("{} Interrater Agreement Accuracy, IOU = {}".format(amyloid_class, round(iou_threshold,2)), fontsize=12)
        plt.savefig("figures/interrater_agreement_accuracy_{}_{}.png".format(amyloid_class, round(iou_threshold,2)), dpi=300)

def plotPRC(annotator="NP1",val_type="prospective", separate_legend=True):
    """
    Given a PRC table and an annotator to determine which annotation set to use, will plot a PRC curve with different IOU thresholds
    if separate_legend, will plot the legend separately
    """
    for amyloid_class in ["Cored", "CAA"]:
        if separate_legend:
            fig = plt.figure("main figure")
            legendFig = plt.figure("legend plot")
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
        lines = []
        labels = []
        for iou_threshold in [.2, .4, .6, .8]:
            df = pd.read_csv("PRC_tables/{}_PRC_table_{}_iou_{}_{}.csv".format(val_type, annotator, round(iou_threshold, 2), amyloid_class))
            # precision = df['Precision']
            # recall = df['Recall']
            precision, recall, thresholds = precision_recall_curve(list(df['TP']),list(df['Conf']))
            AP = average_precision_score(list(df['TP']),list(df['Conf'])) ##TP is the list of labels 1 = (+), 0 = (-)
            line, = ax.plot(recall, precision, label="AP@{} = {}".format(iou_threshold, str(round(AP, 2))))
            lines.append(line)
            labels.append("AP@{} = {}".format(iou_threshold, str(round(AP, 3))))
        ax.set_xlabel("Recall", fontname="Times New Roman", fontsize=12)
        ax.set_ylabel("Precision", fontname="Times New Roman", fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("{} PRC, {}".format(amyloid_class, annotator))
        if separate_legend:
            fig.savefig("figures/PRC_plot_{}_{}.png".format(annotator, amyloid_class))
            legendFig.legend(lines, labels, fontsize=10, loc="center")
            legendFig.savefig("figures/PRC_plot_{}_{}_legend.png".format(annotator, amyloid_class))
            fig.clf()
            legendFig.clf()
        else:
            #Shrink current axis and place legend outside plot, top right corner 
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
            ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.35))
            plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
            plt.savefig("figures/PRC_plot_{}_{}.png".format(annotator, amyloid_class))

def getPrecisionsOfAnnotatorsRelativeToEachOther():
    """
    Merry go round - treat NP N as truth, have A != N evaluated against NP N's truth space,
    repeat for all N and all IOU and get precision 
    Will save a dictionary with key: amyloid_class, key: "NP{N}" (this outer key will represent who is ground truth), key: "NP{A}!=N}", key: iou_threshold, value: precision (TP/(TP + FP))
    """
    annotators = ["NP{}".format(i) for i in range(1, 5)]
    amyloid_classes = ["Cored", "CAA"]
    iou_thresholds = list(np.arange(0.1, 1.0, 0.1))
    precision_dict = {amyloid_class: {annotator1: {annotator2: {iou_threshold: -1 for iou_threshold in iou_thresholds} for annotator2 in annotators if annotator2 != annotator1} for annotator1 in annotators} for amyloid_class in amyloid_classes}
    for iou_threshold in iou_thresholds:
        ##annotator 1 will be fixed as ground truth 
        for annotator1 in annotators:
            annotation1 = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator1), "rb"))
            ##annotator 2 will be the 'model' predicting
            for annotator2 in annotators:
                if annotator2 == annotator1:
                    continue
                annotation2 =  pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator2), "rb"))
                for amyloid_class in amyloid_classes:
                    TPs, FPs = 0, 0 
                    for img_name in annotation1.keys():
                        entries1 = annotation1[img_name]
                        ##reformat and only keep class relevant to amyloid class entries 
                        entries1 = [[dictionary['x1'], dictionary['y1'], dictionary['x2'], dictionary['y2'], class_label] for dictionary, class_label in entries1 if class_label == amyloid_class]
                        entries2 = annotation2[img_name]
                        entries2 = [[dictionary['x1'], dictionary['y1'], dictionary['x2'], dictionary['y2'], class_label] for dictionary, class_label in entries2 if class_label == amyloid_class]
                        for entry2 in entries2:
                            isPos = False
                            for entry1 in entries1:
                                ##if same class label and IOU thresh met
                                if entry2[4] == entry1[4] and IOU(entry2[0:4], entry1[0:4]) >= iou_threshold:
                                    isPos = True
                                    entries1.remove(entry1) ##if an annotation from entry 1 is used to match a TP, we can't use it again because of PASCAL VOC 2012 paper precedent
                                    break
                            if isPos:
                                TPs += 1
                            else:
                                FPs += 1
                    precision = TPs / float(TPs + FPs)
                    precision_dict[amyloid_class][annotator1][annotator2][iou_threshold] = precision
    pickle.dump(precision_dict, open("pickles/precision_dict_annotators_relative_to_each_other.pkl", "wb"))

def plotPrecisionsOfAnnotatorsRelativeToEachOther(plotType="aggregate"):
    """
    Plots output of getPrecisionsOfAnnotatorsRelativeToEachOther
    if plotType == "aggregate", will plot an average with shaded std in each direction
    if plotType == "individual" will plot individual NPs and label
    will also plot a final global summary plot for each amyloid class, where each iou_threshold will have an average point (averaged over 12 points)
    """
    annotators = ["NP{}".format(i) for i in range(1, 5)]

    iou_thresholds = list(np.arange(0.1, 1.0, 0.1))
    amyloid_classes = ["Cored", "CAA"]
    precision_dict = pickle.load(open("pickles/precision_dict_annotators_relative_to_each_other.pkl", "rb"))
    color_dict = {"NP1": "#ff8800", "NP2": "#03ebfc", "NP3":"#fc039d", "NP4":"#23ba28"}
    ##dictionary to store summary points, will save this dictionary at end of function
    results_dict = {amyloid_class: {} for amyloid_class in amyloid_classes}
    
    for amyloid_class in amyloid_classes:
        global_ys = []
        for ground_truth in precision_dict[amyloid_class]:
            fig, ax = plt.subplots()
            ys = []
            for annotator2 in precision_dict[amyloid_class][ground_truth]:
                x = []
                y = []
                for iou_threshold in iou_thresholds:
                    x.append(iou_threshold)
                    y.append(precision_dict[amyloid_class][ground_truth][annotator2][iou_threshold])
                ys.append(y)
                global_ys.append(y)
                if plotType == "individual":
                    plt.plot(x,y, linestyle='-', marker='o', label=annotator2, color=color_dict[annotator2])
            if plotType == "aggregate":
                ys = np.array(ys)
                y_avg = ys.mean(axis=0)
                y_std = ys.std(axis=0)
                plt.plot(x, y_avg)
                plt.fill_between(x, y_avg-y_std, y_avg+y_std, alpha=0.5)
            plt.ylim([0.0, 1.0])

            plt.title("Amyloid Class = {}\nGround truth = {}".format(amyloid_class, ground_truth))
            ax.set_xlabel("IOU Threshold", fontname="Times New Roman", fontsize=12)
            ax.set_ylabel("Precision", fontname="Times New Roman", fontsize=12)
            plt.xticks(np.arange(0.1, 1.0, .1))

            if plotType == "individual":
                #Shrink current axis and place legend outside plot, top right corner 
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
                ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.35))
                plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
            plt.savefig("figures/annotator_precisions_relative_to_each_other_{}_{}.png".format(amyloid_class, ground_truth), dpi=300)

        ##now plot global ys - one summary plot if we average over each ground truth (each iou threshold will have 3 x 4 points (3 other annotators, 4 possible ground truths))
        fig, ax = plt.subplots()
        global_ys = np.array(global_ys)
        y_global_avg = global_ys.mean(axis=0)
        y_global_std = global_ys.std(axis=0)
        plt.plot(x,y_global_avg, linestyle='-', marker='o')
        plt.fill_between(x, y_global_avg - y_global_std, y_global_avg + y_global_std, alpha=0.5)
        plt.title("Comparing Annotators to Each Other\nAmyloid Class = {}\n".format(amyloid_class,ground_truth))
        ax.set_xlabel("IOU Threshold", fontname="Times New Roman", fontsize=12)
        ax.set_ylabel("Precision", fontname="Times New Roman", fontsize=12)
        plt.savefig("figures/annotator_precisions_relative_to_each_other_{}_global.png".format(amyloid_class), dpi=300)
        results_dict[amyloid_class] = {"x": x, "avg": y_global_avg, "std": y_global_std}
    pickle.dump(results_dict, open("pickles/annotatorPrecisionRelativeToEachOtherSummary.pkl", "wb"))

def plotTimeChart(iou_threshold=0.5):
    """
    Plots time spent vs AP for each amyloid class, one plot: X axis time spent, y axis AP
    Uses a diamond marker for CAA, and a circle for Cored
    """
    AP_map = pickle.load(open("pickles/APs_per_annotator.pkl", "rb"))
    time_map = {"NP1": 4.3, "NP2":1.5, "NP3":2.1, "NP4":2.2}
    annotators = ["NP{}".format(i) for i in range(1, 5)]
    color_dict = {"NP1": "#ff8800", "NP2": "#03ebfc", "NP3":"#fc039d", "NP4":"#23ba28"}    
    amyloid_classes = ["Cored", "CAA"]
    fig, ax = plt.subplots()
    for amyloid_class in amyloid_classes:
        for annotator in annotators:
            x = time_map[annotator]
            y = AP_map[annotator][amyloid_class][iou_threshold]
            marker = "$*$" if amyloid_class == "Cored" else "$@$"
            if amyloid_class == "Cored": 
                ax.scatter(x, y, s=120, marker=marker, color=color_dict[annotator], label=annotator)
            else: ##plot without label flag 
                ax.scatter(x, y, s=120, marker=marker, color=color_dict[annotator])
        plt.title("Time Spent Annotating\nVersus AP")
        plt.ylim([0.0, 1.0])
        ax.set_xlabel("Annotation Hours", fontname="Times New Roman", fontsize=12)
        ax.set_ylabel("Average Precision @IOU={}".format(iou_threshold), fontname="Times New Roman", fontsize=12)
        #Shrink current axis and place legend outside plot, top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.35))
        plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
        plt.savefig("figures/time_vs_AP.png".format(amyloid_class), dpi=300)
 
def plotAllAnnotations():
    """
    Will iterate over each image in the prospective validation set, and draw colored bboxes for each annotator (one color per annotator)
    """
    if not os.path.isdir("output/AllAnnotations/"):
        os.mkdir("output/AllAnnotations/")
    annotators = ["NP{}".format(i) for i in range(1, 5)]
                    #BGR:  orange #ff8800      blue #03ebfc          pink #fc039d            green #00ff26 (38, 255, 0) 
    color_dict = {"NP1": (0, 136, 255), "NP2": (252, 235, 3), "NP3":(157, 3, 252), "NP4": (40, 186, 35)}
    class_symbols = {"Cored": "*", "CAA":"@"}
    font = cv2.FONT_HERSHEY_SIMPLEX
    annotations_dict = {annotator: {} for annotator in annotators}
    for annotator in annotators:
        annotations = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator), "rb"))
        annotations_dict[annotator] = annotations
    images = list(annotations_dict["NP1"].keys())
    for img_name in images:
        img = cv2.imread("prospective_validation_images/" + img_name)
        for annotator in annotators:
            for entry in annotations_dict[annotator][img_name]:
                dictionary, class_label  = entry[0], entry[1]
                x1 = int(dictionary['x1'])
                y1 = int(dictionary['y1'])
                x2 = int(dictionary['x2'])
                y2 = int(dictionary['y2'])
                color = color_dict[annotator] 
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, class_symbols[class_label], (x1,y1), font, 1.5, color,2,cv2.LINE_AA)
        cv2.imwrite("output/AllAnnotations/".format(annotator) + img_name, img)

def plotImageComparisons(val_type="prospective", overlay_labels=True, overlay_predictions=True):
    """
    if val_type == "prospective":
        will plot the image model prediction boxes and also the annotation boxes from each annotator
        over the validation set for each annotator
    if val_type == "phase1" or "phase2":
        will plot the prediction/annotation comparisons over the data found in data/amyloid_test/
    saves images to output/{annotator}/ directory
    """
    class_symbols = {"Cored": "*", "CAA":"@"}
    if val_type == "prospective":
        annotators = ["consensus"]# + ["NP{}".format(i) for i in range(1, 5)]
    else:
        annotators = [val_type]
    for annotator in annotators:
        if not os.path.isdir("output/{}".format(annotator)):
            os.mkdir("output/{}".format(annotator))
        predictions = pickle.load(open("pickles/{}_validation_predictions.pkl".format(val_type), "rb"))
        
        ##load annotations dict and specify image folder
        if val_type == "prospective":
            if "consensus" in annotator:
                annotations = pickle.load(open("prospective_annotations/{}_annotations_iou_thresh_{}.pkl".format(annotator, 0.2), "rb"))
            else:
                annotations = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator), "rb"))
            img_folder = "prospective_validation_images/"
        else:
            annotations = pickle.load(open("pickles/{}_annotations.pkl".format(val_type), "rb"))
            annotations = {x:annotations[x] for x in annotations.keys() if x in predictions} ##we just want the validation images in {phase}_validation_predictions.pkl 
            img_folder = "data/amyloid_test/"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img_name in annotations:
            img = cv2.imread(img_folder + img_name)
            ##color detections
            if overlay_predictions:
                for entry in predictions[img_name]:
                    dictionary, class_label  = entry[0], entry[1]
                    annotation_class = 1 if class_label == "Cored" else 0
                    x1 = int(dictionary['x1'])
                    y1 = int(dictionary['y1'])
                    x2 = int(dictionary['x2'])
                    y2 = int(dictionary['y2'])
                    color = (255,0,0) if class_label == "CAA" else (0,0,255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            ##labels will be black boxes
            if overlay_labels:
                for entry in annotations[img_name]:
                    dictionary, class_label  = entry[0], entry[1]
                    x1 = int(dictionary['x1'])
                    y1 = int(dictionary['y1'])
                    x2 = int(dictionary['x2'])
                    y2 = int(dictionary['y2'])
                    color = (0,0,0) 
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, class_symbols[class_label], (x1,y1), font, 1.5,(0,0,0),2,cv2.LINE_AA)            
            cv2.imwrite("output/{}/".format(annotator) + val_type + "_" + img_name, img)

def addNPLabelToAnnotations(annotations, NP_id):
    """
    Given a dictionary of key: imagename: value: list of tuples(coordinate dict (x1,y1,x2,y2), class) and a NP_id (i.e. NP1, NP2, etc.)
    will modify annotations dictionary to include NP_id
    i.e. value: list of tuples(coordinate dict (x1,y1,x2,y2), class, NP_id)
    returns new annotations dictionary
    """
    for image_name in annotations.keys():
        l = []
        entries = annotations[image_name]
        for coordinate_dict, class_label in entries:
            l.append((coordinate_dict, class_label, NP_id))
        annotations[image_name] = l
    return annotations

def createMergedOrConsensusBenchmark(benchmark="consensus", iou_threshold=0.5):
    """
    Takes all human annotations and merges them 
    if benchmark == "merged", will simply merge all human annotations and return a new annotation set where overlapping boxes (any overlap >=1px) of the same class are merged
    if benchmark == "consensus", will create a consensus of 2 benchmark based on iou_threshold
        in this consensus case, eliminate any boxes that don’t have any IOU overlap with at least one other box of same class (to get rid of the one-offs)
        this will leave us with pathologies that have a consensus of 2. But there will be too many boxes for one pathology. To get rid of the extra boxes and instead of merging to get one box, do the following:
            for a pathology, for any pair of boxes that have the same class, different NP annotator, and IOU >= iou_threshold, eliminate the larger box
    pickles the final merged or consensus benchmark
    """
    #key: imagename: value: list of tuples(coordinate dict (x1,y1,x2,y2), class)
    annotations = pickle.load(open("prospective_annotations/NP1_annotations.pkl", "rb"))
    annotations = addNPLabelToAnnotations(annotations, "NP1")
    imagenames = list(annotations.keys())

    ##combine all annotations into one annotation set 
    merged_dict = annotations
    annotators = ["NP{}".format(i) for i in range(2, 5)]
    for annotator in annotators:
        annotations = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator), "rb"))       
        annotations = addNPLabelToAnnotations(annotations, annotator)
        for image_name in imagenames:
            merged_dict[image_name] += annotations[image_name]

    if benchmark == "consensus":
        ##for consensus, remove any boxes that don't have any IOU>threshold with any other boxes of same class
        for image_name in merged_dict.keys():
            l = []
            entries = merged_dict[image_name]
            for entry1 in entries:
                overlaps_with_any = False
                coord_dict1 = entry1[0]
                ##reformat to tuple (x1, y1, x2, y2) instead of dictionary for input to IOU function 
                new_coord1 = (coord_dict1["x1"], coord_dict1["y1"], coord_dict1["x2"], coord_dict1["y2"]) 
                for entry2 in entries:
                    if entry1 != entry2:
                        coord_dict2 = entry2[0]
                        new_coord2 = (coord_dict2["x1"], coord_dict2["y1"], coord_dict2["x2"], coord_dict2["y2"]) 
                        ##if IOU >= iou_threshold and classes are the same, then keep this box, else we will remove it (by not adding it to l which will be the new list to replace the current one)
                        if IOU(new_coord1, new_coord2) >= iou_threshold and entry1[1] == entry2[1]:
                            overlaps_with_any = True
                            break
                if overlaps_with_any:
                    l.append(entry1)
            merged_dict[image_name] = l
        ##for consensus iterate again over boxes, and for any box A, find any other box B with IOU>threshold and larger than A, and remove B
        for image_name in merged_dict.keys():
            to_remove = []
            entries = merged_dict[image_name]
            for entry1 in entries: 
                coord_dict1 = entry1[0]
                new_coord1 = (coord_dict1["x1"], coord_dict1["y1"], coord_dict1["x2"], coord_dict1["y2"]) 
                area1 = (coord_dict1["x2"] - coord_dict1["x1"]) * (coord_dict1["y2"] - coord_dict1["y1"])
                NP_id1 = entry1[2]
                for entry2 in entries:
                    coord_dict2 = entry2[0]
                    new_coord2 = (coord_dict2["x1"], coord_dict2["y1"], coord_dict2["x2"], coord_dict2["y2"]) 
                    area2 = (coord_dict2["x2"] - coord_dict2["x1"]) * (coord_dict2["y2"] - coord_dict2["y1"])
                    NP_id2 = entry2[2]
                    if entry1 != entry2:
                        if IOU(new_coord1, new_coord2) >= iou_threshold and entry1[1] == entry2[1] and area2 > area1 and NP_id1 != NP_id2:
                            to_remove.append(entry2)
            ##remove entries that are part of to_remove
            merged_dict[image_name] = [entry for entry in merged_dict[image_name] if entry not in to_remove]
             
    if benchmark == "merged":
        ##now that they're all in one dictionary, let's reformat 
        ##first reformat to key: imagename: value: list of tuples(bbox (x1, y1, width, height), class labels of bbox (i.e. (cored, diffuse, CAA))
        for image_name in merged_dict.keys():
            l = []
            for dictionary, class_label in merged_dict[image_name]:
                x1 = int(dictionary['x1'])
                y1 = int(dictionary['y1'])
                x2 = int(dictionary['x2'])
                y2 = int(dictionary['y2'])
                if class_label == "Cored":
                    new_label = (1,0,0)
                if class_label == "CAA":
                    new_label = (0,0,1)
                l.append(((x1,y1,x2-x1, y2-y1), new_label))
            merged_dict[image_name] = l

        ##now combine overlapping bboxes that have the same class label and also iou >= iou_threshold
        merged_dict = combineOverlappingBboxes(pickle_name=None, mapp=merged_dict)
        ##convert back to original format of key: imagename: value: list of tuples(coordinate dict (x1,y1,x2,y2), class)
        for image_name in merged_dict.keys():
            l = []
            for bbox_coord, class_labels in merged_dict[image_name]:
                x1 = bbox_coord[0]
                y1 = bbox_coord[1]
                x2 = bbox_coord[0] + bbox_coord[2]
                y2 = bbox_coord[1] + bbox_coord[3]
                if class_labels[2] == 1:
                    annotation_class = "CAA"
                if class_labels[0] == 1:
                    annotation_class = "Cored"
                l.append(({"x1":x1, "y1":y1, "x2":x2, "y2": y2}, annotation_class))
            merged_dict[image_name] = l

    ##get rid of NP_ids
    for image_name in merged_dict.keys():
        l = []
        entries = merged_dict[image_name]
        for coordinate_dict, class_label, NP_id in entries:
            l.append((coordinate_dict, class_label))
        merged_dict[image_name] = l

    ##finally save to pickle
    pickle.dump(merged_dict, open("prospective_annotations/{}_annotations_iou_thresh_{}.pkl".format(benchmark, round(iou_threshold, 2)), "wb"))









shutil.rmtree("output/")
os.mkdir("output/")

# #analysis of model v1 and v2 predictions
# for phase in ["phase1", "phase2"]:
#     convertPreProspectiveAnnotationsToPickle(phase=phase)
#     runModelOnValidationImages(val_type=phase)
#     for iou_threshold in np.arange(0.1, 1.0, 0.1):
#         compareAnnotationsToPredictions(iou_threshold=iou_threshold, annotator=phase, val_type=phase)
#     plotPRC(annotator=phase, val_type=phase, separate_legend=False)
#     plotImageComparisons(val_type=phase, overlay_labels=True, overlay_predictions=True)
# plotAPsForPhases()

# #prospective validation of model v2 predictions
# runModelOnValidationImages(val_type="prospective")
# for iou_threshold in np.arange(.1, 1, .1):
#     createMergedOrConsensusBenchmark(benchmark="consensus", iou_threshold=iou_threshold)
# for annotator in ["consensus"] + ["NP{}".format(i) for i in range(1, 5)]:
#     for iou_threshold in np.arange(0.1, 1.0, 0.1):
#         compareAnnotationsToPredictions(iou_threshold=iou_threshold, annotator=annotator, val_type="prospective")
#     plotPRC(annotator=annotator, val_type="prospective")
#     getAnnotationOverlaps(annotator, iou_threshold=0.05)
# getPrecisionsOfAnnotatorsRelativeToEachOther()
# plotPrecisionsOfAnnotatorsRelativeToEachOther(plotType="aggregate")
# plotAPsForProspective(plotAvgOverlay=True)
# plotImageComparisons(val_type="prospective", overlay_labels=True, overlay_predictions=True)

##other analyses
# findLowPerformanceImages("Cored", "consensus", iou_threshold=0.5)
# plotAllAnnotations()
# getInterraterAgreement(iou_threshold=0.5)
plotInterraterAgreement()
# plotTimeChart()











