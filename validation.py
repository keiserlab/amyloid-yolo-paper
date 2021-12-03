"""
Script pertaining to the validation of the study after all model training 
For Lise's dataset:
    consensus annotations: /srv/home/lminaud/tile_seg/consensus_csv/consensus_experts/consensus_2_complete.csv
    WSIs:
            UCI - 24: .svs rescanned: /srv/nas/mk2/projects/alzheimers/images/UCI/UCI_svs/
            UCLA/UCD: /srv/nas/mk2/projects/alzheimers/images/ADBrain/
            WSIs renamed with random ID: /srv/nas/mk2/projects/alzheimers/images/processed/processed_wsi/svs/
    1536 tiles:
            /srv/home/lminaud/tiles_backup/
            The csv name is: CAA_img_Daniel_project.csv
    image_details: /srv/home/lminaud/tile_seg/image_details.csv
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
import socket
from core import *

def calculatePlaqueCountsPerWSI(task, save_images=False):
    """
    if task == "CERAD all":
        will run the model over all of the CERAD dataset
    if task == "CERAD hold-out":
        will only run the model over just the CERAD hold-out dataset
    if task == "lise dataset":
        will run the model over Lise's dataset (just the top 12 WSIs with the most number of CAAs)
    Runs model over the 1536 tiles
    directory must contain subdirectories with the WSI name, and a 0/ directory inside those subdirectories with the actual 1536 pixel images 
    directory = 'data/CERAD/1536_tiles/' contains the 1536 x 1536 pixel CERAD test WSIs
    directory = 'data/MRPI_tiles/' contains the 1536 x 1536 pixel CAA images from Lise's study
    Saves a dictionary with key: WSI to key: Cored or CAA to value: count
    Saves another dictionary with key: WSI, key: 1536 x 1536 full path key: Cored or CAA to value: count
    If save_images, saves 1536 images with predicted bboxes to output/
    """
    ##load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
    model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_105.pth"))
    model.eval() 

    if task == "CERAD all":
        prefix = "CERAD_"
        directory = 'data/CERAD/1536_tiles/'
    if task == "CERAD hold-out":
        prefix = "CERAD_holdout_"
        directory = 'data/CERAD/1536_tiles/'
    if task == "lise dataset":
        prefix = "Lise_"
        directory = 'data/MRPI_tiles/'

    WSI_directories = os.listdir(directory)
    random.shuffle(WSI_directories)

    if task == "CERAD hold-out":
        holdouts = os.listdir(directory + "Dataset3HoldOut/")
        holdouts = [x.replace(".svs", "") for x in holdouts]
        WSI_directories = [x for x in WSI_directories if x in holdouts]
    
    ##dictionary to store counts of each pathology for each WSI
    WSI_dictionary = {WSI: {"Cored":0, "CAA": 0} for WSI in os.listdir(directory)} 
    ##make 1536 dictionary for more granular record keeping for selecting validation tiles?
    dictionary_1536 = {WSI:{} for WSI in os.listdir(directory)}
    ##instantiate new in case pickles already exists 
    pickle.dump(WSI_dictionary, open("pickles/" + prefix + "WSI_plaque_counts_dictionary.pkl", "wb"))
    pickle.dump(dictionary_1536, open("pickles/" + prefix + "1536_plaque_counts_dictionary.pkl", "wb"))
    
    if prefix == "Lise_":
        WSIs_of_interest = getWSIsWithMostCAAs(n=12)

    for WSI in WSI_directories:
        if prefix == "Lise_" and WSI not in WSIs_of_interest:
            continue
        img_dir = directory + WSI + "/0/" 
        subdirectories = os.listdir(img_dir)
        random.shuffle(subdirectories)
        for subdirectory in subdirectories:
            dataloader = DataLoader(
            ImageFolder(img_dir + subdirectory, transform= \
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
                    detections = non_max_suppression(detections, 0.8, 0.4)
                # Save image and detections
                imgs.extend(img_paths)
                img_detections.extend(detections)
            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, 20)]
            # Iterate through images and save plot of detections
            for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
                ##instantiate 1536 x 1536 counts dictionary, if detections exists, then this will be updated
                if path not in dictionary_1536[WSI].keys():
                    dictionary_1536[WSI][path] = {"Cored": 0, "CAA": 0}
                CAA_found = False
                # Create plot
                img = np.array(Image.open(path))
                plt.figure()
                fig, ax = plt.subplots(1, figsize=(19.95, 19.95))
                ax.imshow(img)
                # Draw bounding boxes and labels of detections
                if detections is not None:
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections, 416, img.shape[:2])
                    detections = mergeDetections(detections) 
                    detections = filterDetectionsByCAAModel(path, detections, classes)
                    if len(detections) == 0: ##possible that we removed all of the detections
                        continue 
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        if classes[int(cls_pred)] == "Cored":
                            WSI_dictionary[WSI]["Cored"] += 1
                            dictionary_1536[WSI][path]["Cored"] += 1
                        if classes[int(cls_pred)] == "CAA":
                            WSI_dictionary[WSI]["CAA"] += 1
                            dictionary_1536[WSI][path]["CAA"] += 1
                            CAA_found = True
                        # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # Add label
                        plt.text(
                            x1,
                            y1,
                            s=classes[int(cls_pred)],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )

                    # Save generated image with detections
                    plt.axis("off")
                    plt.gca().xaxis.set_major_locator(NullLocator())
                    plt.gca().yaxis.set_major_locator(NullLocator())

                    filename = os.path.basename(path).split(".")[0]
                    if "CERAD" in path:
                        output_path = path[path.find("CERAD"):].replace("/", "_").replace(".jpg", "") + ".png"
                    if "lminaud" in path:
                        output_path = path[path.find("lminaud"):].replace("/", "_").replace(".jpg", "") + ".png"
                    if save_images:
                        plt.savefig("output/" + output_path, bbox_inches="tight", pad_inches=0.0, dpi=300)
                    plt.close()
    pickle.dump(WSI_dictionary, open("pickles/" + prefix + "WSI_plaque_counts_dictionary.pkl", "wb"))
    pickle.dump(dictionary_1536, open("pickles/" + prefix + "1536_plaque_counts_dictionary.pkl", "wb"))
    return

def plotCERADVsCounts(plaque_type="Cored", CERAD_type="CERAD"):
    """
    Plots a boxplot for each CERAD category of the counts of PLAQUE_TYPE
    if CERAD_type = "CERAD" will use the real CERAD score categories ["none", "sparse", "moderate", "frequent"]
    else if CERAD_type = "Cored_MTG" will use the "Cored_MTG" categories [0,1,2,3]
    else if CERAD_type = "CAA_MTG" will use the "CAA_MTG" categories [0,1,2,3]
    """
    fig, ax = plt.subplots()
    if CERAD_type == "CERAD":
        categories = ["none", "sparse", "moderate", "frequent"]
        column_key = "CERAD"
        ax.set_xlabel("CERAD", fontname="Times New Roman", fontsize=12)
    if "MTG" in CERAD_type: ##if CERAD-like
        categories = [i for i in range(0, 4)]
        column_key = CERAD_type
        ax.set_xlabel(CERAD_type, fontname="Times New Roman", fontsize=12)
    WSI_plaque_counts = pickle.load(open("pickles/CERAD_WSI_plaque_counts_dictionary.pkl", "rb"))
    cerad_scores = pd.read_csv("csvs/CERAD_scores.csv")
    #key CERAD category, value: list of plaque type counts
    cerad_scores_map = {cat: [] for cat in categories} 
    for index, row in cerad_scores.iterrows():
        WSI_name = row["WSI_ID"]
        if WSI_name not in WSI_plaque_counts:
            print("{} not found in WSI plaque counts dictionary".format(WSI_name))
            continue
        cerad_scores_map[row[column_key]].append(WSI_plaque_counts[WSI_name][plaque_type])
    ax.boxplot([cerad_scores_map[cat] for cat in categories])
    ax.set_ylabel("{} Count According to Model".format(plaque_type), fontname="Times New Roman", fontsize=12)
    categories = [str(cat) + "\nn=" + str(len(cerad_scores_map[cat])) + " WSIs" for cat in categories]
    ax.set_xticklabels(categories,fontsize=10, fontname="Times New Roman")
    plt.title("CERAD Correlation with Predicted {} Counts".format(plaque_type))
    plt.gcf().subplots_adjust(bottom=0.14, top=.89)
    plt.savefig("figures/CERAD_correlation_{}_{}.png".format(plaque_type, CERAD_type), dpi=300)

def getWSIsWithMostCAAs(n=1):
    """
    Runs through the consensus of 2 annotations for Lise's dataset,
    For each stain, gets the top n WSIs as a list of strings for the WSIs with the greatest sum of positive annotations sum(lepto, parenchymal, capillary)
    Constructs a dictionary with key: stain, value: list of top n WSIs for that stain 
    Returns the final list of WSI names (4 stains x n total)
    """
    df = pd.read_csv("csvs/consensus_2_complete.csv")
    ##dictionary of key: WSI to value: sum of leptomeningeal, parenchymal, capillary
    WSI_dict = {WSI : 0 for WSI in df["source"]}
    for index, row in df.iterrows():
        WSI_dict[row["source"]] += row["leptomeningeal"] + row["parenchymal"] + row["capillary"]
    ##list of (stain, keyword that indicates stain) tuples
    stain_keywords = [("6E10", "beta_amyloid"), ("4G8", "4G8"), ("ABeta40", "Abeta40"), ("ABeta42", "Abeta42")]
    ##dictionary of key: stain, value: list of (WSI name, CAA sum for WSI) tuples
    stain_dict = {stain_keyword[0]: [(k, v) for k, v in sorted(WSI_dict.items(), key=lambda item: item[1]) if stain_keyword[1] in k] for stain_keyword in stain_keywords}
    ##only keep top n
    stain_dict = {stain: stain_dict[stain][-n:] for stain in stain_dict}
    final_list = [tup[0] for stain in ["6E10", "4G8", "ABeta40", "ABeta42"] for tup in stain_dict[stain] ]
    return final_list

def createPlaqueCountsDictionaryByHumanAnnotation(): 
    """
    Creates a dictionary with key: WSI, key: 1536 x 1536 pixel image: value: count of CAA (+) human annotations
    """
    ##first create map from key: 256px imagename to value: (source, tile_column, tile_row)
    details = pd.read_csv("csvs/image_details_lise.csv")
    map_256 = {}
    for index, row in details.iterrows():
        map_256[row["imagename"]] = (row["source"], row["tile_column"], row["tile_row"])
    pickle.dump(map_256, open("pickles/Lise_human_annotation_details_locations.pkl", "wb")) 

    ##next add (source, tile_column, tile_row) to annotations df
    df = pd.read_csv("csvs/consensus_2_complete.csv")
    df["source"] = ["" for i in range(0, len(df))]
    df["tile_column"] = ["" for i in range(0, len(df))]
    df["tile_row"] = ["" for i in range(0, len(df))]
    for index, row in df.iterrows():
        location_info = map_256[row["tilename"]]
        df.at[index, "source"] = location_info[0]
        df.at[index, "tile_column"] = location_info[1]
        df.at[index, "tile_row"] = location_info[2]

    ##create final dictionary
    return_dictionary = {WSI: {} for WSI in list(set(df["source"]))}
    ##reset in case this was made before
    pickle.dump(return_dictionary, open("pickles/Lise_human_annotation_1536_plaque_counts_dictionary.pkl", "wb")) 

    for index, row in df.iterrows():
        path = "data/MRPI_tiles/" + row["source"] + "/0/" + str(row["tile_row"]) + "/" + str(row["tile_column"]) + ".jpg"
        CAA_sum = int(row["leptomeningeal"] + row["parenchymal"] + row["capillary"])
        if path not in return_dictionary:
            return_dictionary[row["source"]][path] = CAA_sum
        else:
            return_dictionary[row["source"]][path] += CAA_sum
    pickle.dump(return_dictionary, open("pickles/Lise_human_annotation_1536_plaque_counts_dictionary.pkl", "wb")) 

def getStain(string):
    """
    Given string, will return the stain
    """
    stain = ""
    if "4G8" in string:
        stain = "4G8"
    if "Abeta42" in string:
        stain = "ABeta42"
    if "Abeta40" in string:
        stain = "ABeta40"
    if "beta_amyloid" in string:
        stain = "6E10"
    if stain == "":
        raise Exception("cannot determine stain from string: {}".format(string))
    else:
        return stain 

def pullValidationImages():
    """
    Method to write a dataframe with a list of all of the validation images we'll use for the study
    For each of the 4 stains:
        50 distinct fields total (1536 x 1536 px):
            select top 12 WSIs based on summation of human CAA counts
            for each slide:
                pick field with largest count of CAA (+) model predictions
                pick field with largest count of CAA (+) human annotations 
                pick top 2 fields with largest count of Cored (+) model predictions
            from 2 other WSIs outside original 12: pick 2 more fields completely at random / or pick 2 more fields with no CAA (+) human annotation
    """
    ## ongoing list of all images that are going to be part of the validation set 
    selected_images = [] 
    ## write DF to keep track of all relevant info (like coming from model enrichment or from human annotation)
    return_df = pd.DataFrame()

    ## grab images selected by model predictions first
    dictionary_1536 = pickle.load(open("pickles/Lise_1536_plaque_counts_dictionary.pkl", "rb"))

    ##now let's grab the images selected by greatest count of human annotations for only CAA (Cored annotations don't exist!)
    top_12 = getWSIsWithMostCAAs(n=12)
    human_annotations_dict = pickle.load(open("pickles/Lise_human_annotation_1536_plaque_counts_dictionary.pkl", "rb"))
    relevant_slides = {k: human_annotations_dict[k] for k in human_annotations_dict.keys() if k in top_12}
    
    for key in relevant_slides.keys():
        relevant_slides[key] = sorted(relevant_slides[key].items(),  key=lambda item: item[1])

    ## list for images to pull
    human_selected = []
    for key in relevant_slides.keys():
        ## if there is a tie for highest plaque count, we will select one of the contenders at random
        highest_plaque_count = relevant_slides[key][-1][1]
        contenders = []
        second_best_contenders = []
        ## iterate over the list of fields for this WSI to find all of the contenders
        for i in range(len(relevant_slides[key]) -1, -1, -1):
            plaque_count = relevant_slides[key][i][1]
            if plaque_count != highest_plaque_count:
                second_highest_plaque_count = relevant_slides[key][i][1]
                for j in range(i, -1 , -1):
                    plaque_count = relevant_slides[key][j][1]
                    if plaque_count == second_highest_plaque_count:
                        second_best_contenders.append(relevant_slides[key][j][0])
                    else:
                        break
                break
            else:
                contenders.append(relevant_slides[key][i][0])
        ## shuffle the list of contenders and select one at random
        random.shuffle(contenders)
        random.shuffle(second_best_contenders)
        contenders = [x for x in contenders if x not in selected_images]
        second_best_contenders = [x for x in second_best_contenders if x not in selected_images]        
        if len(contenders) != 0:
            human_selected.append(contenders[0])
            selected_images.append(contenders[0])
        else:
            human_selected.append(second_best_contenders[0])
            selected_images.append(second_best_contenders[0])
    ##write to df 
    for img_name in human_selected:
        return_df = return_df.append({"Image Name" : img_name, "Stain" : getStain(img_name), "Selected by" : "Human", "Amyloid Class" : "CAA"}, 
            ignore_index = True)

    ## first CAA, then Cored 
    for amyloid_type in ["CAA", "Cored"]:
        ## most of dictionary_1536 is empty (except 48 slides) because we did not process them to save compute
        non_empties = {k: dictionary_1536[k] for k in dictionary_1536.keys() if len(dictionary_1536[k]) > 0}
        ## sort by count of amyloid_type (largest at the back of the list)
        for key in non_empties.keys():
            non_empties[key] = sorted(non_empties[key].items(),  key=lambda item: item[1][amyloid_type])
        ## list for images to pull, select field(s) with greatest plaque count
        model_selected = []
        for key in non_empties.keys():

            ##shuffle the entries of non_empties[key], but preserve being sorted by count
            counts_dict = {} #key: count, value: list of (field name, {"CAA": count, "Cored":count}
            for i in range(0, len(non_empties[key])):
                plaque_count = non_empties[key][i][1][amyloid_type]
                if plaque_count not in counts_dict:
                    counts_dict[plaque_count] = [non_empties[key][i]]
                else:
                    counts_dict[plaque_count].append(non_empties[key][i])
            ##shuffle the lists for each count
            for count in counts_dict:
                random.shuffle(counts_dict[count])
            ##sort by key (count), count_items is list of tuples: [(count, list of (field name, {"CAA": count, "Cored":count})]
            counts_items = sorted(counts_dict.items())
            ##our list we're going to construct that is shuffled within each count but still sorted by count
            shuffled_ranked_list = []
            ## iterate over our counts_items
            for i in range(0, len(counts_items)):
                ## iterate over list of (field name, {"CAA": count, "Cored":count})
                for item in counts_items[i][1]:
                    shuffled_ranked_list.append(item)
            non_empties[key] = shuffled_ranked_list
            ## count to keep track of how many we've added to selected_images
            added = 0
            for i in range(len(non_empties[key]) - 1, -1, -1):
                field = non_empties[key][i][0]
                if field not in selected_images:
                    model_selected.append(field)
                    selected_images.append(field)
                    added += 1
                if amyloid_type == "CAA" and added == 1: ##we only want one model selected for CAA
                    break
                if amyloid_type == "Cored" and added == 2: ##we want two model selected for Cored 
                    break
        #write to df 
        for img_name in model_selected:
            return_df = return_df.append({"Image Name" : img_name, "Stain" : getStain(img_name), "Selected by" : "Model", "Amyloid Class" : amyloid_type}, 
                ignore_index = True)
   
    ##finally let's pull the 8 WSIs (2 for each of 4 stain types) that are outside the top 12 and select completely at random (remember that these WSIs don't have any fields instantiated in their dict to save on compute)
    outside_top_12 = [k for k in dictionary_1536.keys() if k not in top_12]
    outside_top_12.remove("CAA_img_Daniel_project.csv")
    random.shuffle(outside_top_12)
    stains = ["4G8", "ABeta42", "ABeta40", "6E10"]
    ## dict with key: stain to value: list of fields
    stains_dict = {stain: [] for stain in stains}
    for WSI in outside_top_12:
        files = []  
        stain = getStain(WSI)
        if len(stains_dict[stain]) < 2:
            ##get fields of WSI
            dirName = "data/MRPI_tiles/{}/0/".format(WSI)
            for (dirpath, dirnames, filenames) in os.walk(dirName):
                filenames = [dirpath + "/" + f for f in filenames]
                files += filenames
            random.shuffle(files)
            stains_dict[stain].append(files.pop())
            continue 
    ##final list of randomly selected images
    random_selected = []
    for stain in stains_dict:
        random_selected += stains_dict[stain]

    ##write to df 
    for img_name in random_selected:
        return_df = return_df.append({"Image Name" : img_name, "Stain" : getStain(img_name), "Selected by" : "Random", "Amyloid Class" : "N/A"}, 
            ignore_index = True)

    return_df.to_csv("csvs/prospective_validation_images.csv")

def createValidationImagesDirectory():
    """
    Creates and fills the directory prospective_validation_images/
    """
    df = pd.read_csv("csvs/prospective_validation_images.csv")
    if os.path.isdir("prospective_validation_images/"):
        shutil.rmtree("prospective_validation_images/")
    os.mkdir("prospective_validation_images/")
    for path in df["Image Name"]:
        shutil.copy(path, "prospective_validation_images/{}".format(path.replace("/", "_")))

def speedCheck(use_gpu=True, include_merge_and_filter=True):
    """
    Iterate over CERAD directory and run model over each WSI,
    calculates average time spent / WSI 
    tries a couple different batch sizes for comparison purposes
    use_gpu is a flag to determine if we want to use the gpu for the speed test
    include_merge_and_filter is a flag that determines whether we include the final output merge and CAA model filter 
    saves a dictionary called "run_times_use_gpu_{}.pkl".format(use_gpu)
    """
    hostname = socket.gethostname() 
    directory = 'data/CERAD/1536_tiles/'
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
        model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_105.pth"))
    else:
        device = torch.device("cpu")
        model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
        model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_105.pth", map_location=torch.device("cpu")))
    model.eval() 
    WSI_directories = os.listdir(directory)
    # WSI_directories = WSI_directories[0:1]
    random.shuffle(WSI_directories)
    # batch_sizes = [1,2,4,8,16,32]
    batch_sizes = [1]
    time_dict = {bs: {"machine": -1, "time spent": -1, "down time": -1, "model time spent": -1, "avg time / WSI": -1, "avg time / 1536 img": -1} for bs in batch_sizes}
    for batch_size in batch_sizes:
        num_1536 = 0
        down_time = 0 #will represent time spent on doing things other than model computation, like listing directories and creating data loaders
        t0 = time.time()
        for WSI in WSI_directories:#just go through 1 WSI
            t1 = time.time()
            img_dir = directory + WSI + "/0/" 
            subdirectories = os.listdir(img_dir)
            random.shuffle(subdirectories)
            t2 = time.time()
            down_time += t2 - t1
            for subdirectory in subdirectories: 
                t3 = time.time()
                dataloader = DataLoader(
                ImageFolder(img_dir + subdirectory, transform= \
                    transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)])),
                batch_size=batch_size,
                shuffle=False,
                num_workers=12,
                ) 
                classes = load_classes("data/custom/classes.names")  # Extracts class labels from file
                Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                t4 = time.time()
                down_time += t4 - t3
                imgs, img_detections = [], []
                for batch_i, (img_paths, input_imgs) in enumerate(dataloader): ##this for iteration takes a couple seconds for the first batch
                    input_imgs = Variable(input_imgs.type(Tensor))
                    input_imgs = input_imgs.to(device)
                    with torch.no_grad():
                        detections = model(input_imgs)
                        detections = non_max_suppression(detections, 0.8, 0.4)
                        ##if we want to also include the merge and CAA filter 
                        if include_merge_and_filter:
                            for img_i, (path, detections) in enumerate(zip(img_paths, detections)):
                                if detections is not None:
                                    detections = rescale_boxes(detections, 416, (1536, 1536))
                                    detections = mergeDetections(detections) 
                                    detections = filterDetectionsByCAAModel(path, detections, classes)
                    num_1536 += len(img_paths)
                    # mem_map = get_gpu_memory_map()
                    # print(mem_map)
        final_time = time.time()
        model_time_spent = final_time - t0 - down_time
        time_dict[batch_size]["machine"] = hostname
        time_dict[batch_size]["time spent"] = final_time
        time_dict[batch_size]["down time"] = down_time
        time_dict[batch_size]["model time spent"] = model_time_spent
        time_dict[batch_size]["avg time / WSI"] = model_time_spent / float(len(WSI_directories))
        time_dict[batch_size]["avg time / 1536 img"] = model_time_spent / float(num_1536)
        print("machine: ", hostname)
        print("batch size: ", batch_size)
        print("use gpu: ", use_gpu)
        print("num 1536 images: ", num_1536)
        print("time spent", final_time)
        print("down time: ", down_time)
        print("model time spent: ", model_time_spent)
        print("avg time per WSI: ", model_time_spent / float(len(WSI_directories)))
        print("avg time per 1536 image: ", model_time_spent / float(num_1536))
    pickle.dump(time_dict, open("pickles/run_times_use_gpu_{}_{}.pkl".format(use_gpu, hostname), "wb"))



calculatePlaqueCountsPerWSI(task="lise dataset")
# calculatePlaqueCountsPerWSI(task="CERAD all", save_images=False)
# plotCERADVsCounts(plaque_type = "Cored", CERAD_type="CERAD")
# plotCERADVsCounts(plaque_type = "CAA", CERAD_type="CERAD")
# plotCERADVsCounts(plaque_type = "Cored", CERAD_type="Cored_MTG")
# plotCERADVsCounts(plaque_type = "CAA", CERAD_type="CAA_MTG")

# createPlaqueCountsDictionaryByHumanAnnotation()

# speedCheck(use_gpu=True)
# speedCheck(use_gpu=False)



##CAREFUL ABOUT RUNNING THESE, DON'T WANT TO OVERWRITE 
# pullValidationImages()
# createValidationImagesDirectory()




