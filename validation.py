"""
Script pertaining to CERAD-like analysis and speed runs
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
from scipy.stats import ttest_ind
import statsmodels.stats.power as smp
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
                fig, ax = plt.subplots(1, figsize=(6.65, 6.65))
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
        ax.set_xlabel("CERAD-like Score", fontname="Times New Roman", fontsize=12)
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
    
    print(cerad_scores_map)
    ##overlay points on top of boxplot
    for i in range(0, len(categories)):
        ax.scatter([i + 1 + random.uniform(-.02, .02) for x in cerad_scores_map[categories[i]]], cerad_scores_map[categories[i]], s=6, color="blue")
    
    ax.set_ylabel("{} Count According to Model".format(plaque_type), fontname="Times New Roman", fontsize=12)
    categories = [str(cat) + "\nn=" + str(len(cerad_scores_map[cat])) + " WSIs" for cat in categories]
    ax.set_xticklabels(categories,fontsize=10, fontname="Times New Roman")
    plt.title("Model Correlation with CERAD-like Score".format(plaque_type))
    plt.gcf().subplots_adjust(bottom=0.14, top=.89)
    plt.savefig("figures/CERAD_correlation_{}_{}.png".format(plaque_type, CERAD_type), dpi=300)
 
def plotCERADStatisticalSignificance(plaque_type="Cored"):
    """
    """
    WSI_plaque_counts = pickle.load(open("pickles/CERAD_WSI_plaque_counts_dictionary.pkl", "rb"))
    cerad_scores = pd.read_csv("csvs/CERAD_scores.csv")
    categories = ["none", "sparse", "moderate", "frequent"]
    cerad_scores_map = {cat: [] for cat in categories} 
    for index, row in cerad_scores.iterrows():
        WSI_name = row["WSI_ID"]
        if WSI_name not in WSI_plaque_counts:
            print("{} not found in WSI plaque counts dictionary".format(WSI_name))
            continue
        cerad_scores_map[row["CERAD"]].append(WSI_plaque_counts[WSI_name][plaque_type])
    t_test_map = {(cat1, cat2): -1 for cat1 in categories for cat2 in categories} #key: (CERAD category1, CERAD category2), value: (t-statistic, p-value) 
    grid = []
    for key in cerad_scores_map:
        l = []
        for key2 in cerad_scores_map:
            t, p = ttest_ind(cerad_scores_map[key], cerad_scores_map[key2])
            effect_size = (np.mean(cerad_scores_map[key]) - np.mean(cerad_scores_map[key2])) / float(np.sqrt((np.std(cerad_scores_map[key])**2 + np.std(cerad_scores_map[key2])**2) / float(2))) ##Cohen's d
            nobs = len(cerad_scores_map[key]) + len(cerad_scores_map[key2])
            power = smp.ttest_power(effect_size, nobs=nobs, alpha=0.05, alternative='two-sided')
            t_test_map[key, key2] = float(t), float(p)
            l.append(float(p))
        grid.append(l)
    print(t_test_map)
    grid = np.asarray(grid)
    print(grid.shape)
    fig, ax = plt.subplots()
    im = ax.imshow(grid,vmin=0, vmax=0.30, cmap="coolwarm")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    plt_labels = categories
    ax.set_xticklabels(plt_labels,fontsize=11)
    ax.set_yticklabels(plt_labels,fontsize=11)
    for i in range(len(categories)):
        for j in range(len(categories)):
            if grid[i][j] < .001:
                text = ax.text(j, i, "{:.2e}".format(grid[i][j]), ha="center", va="center", color="white", fontsize=11)
            else:
                text = ax.text(j, i, str(round(grid[i][j], 3)), ha="center", va="center", color="white", fontsize=11)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    ax.set_title("t-test p-values", fontsize=12)
    plt.savefig("figures/CERAD-t-test-p-values.png", dpi=300)

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
    pickle.dump(time_dict, open("pickles/run_times_use_gpu_{}_merge_and_filter_{}_{}.pkl".format(use_gpu, include_merge_and_filter, hostname), "wb"))

def calculateAvgSpeedOfTangSlidingWindow():
    """
    As first described in https://www.nature.com/articles/s41467-019-10212-1,
    one valid approach to counting the number of Cored and CAA pathologies is to perform a sliding window approach, and then segment the resulting heatmap:
    https://github.com/keiserlab/plaquebox-paper/blob/master/3)%20Visualization%20-%20Prediction%20Confidence%20Heatmaps.ipynb
    This method calculates and prints the average time to draw a heatmap per WSI according to the tqdm output
    """
    tqdms = ["28/28 [2:24:17<00:00, 309.19s/it]",  
            "49/49 [8:55:56<00:00, 656.25s/it]",  
            "28/28 [2:40:59<00:00, 345.00s/it]",  
            "28/28 [2:45:04<00:00, 353.73s/it]",  
            "28/28 [2:45:54<00:00, 355.52s/it]",  
            "28/28 [2:33:49<00:00, 329.63s/it]",  
            "27/27 [2:35:41<00:00, 345.99s/it]",  
            "26/26 [3:13:59<00:00, 447.68s/it]",  
            "28/28 [3:08:24<00:00, 403.74s/it]",  
            "21/21 [2:04:40<00:00, 356.22s/it]",
            "25/25 [2:35:53<00:00, 374.13s/it]",  
            "27/27 [2:26:48<00:00, 326.22s/it]",  
            "31/31 [3:14:01<00:00, 375.53s/it]",  
            "28/28 [3:23:10<00:00, 435.36s/it]",  
            "26/26 [2:35:15<00:00, 358.28s/it]",  
            "28/28 [2:47:33<00:00, 359.04s/it]",  
            "28/28 [2:43:00<00:00, 349.29s/it]",  
            "31/31 [3:25:29<00:00, 397.72s/it]",  
            "25/25 [2:08:55<00:00, 309.42s/it]",  
            "28/28 [2:56:10<00:00, 377.54s/it]"]
    total_seconds = 0
    for tqdm in tqdms:
        time = tqdm[tqdm.find("[") + 1:tqdm.find("<")]
        hours, minutes, seconds = time.split(":")
        total_seconds += float(hours)*60*60 + float(minutes)*60 + float(seconds)
    avg_seconds = total_seconds / float(len(tqdms))
    print(avg_seconds, len(tqdms))
   




shutil.rmtree("output/")
os.mkdir("output/")
# comparePreMergeLabelsWithPostMerge(sample_size=100)
# calculatePlaqueCountsPerWSI(task="CERAD all", save_images=False)
# calculatePlaqueCountsPerWSI(task="lise dataset")
# plotCERADVsCounts(plaque_type = "Cored", CERAD_type="CERAD")
# plotCERADVsCounts(plaque_type = "Cored", CERAD_type="Cored_MTG")
# plotCERADVsCounts(plaque_type = "CAA", CERAD_type="CAA_MTG")
# plotCERADStatisticalSignificance()
speedCheck(use_gpu=True, include_merge_and_filter=True)
speedCheck(use_gpu=False, include_merge_and_filter=True)
# calculazteAvgSpeedOfTangSlidingWindow()




