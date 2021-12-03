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

shutil.rmtree("outputs/")
os.mkdir("outputs/")

def selectCoredAndCAABBoxImages():
    """
    Writes new csv that looks a lot like image_details_.csv but adds a new column with the image annotation for that bbox (according a consensus of 2),
    and also filters to only include images that are positive for either cored or CAA (or both)
    """
    ##first creat a mapp of phase 1 image name to 3 class annotation
    binary_consensus_labels = pd.read_csv("/srv/home/dwong/Ziqi_Amyloid_ML/ziqi_csvs/binary_labels/strict_agreed_by_2.csv")
    mapp = {} ##key: 256 image name to value: annotation tuple (cored, diffuse, CAA)
    for index,row in binary_consensus_labels.iterrows():
        full_path_img = row["imagename"]
        img_name = full_path_img[full_path_img.find("/") + 1:]
        mapp[img_name] = (int(row["cored"]), int(row["diffuse"]), int(row["CAA"]))
    ##add columns cored, diffuse, and CAA to image_details dataframe
    image_details = pd.read_csv("image_details_phase1.csv")
    image_details = image_details[image_details.imagename.isin(mapp)]
    print(len(image_details))
    image_names = image_details["imagename"]
    annotations = [mapp[img_name] for img_name in image_names]
    image_details["cored"] = [x[0] for x in annotations]
    image_details["diffuse"] = [x[1] for x in annotations]
    image_details["CAA"] = [x[2] for x in annotations]
    ##filter for only cored and CAA
    image_details = image_details[image_details.cored.eq(1) | image_details.CAA.eq(1)]
    print("inter", len(image_details))
    image_details.to_csv("annotated_image_details_only_cored_and_CAA.csv")


def createMapFrom1536ToBboxes(csv_name, discriminate_classes=True):
    """
    given all information present in CSV_NAME
    creates a mapp of key: full path 1536 image name, to key: amyloid_class (and key: "all") to list of bbox coordinates
    key "all" contains all of the coordinates regardless of amyloid class
    """
    df = pd.read_csv(csv_name)
    prefix = "/srv/home/ztang/Amyloid_ML/data/normalized_tiles/"
    sources = df["source"]
    rows = df["tile_row"]
    columns = df["tile_column"]
    img_names = []
    for i in range(0, len(sources)):
        img_names.append(prefix + sources[i] + "/0/" + str(rows[i]) + "/" + str(columns[i]) + ".jpg")
    keys = ["cored", "diffuse", "CAA", "all"]
    mapp = {img : {key: [] for key in keys} for img in img_names}
    amyloid_classes = ["cored", "diffuse", "CAA"]
    
    print(len(mapp))
    
    for index, row in df.iterrows():
        img_name = prefix + row["source"] + "/0/" + str(row["tile_row"]) + "/" + str(row["tile_column"]) + ".jpg"
        bbox_coord = row["blob coordinates (xywh)"]
        bbox_coord = bbox_coord.replace("[", "").replace("]", "").split(" ")
        bbox_coord = [int(x) for x in bbox_coord if x != ""]
        if discriminate_classes:
            for amyloid_class in amyloid_classes:
                if row[amyloid_class] == 1:
                    mapp[img_name][amyloid_class].append(bbox_coord)
        mapp[img_name]["all"].append(bbox_coord)

    pickle.dump(mapp, open("map_1536_img_name_to_coordinates_from_{}.pkl".format(csv_name).replace(".csv", ""), "wb")) 
    print(len(mapp))

def overlayOn1536(pickle_name, color_different_classes=True):
    """
    for visualization purposes only
    saves 1536 images with red bboxes around all detected plaques
    pickle_name specifies the name of the pickle file containging a map from 1536 image name to bbox coordinates
    """
    relevant_classes = ["cored", "CAA"]
    color = {"cored": (0,0,255), "CAA": (255,0,0)}
    mapp = pickle.load(open(pickle_name, "rb"))
    for img_name in mapp:
        annotated_img = cv2.imread(img_name)
        if color_different_classes:
            for amyloid_class in relevant_classes:
                annotated_img = drawBBoxes(annotated_img, mapp[img_name][amyloid_class], color=color[amyloid_class])
        else:
            annotated_img = drawBBoxes(annotated_img, mapp[img_name]["all"])
        prefix = "/srv/home/ztang/Amyloid_ML/data/normalized_tiles/"
        save_name = img_name.replace(prefix, "").replace("/", "")
        cv2.imwrite("outputs/" + save_name, annotated_img)
        print(annotated_img.shape)

def drawBBoxes(img, bboxes, color=(0,0,255)):
    """
    given an IMG and a list of bbox coordinates BBOXES, will draw a red square for each bbox coordinate, and return the img
    """
    for bbox_coord in bboxes:
        x1 = bbox_coord[0]
        y1 = bbox_coord[1]
        x2 = bbox_coord[0] + bbox_coord[2]
        y2 = bbox_coord[1] + bbox_coord[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img,str(x1) + "," + str(y1),(x1,y1), font, .4,(0,0,255),1,cv2.LINE_AA)
    return img

def visualize256CoredAndCAABBoxImages():
    df = pd.read_csv("annotated_image_details_only_cored_and_CAA.csv")
    images = df["imagename"]
    sources = df["source"]
    prefix = "/srv/home/ztang/Amyloid_ML/data/tile_seg/blobs_bboxes/"
    for i in range(0, len(images)):
        print(i)
        img = cv2.imread(prefix + sources[i] + "/" + images[i])
        cv2.imwrite("outputs/" + images[i], img)

#=========================================================================
#=========================================================================
#=========================================================================
# def selectCoredAndCAABBoxImages():
#     """
#     Writes new csv that looks a lot like image_details_.csv but adds a new column with the image annotation for that bbox (according a consensus of 2),
#     and also filters to only include images that are positive for either cored or CAA (or both)
#     """
#     ##first creat a mapp of phase 1 image name to 3 class annotation
#     binary_consensus_labels = pd.read_csv("/srv/home/dwong/Ziqi_Amyloid_ML/ziqi_csvs/binary_labels/strict_agreed_by_2.csv")
#     mapp = {} ##key: 256 image name to value: annotation tuple (cored, diffuse, CAA)
#     for index,row in binary_consensus_labels.iterrows():
#         full_path_img = row["imagename"]
#         img_name = full_path_img[full_path_img.find("/") + 1:]
#         mapp[img_name] = (int(row["cored"]), int(row["diffuse"]), int(row["CAA"]))
#     ##add columns cored, diffuse, and CAA to image_details dataframe
#     image_details = pd.read_csv("image_details_phase1.csv")
#     image_details = image_details[image_details.imagename.isin(mapp)]
#     print(len(image_details))
#     image_names = image_details["imagename"]
#     annotations = [mapp[img_name] for img_name in image_names]
#     image_details["cored"] = [x[0] for x in annotations]
#     image_details["diffuse"] = [x[1] for x in annotations]
#     image_details["CAA"] = [x[2] for x in annotations]
#     ##filter for only cored and CAA
#     image_details = image_details[image_details.cored.eq(1) | image_details.CAA.eq(1)]
#     print("inter", len(image_details))
#     image_details.to_csv("annotated_image_details_only_cored_and_CAA.csv")


# def createMapFrom1536ToBboxes(csv_name, discriminate_classes=True):
#     """
#     given all information present in CSV_NAME
#     creates a mapp of key: full path 1536 image name, 
#     to key: amyloid_class (and key: "all") to list of bbox coordinates
#     key "all" contains all of the coordinates regardless of amyloid class
#     """
#     df = pd.read_csv(csv_name)
#     prefix = "/srv/home/ztang/Amyloid_ML/data/normalized_tiles/"
#     sources = df["source"]
#     rows = df["tile_row"]
#     columns = df["tile_column"]
#     img_names = []
#     for i in range(0, len(sources)):
#         img_names.append(prefix + sources[i] + "/0/" + str(rows[i]) + "/" + str(columns[i]) + ".jpg")
#     keys = ["cored", "diffuse", "CAA", "all"]
#     mapp = {img : {key: [] for key in keys} for img in img_names}
#     amyloid_classes = ["cored", "diffuse", "CAA"]
    
#     print(len(mapp))
    
#     for index, row in df.iterrows():
#         img_name = prefix + row["source"] + "/0/" + str(row["tile_row"]) + "/" + str(row["tile_column"]) + ".jpg"
#         bbox_coord = row["blob coordinates (xywh)"]
#         bbox_coord = bbox_coord.replace("[", "").replace("]", "").split(" ")
#         bbox_coord = [int(x) for x in bbox_coord if x != ""]
#         if discriminate_classes:
#             for amyloid_class in amyloid_classes:
#                 if row[amyloid_class] == 1:
#                     mapp[img_name][amyloid_class].append(bbox_coord)
#         mapp[img_name]["all"].append(bbox_coord)

#     pickle.dump(mapp, open("map_1536_img_name_to_coordinates_from_{}.pkl".format(csv_name).replace(".csv", ""), "wb")) 
#     print(len(mapp))

# def createMapFrom1536ToBboxesAndPredictions(pickle_name):
#     """
#     From a pickle containing a map of key: img_name to key: "all" to value: list of bbox coordinates
#     will create a new map that also has class predictions for each of these bbox coordinates 
#     i.e. key: img_name to value: list of (bbox coordinate, bbox class predictions)
#     """
#     model = torch.load("model_all_fold_3_thresholding_2_l2.pkl")
#     norm = np.load("normalization.npy", allow_pickle=True).item()
#     data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
#     mapp = pickle.load(open(pickle_name, "rb"))
#     new_mapp = {}
#     for img_name in mapp:
#         new_mapp[img_name] = []
#         img = cv2.imread(img_name)
#         bboxes = mapp[img_name]["all"]
#         for bbox_coord in bboxes:
#             img_256 = get256Img(img, bbox_coord)
#             bbox_class_preds = getClassPreds(img_256, model, data_transforms)
#             new_mapp[img_name].append((bbox_coord, bbox_class_preds))
#     pickle.dump(new_mapp, open("map_1536_img_name_to_coordinates_and_preds.pkl", "wb")) 

#=========================================================================
#=========================================================================
#=========================================================================


#visualize all blobs detected
# createMapFrom1536ToBboxes("image_details_phase1.csv", discriminate_classes=False)
# overlayOn1536("map_1536_img_name_to_coordinates_from_image_details_phase1.pkl", color_different_classes=False)


# ##visualize only cored or CAA blobs detected in phase 1
# selectCoredAndCAABBoxImages()
# createMapFrom1536ToBboxes("annotated_image_details_only_cored_and_CAA.csv", discriminate_classes=True)
overlayOn1536("map_1536_img_name_to_coordinates_from_annotated_image_details_only_cored_and_CAA.pkl", color_different_classes=True)


##problem: phase 1 annotations don't annotate every single blob, 
##so on a 1536 x 1536 tile we only have SOME of the blobs class-labeled.
##This is an incomplete annotation for a 1536 image, with missing instances
##...probably can't train then on scale of 1536 images


# visualize256CoredAndCAABBoxImages()







