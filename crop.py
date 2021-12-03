"""
Script for processing the WSIs used for the CERAD analysis
The WSIs are pulled from Ziqi Tang's Nature Comm study:
https://github.com/keiserlab/plaquebox-paper
"""

import pyvips as Vips
import shutil
import os
import pickle

def save_and_tile(image_to_segment, output_dir, tile_size=1536):
    basename = os.path.basename(image_to_segment.filename)
    base_dir_name = os.path.join(output_dir, basename.split('.svs')[0])
    print("base dir name", base_dir_name)
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
    Vips.Image.dzsave(image_to_segment, base_dir_name,
                        layout='google',
                        suffix='.jpg[Q=90]',
                        tile_size=tile_size,
                        depth='one',
                        properties=True)
    return None
    
def cropCERADWSIs():
    """
    Takes WSI directories (Dataset1a/, Dataset1b/, Dataset2/, Dataset3HoldOut/)
    found in directories data/CERAD/ and makes crops of size 1536 x 1536 pixels, 
    writes crops to directory data/CERAD/1536_tiles/
    dz_save sometimes creates temporary folder names and fails to rename to original, so we also write a file called "pickles/temporary_WSI_map.pkl" which keeps track of these mappings for renaming later 
    """
    subfolders = ["Dataset1a/", "Dataset1b/", "Dataset2/", "Dataset3HoldOut/"]
    directories = ["data/CERAD/{}".format(folder) for folder in subfolders]
    temp_map = {} ##sometimes dzsave makes temp directories holding the crops, but fails to rename them, map temp name to actual WSI name 
    for WSI_DIR in directories:
        SAVE_DIR = 'data/CERAD/1536_tiles/'
        wsi_files = os.listdir(WSI_DIR)
        imagenames = sorted(wsi_files)
        failed_images = []
        for imagename in imagenames:
            vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
            if vips_img.get('aperio.AppMag') == '40':
                print("resizing 40x to 20x")
                vips_img = vips_img.resize(0.5)
                print("resized")
            try:
                temp_map[vips_img.filename] = imagename.replace(".svs", "")
                save_and_tile(vips_img, SAVE_DIR)
            except:
                print("could not tile: {}".format(imagename))
                print("attempting divide and conquer")
                try:
                    divideAndConquerLargeWSI(vips_img, WSI_DIR, SAVE_DIR)
                    print("divide and conquer succeeded")
                except:
                    failed_images.append(imagename)
                    print("failed both save_and_tile and divide and conquer")
        print("failed to tile: {}".format(failed_images))
    pickle.dump(temp_map, open("pickles/temporary_WSI_map.pkl", "wb"))

def divideAndConquerLargeWSI(vips_img, SOURCE_DIR, SAVE_DIR):
    """
    If a WSI is too large and crashes the built-in save_and_tile function, then we need to divide and conquer,
    and split the WSI into smaller tiles, and then call save_and_tile on these smaller tiles
    SOURCE_DIR is the directory containing the source WSIs
    SAVE_DIR is the directory to save the crops to
    """
    width = vips_img.width
    height = vips_img.height
    multiple = 1536*16 # size of the cropped image 
    indice = 0 ## indice of the tiled image. E.g. indice_1s-70081-U-TCtx-beta_amyloid
    max_top = [top for top in range(0, height, multiple)] # get max_top[-1] to have the last value 
    max_left = [left for left in range(0, width, multiple)]
    for top in range(0, height, multiple): # 0, 1536, 1536x2, 1536x3
        for left in range (0, width, multiple):
            indice+=1
            try: # normal case - crop with multiples of 1536 
                crop_and_tile(SOURCE_DIR, SAVE_DIR, vips_img, left, top, multiple, multiple, indice)
            except: # if special case -> cannot crop with a multiple of 1536 x 1536 
                if (top == max_top[-1] and left == max_left[-1]): ## last crop -> bottom right 
                    new_height = height - top
                    new_width = width - left 
                    crop_and_tile(SOURCE_DIR, SAVE_DIR, vips_img, left, top, new_width, new_height, indice)
                else:   
                    if top == max_top[-1]: ## end of the image - final height
                        new_height = height - top
                        crop_and_tile(SOURCE_DIR, SAVE_DIR, vips_img, left, top, multiple, new_height, indice)
                    if left == max_left[-1]: ## end of the image - final width
                        new_width = width - left
                        crop_and_tile(SOURCE_DIR, SAVE_DIR, vips_img, left, top, new_width, multiple, indice)

def crop_and_tile(SOURCE_DIR, SAVE_DIR, vips_img, left, top, param_width, param_height, indice):
    """
    Crops a vips_img into a smaller chunk defined by left, top, param_width, param_height,
    and then calls save_and_tile
    SOURCE_DIR is the directory containing the source WSIs
    SAVE_DIR is the directory to save the crops to
    """
    crop = vips_img.crop(left, top, param_width, param_height) 
    pathname = vips_img.filename
    crop.filename = pathname
    basename = os.path.basename(crop.filename)
    basename = str(indice)+"_"+basename
    crop.filename = SOURCE_DIR+basename
    print(os.path.basename(crop.filename))
    save_and_tile(crop, SAVE_DIR)

def clear1536Directory():
    """
    Clears the 1536 directory of all folders except those corresponding to hold-out
    """
    holdouts = os.listdir("data/CERAD/Dataset3HoldOut/")
    holdouts = [x.replace(".svs","") for x in holdouts]
    dir_1536 = "data/CERAD/1536_tiles/"
    for subdirectory in os.listdir(dir_1536):
        if subdirectory not in holdouts:
            shutil.rmtree(dir_1536 + subdirectory)

def merge1536Subdirectories():
    """
    If we divide and conquer some large slides, we'll have multiple subfolders of one WSI present,
    e.g. 1_WSIname, 2_WSIname, etc.
    we need to merge them into one folder WSIname,
    this method finds folders belonging to the same WSI and groups them together, moving all image directories into the subfolder WSIname/0/ and creating new subfolder names to avoid collisions
    """
    ##iterate over each sub 
    ##if prefix begins with 1_ then increment and merge, if increment does not exist then break
    dir_1536 = "data/CERAD/1536_tiles/"
    for subdirectory in os.listdir(dir_1536):
        if subdirectory[0:2] == "1_": ##indicates that we've reached a folder created by divide and conquer
            counter = 1 ##prefix of divide and conquer folder
            WSI_name = subdirectory[2:]
            target_dir = dir_1536 + WSI_name + "/"  ##new directory to create and move files to
            shutil.rmtree(target_dir) ##this directory should have been partially made but incomplete because dz_save crashed out, so let's clear it 
            os.mkdir(target_dir)
            os.mkdir(target_dir + "/0/") ##create the /0/ subdirectory for sake of consistency with other folders in the 1536_tiles directory, dz_save creates the /0/ subfolder automatically
            ##increment our counter and move files if this directory exists
            while os.path.isdir(dir_1536 + "{}_".format(counter) + WSI_name):
                for subdirectory2 in os.listdir(dir_1536 + "{}_".format(counter) + WSI_name + '/0/'): ##each of the sub image directories
                    new_name = WSI_name + "_" + str(counter) + "_" + subdirectory2 ##give them a new name so that when we move them to target_dir, we won't have collisions
                    os.rename(dir_1536 + "{}_".format(counter) + WSI_name + "/0/" + subdirectory2,  dir_1536 + "{}_".format(counter) + WSI_name + "/0/" + new_name)
                    shutil.move(dir_1536 + "{}_".format(counter) + WSI_name + "/0/" + new_name, target_dir + '/0/' + new_name)
                shutil.rmtree(dir_1536 + "{}_".format(counter) + WSI_name) ##delete this divide and conquer directory 
                counter += 1 ##increment the counter to see if counter + 1 divide and conquer directory exists, and then repeat the same process if it does

def renameTempDirectories():
    """
    Renames the temporary image directories that dz_save made to their proper WSI name
    """
    dir_1536 = "data/CERAD/1536_tiles/"
    temp_map = pickle.load(open("pickles/temporary_WSI_map.pkl", "rb"))
    for temp_name in temp_map.keys():
        if "temp" in temp_name:
            os.rename(dir_1536 + temp_name, dir_1536 + temp_map[temp_name])


clear1536Directory()
cropCERADWSIs()
merge1536Subdirectories()
renameTempDirectories()

