# amyloid-yolo-paper
author: Daniel Wong (wongdanr@gmail.com)

## Open access image data
LINK HERE <br />
Identifier: DOI HERE
Please download the zip file called data.zip and place in the amyloid-yolo-paper/ directory 

## Installation Instructions:
We've included an example conda environment in this repository called YOLOv3_.yml. To install the necessary packages, simply install conda first (https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and then
'conda env create -f YOLOv3_.yml -n YOLOv3'
to create a new conda environment (called YOLOv3) from the .yml file. Install time should take just a few minutes. 
Alternatively, we've listed the python packages and version numbers in requirements.txt

## Hardware and Software Specifications:
All deep learning models were trained using Nvidia Geforce GTX 1080 GPUs with a 64 CPU machine.
We used a CentOS Linux operating system (version 7).

## Content:

**checkpoints:**<br /> contains different PyTorch models saved at each epoch during training of model version 2. The model "yolov3_ckpt_105.pth" was the final one used for prospective validation.

**checkpoints_modelv1:**<br /> contains different PyTorch models saved at each epoch during training of model version 1. The model "yolov3_ckpt_157.pth" was used for making the CAA training labels for training model version 2. 

**config**<br /> contains original configuration files from the repo: https://github.com/eriklindernoren/PyTorch-YOLOv3

**core.py**<br /> contains most of the core method and class definitions of the study

**crop.py**<br /> is the script used to crop the WSIs into smaller 1536 x 1536 pixel tiles

**data:**<br />
This folder contains the image dataset and labels: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**amyloid_test:** contains all of the raw test set images<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**amyloid_train:** contains all of the raw training set images<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**CERAD:** contains all of the image data pertaining to the CERAD validation analysis. The dataset is pulled from https://zenodo.org/record/1470797#.YapievHMK3I<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**custom:** contains labels, the training validation split, and raw images used for training model version 2 <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**MRPI_tiles:** contains all of the 1536 x 1536 pixel tiles in the MRPI grant. Only the tiles used for prospective validation are released in this study. The full dataset will be released in a subsequent publication. <br /> 

**detect.py**<br /> is the script used to run the object detector on images and save the resulting boxed output images to output/. E.g. to run the model detection on the prospective validation images: 
python3 detect.py --image_folder prospective_validation_images/ --class_path data/custom/classes.names --model_def config/yolov3-custom.cfg  --checkpoint_model yolov3_ckpt_105.pth --conf_thres 0.8 --weights_path checkpoints/yolov3_ckpt_105.pth --img_size 416 --merge_boxes True --filter_CAA_detections_by_model True

**figures**<br /> is a destination directory for saving figure images

**models.py**<br /> contains method and class definitions relevant to the model architecture

**original_data**<br /> contains the original labels used for training model version 1. These labels are the raw bounding bounding box annotations from a consensus of 2 experts where any overlapping boxes of the same class are merged into a super box. Contrast this with labels found in the data/custom/ directory. For these labels, only the training set labels are modified such that CAA predictions from model version 1 are stipulated as label data (to train model version 2).

**output/**<br /> is a temporary destination directory for writing different image outputs for inspection

**pickles**<br /> contains different pickle files 
**PRC_tables**<br /> contains various intermediate CSVs used for calculating precision recall metrics during the prospective validation phase of the study

**prospective_annotations**<br /> contains raw expert annotation for the prospective validation phase of the study

**prospective.py**<br /> contains the method definitions and runner code for the prospective validation phase of the study

**prospective_validation_images**<br /> contains the raw images used in the prospective validation phase of the study

**pyvips.yml** is an example conda environment that can be used for installing the necessary packages for cropping the WSI images

**test.py**<br /> is the script used to evaluate the model. E.g. python3 test.py  --model_def config/yolov3-custom.cfg --data_config config/custom.data --weights_path checkpoints/yolov3_ckpt_105.pth --img_size 416 

**train.py**<br /> is the script used to train the model, saves to checkpoints/ directory 

**unit_test.py**<br /> contains various unit tests

**utils**<br /> contains various utility scripts, originally pulled from https://github.com/eriklindernoren/PyTorch-YOLOv3

**validation.py**<br /> contains method definitions for analysis post-training and pre-prospective validation. 

**weights**<br /> contains pre-trained dark net weights

**YOLOv3.yml**<br /> is an example conda environment that can be used for installing the necessary packages.





