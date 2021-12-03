import shutil
import os

shutil.rmtree("output/")
os.mkdir("output/")
if os.path.isfile("train_images/"):
	shutil.rmtree("train_images/")
	os.mkdir("train_images/")