"""
Script for cleaning things up, delete later
"""
from core import *


# # ##let's rename the keys to have images with path names just in this directory
# pickle_name = "pickles/map_1536_img_name_to_coordinates_and_preds_strong_labels_combined_bboxes.pkl"
# pickle_name = "pickles/map_1536_img_name_to_coordinates_and_preds_weak_label_False.pkl"
# old_mapp = pickle.load(open(pickle_name, "rb")) ##of format img_name: [(bbox tuple, label tuple), ... ], in space 0->1536
# new_mapp = {}
# for key in old_mapp.keys():
# 	img_name = key.replace("/srv/home/ztang/Amyloid_ML/data/normalized_tiles/", "")
# 	img_name = img_name.replace("/", "_")
# 	img_name = "data/custom/images/" + img_name
# 	if not os.path.exists(img_name):
# 		print("DNE:", img_name)
# 	else:
# 		print("image exists!")
# 	new_mapp[img_name] = old_mapp[key]

# pickle.dump(new_mapp, open(pickle_name, "wb"))
# new_mapp = pickle.load(open(pickle_name, "rb"))
# print(new_mapp)



# pickle_name = "pickles/map_1536_img_name_to_coordinates_and_preds_weak_label_False.pkl"
# mapp = pickle.load(open(pickle_name, "rb"))
# assert preProcess(weak_label=False) == mapp



mapp = pickle.load(open("pickles/Lise_human_annotation_1536_plaque_counts_dictionary.pkl", "rb"))
print(mapp)

# df = pd.read_csv("csvs/prospective_validation_images.csv")
# for index, row in df.iterrows():
# 	image_name = row["Image Name"]
# 	df.at[index, "Image Name"] = image_name.replace("/srv/home/lminaud/tiles_backup/", "data/MRPI_tiles/")
# print(df)
# df.to_csv("csvs/prospective_validation_images.csv")








