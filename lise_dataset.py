from core import *
"""
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

def getWSIsWithMostCAAs():
	"""
	runs through the consensus of 2 annotations for Lise's dataset,
	For each stain, gets the top 10 WSIs as a list of strings for the WSIs with the greatest sum of positive annotations sum(lepto, parenchymal, capillary)
	Constructs a dictionary with key: stain, value: list of top 10 WSIs for that stain 
	Returns the final list of WSI names (30 total)
	"""
	df = pd.read_csv("/srv/home/lminaud/tile_seg/consensus_csv/consensus_experts/consensus_2_complete.csv")
	##dictionary of key: WSI to value: sum of leptomeningeal, parenchymal, capillary
	WSI_dict = {WSI : 0 for WSI in df["source"]}
	for index, row in df.iterrows():
		WSI_dict[row["source"]] += row["leptomeningeal"] + row["parenchymal"] + row["capillary"]
	
	##list of (stain, keyword that indicates stain) tuples
	stain_keywords = [("6E10", "beta_amyloid"), ("4G8", "4G8"), ("ABeta40", "Abeta40")]
	##dictionary of key: stain, value: list of (WSI name, CAA sum for WSI) tuples
	stain_dict = {stain_keyword[0]: [(k, v) for k, v in sorted(WSI_dict.items(), key=lambda item: item[1]) if stain_keyword[1] in k] for stain_keyword in stain_keywords}
	##only keep top 10
	stain_dict = {stain: stain_dict[stain][-10:] for stain in stain_dict}
	final_list = [tup[0] for stain in ["6E10", "4G8", "ABeta40"] for tup in stain_dict[stain] ]
	print(stain_dict)
	print(final_list)
	
print(getWSIsWithMostCAAs())
