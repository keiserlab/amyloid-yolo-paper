from core import *
import unittest

class DataSetTests(unittest.TestCase):
    """
    various tests for our dataset set up 
    """

    def testTrainingTestSplit(self):
        """
        Makes sure train and test do not overlap 
        """
        valid_images = []
        with open("data/custom/valid.txt") as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace("\n", "")
                valid_images.append(line)
        file.close()
        train_images = []
        with open("data/custom/train.txt") as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace("\n", "")
                train_images.append(line)
        file.close()
        train_images, valid_images = set(train_images), set(valid_images)
        self.assertTrue(len(train_images.intersection(valid_images)) == 0)

    def testValidationImages(self):
        """
        Various tests to make sure our validation set is constructed correct
        """
        ## Make sure our validation set has the right number of distinct WSIs
        df = pd.read_csv("csvs/prospective_validation_images.csv")
        WSIs = set()
        for index, row in df.iterrows():
            field = row["Image Name"] 
            start_index = field.find("data/MRPI_tiles/") + 16
            WSI = ""
            for i in range(start_index, len(field)):
                if field[i] == "/":
                    break
                else:
                    WSI += field[i]
            WSIs.add(WSI)
        self.assertTrue(len(WSIs) == 56) ##12 WSIs, 4 stains + 8 WSIs at random 

        ##make sure all images are distinct
        self.assertTrue((len(df["Image Name"]) == len(set( df["Image Name"]))))

        ##make sure we have even split between "CAA" and "Cored" enrichment, and 96 total for each (4 stains, 12 WSIs, 2 from each slide)
        CAA_count = len([x for x in list(df["Amyloid Class"]) if x == "CAA"])
        Cored_count = len([x for x in list(df["Amyloid Class"]) if x == "Cored"])
        self.assertTrue(CAA_count == Cored_count == 96)

        ##make sure we have exactly 144 model enriched images (4 stains , 12 WSIs, 3 per WSI), and 48 human enriched images (4, stains, 12 WSIs, 1 per WSI)
        model_enriched_count = len([x for x in list(df["Selected by"]) if x == "Model"])
        human_enriched_count = len([x for x in list(df["Selected by"]) if x == "Human"])
        self.assertTrue(model_enriched_count == 144)
        self.assertTrue(human_enriched_count == 48)

        ##make sure validation images have equal stain distribution = 50 image_details
        stains_list = [x for x in list(df["Stain"])]
        stains = ["4G8", "6E10", "ABeta40", "ABeta42"]
        for stain in stains:
            stain_count = len([x for x in stains_list if x == stain])
            self.assertTrue(stain_count == 50)

  

unittest.main()
