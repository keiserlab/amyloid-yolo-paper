import unittest
from core import *

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

    def testTrainValidConsistencyForBothTrainingIterations(self):
        """
        Makes sure that the train.txt and valid.txt for the first phase of training (before model bootstrapping) is equivalent to
        train.txt and valid.txt used during the second phase of training
        """
        directories = ["data/custom/", "original_data/"]
        train1, train2, valid1, valid2 = [], [], [], []
        for directory in directories:
            with open(directory + "train.txt") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    if directory == "data/custom/":
                        train1.append(line)
                    if directory ==  "original_data/":
                        train2.append(line)
            file.close()
            with open(directory + "valid.txt") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    if directory == "data/custom/":
                        valid1.append(line)
                    if directory ==  "original_data/":
                        valid2.append(line)
            file.close()
        self.assertTrue(set(train1) == set(train2))
        self.assertTrue(set(valid1) == set(valid2))

    def testValidationImages(self):
        """
        Various tests to make sure our validation set is constructed correctly
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

    def testPreprocess(self):
        """
        test to ensure the return of the preprocess function matches the saved map_1536_img_name_to_coordinates_and_preds_weak_label_False.pkl file
        """
        pickle_name = "pickles/map_1536_img_name_to_coordinates_and_preds_weak_label_False.pkl"
        mapp = pickle.load(open(pickle_name, "rb"))
        self.assertTrue(preProcess(weak_label=False) == mapp)

class CoreFunctionTests(unittest.TestCase):
    """
    various tests for core functions
    """
    def testIOU(self):
        """
        test core IOU function
        """
        box1 = [100, 100, 200, 200]
        box2 = box1
        self.assertTrue(IOU(box1, box2) == 1)
        box1 = [100, 100, 200, 200]
        box2 = [201,201, 300, 300]
        self.assertTrue(IOU(box1, box2) == 0)
        box1 = [100, 100, 200, 200]
        box2 = [150,150, 200, 200]
        print(IOU(box1, box2))
        self.assertTrue( 0.25 <= IOU(box1, box2) <= 0.26)

    def testgetAccuracy(self):
        """
        test getAccuracy function
        """
        l1 = [0,0,0]
        l2 = [1,1,1]
        self.assertTrue(getAccuracy(l1, l2) == 0)
        l1 = [1,1,1]
        l2 = [1,1,1]
        self.assertTrue(getAccuracy(l1, l2) == 1)
        l1 = [0,1,0]
        l2 = [1,1,1]
        self.assertTrue(getAccuracy(l1, l2) == float(1/3))

    def testgetTPs(self):
        """
        test getTPS function
        """
        ##perfect match
        predictions = [[100, 100, 200, 200, 0.9, 0],[201, 201, 300, 300, 0.9, 1]]
        labels = [[100, 100, 200, 200, 0],[201, 201, 300, 300, 1]]
        self.assertTrue(getTPs(predictions, labels, 0.5, Pascal_VOC_scheme=True) == [1, 1])
        ##order invariance
        predictions = [[201, 201, 300, 300, 0.9, 1], [100, 100, 200, 200, 0.9, 0]]
        labels = [[100, 100, 200, 200, 0],[201, 201, 300, 300, 1]]
        self.assertTrue(getTPs(predictions, labels, 0.5, Pascal_VOC_scheme=True) == [1, 1])
        ##perfect match but different classes
        predictions = [[100, 100, 200, 200, 0.9, 1],[201, 201, 300, 300, 0.9, 0]]
        labels = [[100, 100, 200, 200, 0],[201, 201, 300, 300, 1]]
        self.assertTrue(getTPs(predictions, labels, 0.5, Pascal_VOC_scheme=True) == [0, 0])
        ##match but IOU not met
        predictions = [[100, 100, 200, 200, 0.9, 0],[201, 201, 300, 300, 0.9, 1]]
        labels = [[150, 150, 160, 160, 0],[201, 201, 203, 203, 1]]
        self.assertTrue(getTPs(predictions, labels, 0.5, Pascal_VOC_scheme=True) == [0, 0])

class ProspectiveValidationTests(unittest.TestCase):
    """
    various tests for prospective validation results
    """
    def testConsensusBenchmark(self):
        """
        Make sure the consensus benchmark is set up correctly
        """
        ##each consensus label should be found in one of the annotator's label set with a perfect match 
        mapp = pickle.load(open("prospective_annotations/consensus_annotations_iou_thresh_0.5.pkl", "rb"))
        annotations = pickle.load(open("prospective_annotations/NP1_annotations.pkl", "rb"))
        all_mapp = annotations
        for annotator in ["NP{}".format(i) for i in range(2, 5)]:
            annotations = pickle.load(open("prospective_annotations/{}_annotations.pkl".format(annotator), "rb"))       
            for image_name in annotations.keys():
                all_mapp[image_name] += annotations[image_name]
        for img_name in mapp:
            for tup in mapp[img_name]:
                self.assertTrue(tup in all_mapp[img_name])

    def testAnnotatorsRelativeToEachOtherBenchmark(self):
        """
        test to make sure we have the correct averages at iou 0.5 and 0.1
        """
        mapp = pickle.load(open("pickles/precision_dict_annotators_relative_to_each_other.pkl", "rb"))
        annotators = ["NP{}".format(i) for i in range(1, 5)]
        cored_precisions_1, cored_precisions_5 = [], []
        CAA_precisions_1, CAA_precisions_5 = [], []
        for annotator1 in annotators:
            for annotator2 in annotators:
                if annotator1 != annotator2:
                    cored_precisions_5.append(mapp["Cored"][annotator1][annotator2][0.5])
                    CAA_precisions_5.append(mapp["CAA"][annotator1][annotator2][0.5])
                    cored_precisions_1.append(mapp["Cored"][annotator1][annotator2][0.1])
                    CAA_precisions_1.append(mapp["CAA"][annotator1][annotator2][0.1])
        self.assertTrue(0.6 < np.mean(cored_precisions_5) < 0.7)
        self.assertTrue(0.5 < np.mean(CAA_precisions_5) < 0.6)
        self.assertTrue(0.6 < np.mean(cored_precisions_1) < 0.7)
        self.assertTrue(0.6 < np.mean(CAA_precisions_1) < 0.65)

    def testPrecisionMapEmpties(self):
        """
        Make sure any images with value -1 in "pickles/img_precision_maps/precision_img_map_{}_{}_{}.pkl".format(amyloid_class, annotator, iou_threshold)
        has no img prediction from the model 
        """
        annotators = ["NP{}".format(i) for i in range(1, 5)]
        iou_threshold = 0.5
        predictions = pickle.load(open("pickles/prospective_validation_predictions.pkl", "rb"))
        for annotator in annotators:
            for amyloid_class in ["Cored", "CAA"]:
                mapp = pickle.load(open("pickles/img_precision_maps/precision_img_map_{}_{}_{}.pkl".format(amyloid_class, annotator, iou_threshold), "rb"))
                mapp = {x: mapp[x] for x in mapp.keys() if mapp[x] == -1}
                for img_name in mapp:
                    for tup in predictions[img_name]:

unittest.main()
