import sys
import random
# append facelib to module search path
sys.path.append("..")

from facelib.dataset import DataSet
from facelib.feature import SpatialHistogram
from facelib.distance import EuclideanDistance, CosineDistance
from facelib.classifier import NearestNeighbor
from facelib.classifier import SVM
from facelib.model import PredictableModel
from facelib.validation import KFoldCrossValidation, LeaveOneOutCrossValidation
from facelib.visual import subplot
from facelib.util import minmax_normalize, read_images

import numpy as np
# import matplotlib colormaps
import matplotlib.cm as cm
# import for logging
import logging,sys

TESTEUCLIDEANKNN1 = 0
TESTEUCLIDEANKNN3 = 1
TESTEUCLIDEANKNN5 = 2
TESTEUCLIDEANKNN7 = 3
TESTCOSINEKNN1 = 4
TESTCOSINEKNN3 = 5
TESTCOSINEKNN5 = 6
TESTCOSINEKNN7 = 7
TESTSVM = 8

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "USAGE: eigenfaces.py </path/to/images> <setup type>"
        print
        print "Set up modes:"
        print "0 - 1 nearest neighbour with euclidean distance"
        print "1 - 3 nearest neighbour with euclidean distance"
        print "2 - 5 nearest neighbour with euclidean distance"
        print "3 - 7 nearest neighbour with euclidean distance"
        print "4 - 1 nearest neighbour with cosine distance"
        print "5 - 3 nearest neighbour with cosine distance"
        print "6 - 5 nearest neighbour with cosine distance"
        print "7 - 5 nearest neighbour with cosine distance"
        print "8 - Support vector machine"
        print
        print "Other values will default to 0"
        sys.exit()



    # set up a handler for logging
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add handler to facelib modules
    logger = logging.getLogger("facelib")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # load a dataset
    [data, labels] = read_images(sys.argv[1])

    # define feature extraction method
    feature = SpatialHistogram()

    # set up variables
    setup = sys.argv[2]

    testdataperclass = 3
    classcount = 40
    samplecount = 10

    classifier = 0

    # define classifier based on setup
    # if not found default to 1 nearest neighbour
    if setup == TESTEUCLIDEANKNN1:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    elif setup == TESTEUCLIDEANKNN3:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=3)
    elif setup == TESTEUCLIDEANKNN5:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=5)
    elif setup == TESTEUCLIDEANKNN7:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=7)
    elif setup == TESTCOSINEKNN1:
        classifier = NearestNeighbor(dist_metric=CosineDistance(), k=1)
    elif setup == TESTCOSINEKNN3:
        classifier = NearestNeighbor(dist_metric=CosineDistance(), k=3)
    elif setup == TESTCOSINEKNN5:
        classifier = NearestNeighbor(dist_metric=CosineDistance(), k=5)
    elif setup == TESTCOSINEKNN7:
        classifier = NearestNeighbor(dist_metric=CosineDistance(), k=7)
    elif setup == TESTSVM:
        classifier = SVM()
    else:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)


    # define the model as the combination
    model = PredictableModel(feature=feature, classifier=classifier)

    # remove some data from each class to test the model
    
    testdatacount = testdataperclass * classcount

    curraddindex = 0

    testxi = []
    
    for i in range(classcount):
        random.seed()
        allxi = range(samplecount)
        tempxi = random.sample(allxi, testdataperclass)
        tempxi = sorted(tempxi)
        for j in range(len(tempxi)):
            testxi.append(curraddindex + tempxi[j])

        curraddindex += samplecount
    
    testxicount = 0
    
    newdata = []
    newlabels = []
    
    testxilen = len(testxi)
    
    for i in range(len(data)):
        if testxicount < testxilen:
            if i != testxi[testxicount]:
                newdata.append(data[i])
                newlabels.append(labels[i])
            else:
                testxicount += 1
        else:
            newdata.append(data[i])
            newlabels.append(labels[i])
    
    # compute model
    model.compute(newdata, newlabels)

    # test model
    testr = []
    testc = []
    
    tcount = 0
    fcount = 0

    for i in range(testdatacount):
        testr.append(model.predict(data[testxi[i]]))
        if testr[i][0] == labels[testxi[i]]:
            testc.append(True)
            tcount +=  1
        else:
            testc.append(False)
            fcount += 1

    print
    print "Prediction result"
    print testr
    print
    print "Test correctness"
    print testc
    print
    print "False count"
    print fcount
    print
    
    #Eigenfaces is not relevant for spatial histogram
    # perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)

    # perform leave one out validation
    # cv = LeaveOneOutCrossValidation(model)
    cv.validate(data, np.asarray(labels))
    cv.print_results()
