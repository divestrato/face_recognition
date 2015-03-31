import sys
# append facelib to module search path
sys.path.append("..")

from facelib.dataset import DataSet
from facelib.feature import PCA
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

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "USAGE: eigenfaces.py </path/to/images>"
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
    feature = PCA()

    # define a 1-NN classifier with Euclidean Distance
    # classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # define a 1-NN classifier with Cosine Distance
    # classifier = NearestNeighbor(dist_metric=CosineDistance(), k=1)
    classifier = SVM()
    # define a 1-NN classifier with Normalized Correlation


    # define the model as the combination
    model = PredictableModel(feature=feature, classifier=classifier)

    # show eigenfaces
    model.compute(data, labels)

    # turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
        e = model.feature.eigenvectors[:,i].reshape(data[0].shape)
        E.append(minmax_normalize(e,0,255, dtype=np.uint8))

    # plot them and store the plot to "python_eigenfaces_eigenfaces.pdf"
    subplot(title="Eigenfaces", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="eigenfaces.pdf")

    # perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)

    # perform leave one out validation
    # cv = LeaveOneOutCrossValidation(model)
    cv.validate(data, np.asarray(labels))
    cv.print_results()
