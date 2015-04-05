import sys
import json
import argparse
# append facelib to module search path
sys.path.append("..")
import matplotlib.pyplot as plt
from facelib.dataset import DataSet
from facelib.feature import PCA, Fisherfaces, SpatialHistogram, Gaborfilter
from facelib.distance import EuclideanDistance, CosineDistance
from facelib.svm import grid_search
from facelib.classifier import NearestNeighbor
from facelib.classifier import SVM
from facelib.model import PredictableModel
from facelib.validation import KFoldCrossValidation, LeaveOneOutCrossValidation
from facelib.visual import subplot, plot_gray
from facelib.util import minmax_normalize, read_images
from svmutil import *

import numpy as np
# import matplotlib colormaps
import matplotlib.cm as cm
# import for logging
import logging, sys

parser = argparse.ArgumentParser(description='Face recognition. Default is PCA with K-Nearest Euclidian (k=1), Validated with K-Fold (k=10)')
feature_group = parser.add_mutually_exclusive_group()
feature_group.add_argument('--pca', action='store_true', help='use PCA feature extraction')
feature_group.add_argument('--fis', action='store_true', help='use Fisherfaces feature extraction')
feature_group.add_argument('--sph', action='store_true', help='use Spatial Histogram feature extraction')
feature_group.add_argument('--gbr', action='store_true', help='use Gabor Filter feature extraction')

classifier_group = parser.add_mutually_exclusive_group()
classifier_group.add_argument('--svm', nargs='+', help='use SVM classifier')
classifier_group.add_argument('--train', action='store_true', help='train SVM classifier')
classifier_group.add_argument('--kne', type=int, choices=[1, 3, 5, 7], default=0, help='use K-Nearest Neighbor Euclidian classifier')
classifier_group.add_argument('--knc', type=int, choices=[1, 3, 5, 7], default=0, help='use K-Nearest Neighbor Cosine classifier')

validation_group = parser.add_mutually_exclusive_group()
validation_group.add_argument('--kcv', type=int, choices=[5, 10], default=10, help='use K-Fold Cross Validaton')
validation_group.add_argument('--loo', action='store_true', help='use Leave One Out Cross Validaton')

parser.add_argument("dir", type=str, help="Directory of the face database")
parser.add_argument("--debug", action='store_true', help="Enable DEBUG output")
parser.add_argument("--roc", action='store_true', help="Show ROC output")
parser.add_argument("--eigen", action='store_true', help="Show Eigenvector output")

args = parser.parse_args()

# load a dataset
[data, labels] = read_images(args.dir)

# set up a handler for logging
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('TEAM 5 - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add handler to facelib modules
logger = logging.getLogger("facelib")
logger.addHandler(handler)

if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# define feature extraction method
if args.pca:
    feature = PCA()
elif args.fis:
    feature = Fisherfaces()
elif args.sph:
    feature = SpatialHistogram()
elif args.gbr:
    feature = Gaborfilter()
else:
    feature = PCA()

# define classifier
if args.svm and len(args.svm) >= 1:
    best_parameter = svm_parameter("-q")
    best_parameter.C = float(args.svm[0])
    best_parameter.gamma = float(args.svm[1])
    classifier = SVM(param=best_parameter)
elif args.train:
    classifier = SVM()
    model = PredictableModel(feature=feature, classifier=classifier)
    [best_parameter, results] = grid_search(model, data, labels)
    results.sort(key=lambda x: int(x[2]))
    logger.info("Training result C: %.2f, gamma: %2.f, accuracy: %.2f" % (results[-1][0], results[-1][1], results[-1][2]))
    print (json.dumps(results, indent=4))
    sys.exit()
elif args.kne:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=args.kne)
elif args.knc:
    classifier = NearestNeighbor(dist_metric=CosineDistance(), k=args.kne)
else:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)

# define the model as the combination
model = PredictableModel(feature=feature, classifier=classifier)
model.compute(data, labels)

# define extra outputs
if args.eigen:
    # turn the first (at most) 16 eigenvectors into grayscale images
    E = []
    for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
        e = model.feature.eigenvectors[:,i].reshape(data[0].shape)
        E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    subplot(title="Eigenvectors", images=E, rows=4, cols=4, colormap=cm.gray)

if args.roc:
    # TODO: output ROC
    pass

# define cross validation
if args.kcv:
    cv = KFoldCrossValidation(model, k=args.kcv)
    cv.validate(data, labels, description="TEAM 5 Demo")
elif args.loo:
    cv = LeaveOneOutCrossValidation(model)
    cv.validate(data, np.asarray(labels), description="TEAM 5 Demo")
else:
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(data, labels, description="TEAM 5 Demo")

cv.print_results()
