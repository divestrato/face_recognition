import sys
# append tinyfacelib to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacelib modules
from util import read_images
from model import GaborFilterModel

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "USAGE: example_model_eigenfaces.py </path/to/images>"
        sys.exit()

    # read images
    [X,y] = read_images(sys.argv[1])
    # compute the eigenfaces model
    model = GaborFilterModel(X[1:], y[1:])
    # get a prediction for the first observation
    print "expected =", y[0], "/", "predicted =", model.predict(X[0])
