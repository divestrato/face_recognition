import sys
# append tinyfacelib to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacelib modules
from subspace import gabor
from util import normalize, asRowMatrix, read_images
from visual import subplot

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "USAGE: example_eigenfaces.py </path/to/images>"
        sys.exit()

    # read images
    [X,y] = read_images(sys.argv[1])

    # perform a gabor filter
    [D, W, mu] = gabor(asRowMatrix(X), y)

    import matplotlib.cm as cm

    # turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in xrange(min(len(X), 16)):
	    e = W[:,i].reshape(X[0].shape)
	    E.append(normalize(e,0,255))
    # plot them and store the plot to "python_eigenfaces.pdf"
    subplot(title="Gaborfilter AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Gaborfilter", colormap=cm.jet, filename="python_gaborfilter.png")

    from subspace import project, reconstruct

    # reconstruction steps
    steps=[i for i in xrange(10, min(len(X), 320), 20)]
    E = []
    for i in xrange(min(len(steps), 16)):
	    numEvs = steps[i]
	    P = project(W[:,0:numEvs], X[0].reshape(1,-1), mu)
	    R = reconstruct(W[:,0:numEvs], P, mu)
	    # reshape and append to plots
	    R = R.reshape(X[0].shape)
	    E.append(normalize(R,0,255))
    # plot them and store the plot to "python_reconstruction.pdf"
    subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Gaborfilter", sptitles=steps, colormap=cm.gray, filename="python_gaborfilter_reconstruction.png")
