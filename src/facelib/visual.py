from facelib.normalization import minmax

import os as os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import math as math
import random


def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname, 'fontsize':fontsize }

def plot_gray(X,  sz=None, filename=None):
    if not sz is None:
        X = X.reshape(sz)
    X = minmax(I, 0, 255)
    fig = plt.figure()
    implot = plt.imshow(np.asarray(Ig), cmap=cm.gray)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, format="png", transparent=False)

def plot_eigenvectors(eigenvectors, num_components, sz, filename=None, start_component=0, rows = None, cols = None, title="Subplot", color=True):
        if (rows is None) or (cols is None):
            rows = cols = int(math.ceil(np.sqrt(num_components)))
        num_components = np.min(num_components, eigenvectors.shape[1])
        fig = plt.figure()
        for i in range(start_component, num_components):
            vi = eigenvectors[0:,i].copy()
            vi = minmax(np.asarray(vi), 0, 255, dtype=np.uint8)
            vi = vi.reshape(sz)

            ax0 = fig.add_subplot(rows,cols,(i-start_component)+1)

            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax0.get_yticklabels(), visible=False)
            plt.title("%s #%d" % (title, i), create_font('Tahoma',10))
            if color:
                implot = plt.imshow(np.asarray(vi))
            else:
                implot = plt.imshow(np.asarray(vi), cmap=cm.grey)
        if filename is None:
            fig.show()
        else:
            fig.savefig(filename, format="png", transparent=False)

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

def plot_roc(model, data, labels):
    roc = []
    for current_roctestlabel in range(4):
        testdataperclass = 5

        checklabels = []

        all1 = []
        all2 = []

        for i in range(len(data)):
            if(labels[i] == current_roctestlabel):
                checklabels.append(1)
                all1.append(i)
            else:
                checklabels.append(2)
                all2.append(i)

        testxi = []

        random.seed()
        testxi.extend(random.sample(all1, testdataperclass))
        testxi.extend(random.sample(all2, testdataperclass))

        testxi = sorted(testxi)

        testxicount = 0

        newdata = []
        newlabels = []

        testxilen = len(testxi)

        for i in range(len(data)):
            if testxicount < testxilen:
                if i != testxi[testxicount]:
                    newdata.append(data[i])
                    newlabels.append(checklabels[i])
                else:
                    testxicount += 1
            else:
                newdata.append(data[i])
                newlabels.append(checklabels[i])

        # compute model
        model.compute(newdata, newlabels)

        roc_distance = []
        roc_predictedclass = []
        roc_trueclass = []

        for i in range(testxilen):
            #ires = model.predict(data[testxi[i]])
            idis1 = model.distance(data[testxi[i]], 1)

            icls = checklabels[testxi[i]]

            roc_distance.append(idis1)
            roc_trueclass.append(icls)

        # compute data for roc

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        doublecount = testdataperclass * 2

        rocx = []
        rocy = []

        sorted_index = np.argsort(np.asarray(roc_distance))
        for i in range(doublecount-1, -1, -1):
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            curr_tresh = roc_distance[sorted_index[i]]

            for j in range(doublecount - 1, -1, -1):
                if(roc_distance[sorted_index[j]] <= curr_tresh):
                    if roc_trueclass[sorted_index[j]] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if roc_trueclass[sorted_index[j]] == 1:
                        fn += 1
                    else:
                        tn += 1

            tpr = float(tp) / (tp + fn)
            fpr = float(fp) / (fp + tn)

            rocx.append(fpr)
            rocy.append(tpr)

        rocx.append(0)
        rocy.append(0)
        roc.append([rocx, rocy])

    #Interpolate points over 100 data
    x_space = np.linspace(float(0),float(1),100)
    interpolated = [np.interp(x_space,sorted(d[0]),sorted(d[1])) for d in roc]

    #Average the y coordinates
    y_points = [np.average(x) for x in zip(*interpolated)]
    x_points = [x for x in x_space]
    x_points = [0.0] + x_points
    y_points = [0.0] + y_points

    plt.plot(x_points, y_points)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.axis([0, 1, 0, 1.1])
    plt.show()


# using plt plot:
#filename="/home/philipp/facelib/at_database_vs_accuracy_xy.png"
#t = np.arange(2., 10., 1.)
#fig = plt.figure()
#plt.plot(t, r0, 'k--', t, r1, 'k')
#plt.legend(("Eigenfaces", "Fisherfaces"), 'lower right', shadow=True, fancybox=True)
#plt.ylim(0,1)
#plt.ylabel('Recognition Rate')
#plt.xlabel('Database Size (Images per Person)')
#fig.savefig(filename, format="png", transparent=False)
#plt.show()
