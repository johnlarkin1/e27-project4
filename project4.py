# Project 4 for Computer Vision
# Authors: John Larkin and Tom Wilmots
# Date: 04/11/17
# Project: 
#   Train on the MNIST data set using different software and two new machine learning techniques 

import numpy as np
import gzip
import struct
import cv2
import sklearn
import tensorflow as tf
import sklearn.ensemble as skl_ensemble
import sklearn.model_selection as skl_model_select
import time

IMAGE_SIZE = 28

######################################################################
# Read a 32-bit int from a file or a stream

def read_int(f):
    buf = f.read(4)
    data = struct.unpack('>i', buf)
    return data[0]

######################################################################
# Open a regular file or a gzipped file to decompress on-the-fly

def open_maybe_gz(filename, mode='rb'):

    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

######################################################################
# Read the MNIST data from an images file or a labels file. The file
# formats are documented at http://yann.lecun.com/exdb/mnist/
# written by Matt Zucker

def read_mnist(images_file, labels_file):

    images = open_maybe_gz(images_file)

    imagic = read_int(images)
    assert(imagic == 2051)
    icount = read_int(images)
    rows = read_int(images)
    cols = read_int(images)
    assert(rows == IMAGE_SIZE and cols == IMAGE_SIZE)

    print 'reading', icount, 'images of', rows, 'rows by', cols, 'cols.'

    labels = open_maybe_gz(labels_file)

    lmagic = read_int(labels)
    assert(lmagic == 2049)
    lcount = read_int(labels)

    print 'reading', lcount, 'labels.'

    assert(icount == lcount)

    image_array = np.fromstring(images.read(icount*rows*cols),
                                dtype=np.uint8).reshape((icount,rows,cols))

    label_array = np.fromstring(labels.read(lcount),
                                dtype=np.uint8).reshape((icount))
    
    return image_array, label_array

######################################################################
# Recenter to make the data even better:

def recenter_and_get_data(image_file_name, label_file_name):

    # Read images and labels. This is reading the 10k-element test set
    # (you can also use the other pair of filenames to get the
    # 60k-element training set).
    images, labels = read_mnist(image_file_name,
                                label_file_name)


    # This is a nice way to reshape and rescale the MNIST data
    # (e.g. to feed to PCA, Neural Net, etc.) It converts the data to
    # 32-bit floating point, and then recenters it to be in the [-1,
    # 1] range.
    classifier_input = images.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    classifier_input = classifier_input * (2.0/255.0) - 1.0

    ##################################################
    # Now just display some stuff:
    
    print 'images has datatype {}, shape {}, and ranges from {} to {}'.format(
        images.dtype, images.shape, images.min(), images.max())

    print 'input has datatype {}, shape {}, and ranges from {} to {}'.format(
        classifier_input.dtype, classifier_input.shape,
        classifier_input.min(), classifier_input.max())

    return classifier_input, labels

######################################################################
# Classify using adaboost from sklearn 

def train_with_adaboost(image_array, label_array):
    # Documentation:
    # AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    # base_estimator - this is what the ensemble is built off of
    # n_estimators - the max number of estimators at which boosting is terminated
    #                if it's a perfect fit, then we stop early
    classifier = skl_ensemble.AdaBoostClassifier(base_estimator = None, n_estimators = 100)
    start = time.time()
    print 'Please wait. The AdaBoost classifier is being trained...'
    classifier.fit(image_array, label_array)
    end = time.time()
    total_time = end - start
    print 'Classifier has been trained in time: {0:.3f} seconds'.format(total_time)
    return classifier

def test_with_adaboost(clf, image_array, label_array):
    # Going to test and see how we do
    scores = skl_model_select.cross_val_score(clf, image_array, label_array)
    print 'Average score: {0:.3f}'.format(scores.mean())


def main():
    training_images_file = 'MNIST_data/train-images-idx3-ubyte.gz' 
    training_labels_file = 'MNIST_data/train-labels-idx1-ubyte.gz'
    image_array, label_array = recenter_and_get_data(training_images_file, training_labels_file)
    clf = train_with_adaboost(image_array, label_array)

    # Let's try prediction
    testing_images_file = 'MNIST_data/t10k-images-idx3-ubyte.gz'
    testing_labels_file = 'MNIST_data/t10k-labels-idx1-ubyte.gz'
    image_array, label_array = recenter_and_get_data(testing_images_file, testing_labels_file)
    test_with_adaboost(clf, image_array, label_array)


######################################################################

if __name__ == '__main__':
    main()
