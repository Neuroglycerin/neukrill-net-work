#!/usr/bin/env python

import skimage.io
import skimage.transform
import pickle
import sklearn
import numpy as np
import glob
import os
import neukrill_net.utils as utils
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.cross_validation
from sklearn.externals import joblib
import sklearn.metrics

def parse_train_data():
    """
    Parse training data and rescale
    """

    settings = utils.parse_settings('settings.json')

    # get all training file paths and class names
    image_fname_dict = gather_images(settings['train_data_dir'])

    # as images are different sizes rescale all images to 25x25 when reading into matrix
    train_data = []
    train_labels = []
    class_label_list = []
    class_index = 0
    for class_name in image_fname_dict.keys():
        print("class: {0} of 120: {1}".format(class_index, class_name))
        image_fpaths = image_fname_dict[class_name]
        num_image = len(image_fpaths)
        image_array = np.zeros((num_image, 625))

        class_label_list.append(class_name)
        for index in range(num_image):
            image = skimage.io.imread(image_fpaths[index])
            #image_ratio = get_minor_major_ratio(image)

            resized_image = skimage.transform.resize(image, (25,25))
            image_vector = resized_image.ravel()
            image_array[index,] = image_vector
            array_labels = num_image * [class_name]

        class_index += 1
        train_data.append(image_array)
        train_labels = train_labels + array_labels

    X_train = np.vstack(train_data)
    y_train = np.array(train_labels)

    print(class_label_list)

    return X_train, y_train, class_label_list

def gather_images(image_directory):
    """Taking a data directory, will gather all files images and 
    return a dictionary of class_name: image_names"""
    image_fname_dict = {}
    # loop over the images, updating dictionary
    for name in glob.glob(os.path.join(image_directory, '*', '')):
        split_name = name.split('/')
        class_name = split_name[-2]
        image_names = glob.glob(os.path.join(name, '*.jpg'))
        image_fname_dict.update({class_name: image_names})

    return image_fname_dict

def main():
    X, y, class_label_list = parse_train_data()

    with open('class_labels.pkl', 'wb') as labels_fh:
        pickle.dump(class_label_list, labels_fh)

    label_encoder = sklearn.preprocessing.LabelEncoder()

    y = label_encoder.fit_transform(y)

    #clf = sklearn.linear_model.SGDClassifier(n_jobs=-1,
    #                                         loss='log')

    clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1,
                                                  n_estimators=100,
                                                  verbose=1)

    cv = sklearn.cross_validation.StratifiedShuffleSplit(y)

    results = []
    for train, test in cv:
        clf.fit(X[train], y[train])
        p = clf.predict_proba(X[test])
        results.append(sklearn.metrics.log_loss(y[test], p))

    print(results)

    joblib.dump(clf, 'model.pkl', compress=3)

if __name__=='__main__':
    main()
