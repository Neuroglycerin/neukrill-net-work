#!/usr/bin/env python

from __future__ import print_function

import pickle
import json
import numpy as np
import glob
import os
import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing
import neukrill_net.augment as augment

import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.dummy
from sklearn.externals import joblib
import sklearn.metrics
import sklearn.pipeline
import argparse

def main(run_settings_path):
    settings = utils.Settings('settings.json')
    with open(run_settings_path) as rf:
        run_settings = json.load(rf)
    # shoehorn run_settings filename into its own dictionary (for later)
    run_settings['filename'] = os.path.split(
                                        run_settings_path)[-1].split(".")[0]
    # also put the settings in there
    run_settings['settings'] = settings
    if run_settings['model type'] == 'sklearn':
        train_sklearn(run_settings)
    elif run_settings['model type'] == 'pylearn2':
        train_pylearn2(run_settings)
    else:
        raise NotImplementedError("Unsupported model type.")

def train_sklearn(run_settings):
    # unpack settings
    settings = run_settings['settings']

    # get all training file paths and class names
    image_fname_dict = settings.image_fnames

    # this should be parsed from json, but hardcoded for now
    augment_settings = run_settings["preprocessing"]

    # build processing function
    processing = augment.augmentation_wrapper(augment_settings)
    
    # load data as design matrix, applying processing function
    X, y = utils.load_data(image_fname_dict, classes=settings.classes,
                           processing=processing)

    # make a label encoder and encode the labels
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    if run_settings['classifier'] == 'dummy':
        # just a dummy uniform probability classifier for working purposes
        clf = sklearn.dummy.DummyClassifier(strategy='uniform')
    elif run_settings['classifier'] == 'logistic regression':
        clf = sklearn.linear_model.SGDClassifier(n_jobs=-1,
                                                 loss='log')
    elif run_settings['classifier'] == 'random forest':
        forest = sklearn.ensemble.RandomForestClassifier(n_jobs=-1,
                                                  n_estimators=100,
        #                                          verbose=1,
                                                  max_depth=5)
        scaler = sklearn.preprocessing.StandardScaler()
        clf = sklearn.pipeline.Pipeline((("scl",scaler),("clf",forest)))

    # only supporting stratified shuffle split for now
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y,
                                    **run_settings['cross validation'])

    results = []
    for train, test in cv:
        clf.fit(X[train], y[train])
        p = clf.predict_proba(X[test])
        results.append(sklearn.metrics.log_loss(y[test], p))

    print(results)

    # before saving model check there is somewhere for it to save to
    modeldir = os.path.join(settings.data_dir,"models")
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    # save the model in the data directory, in a "models" subdirectory
    # with the name of the run_settings as the name of the pkl
    picklefname = os.path.join(modeldir,run_settings['filename']+".pkl")
    joblib.dump(clf, picklefname, compress=3)

def train_pylearn2(run_settings):
    """
    Function to call operations for running a pylearn2 model using
    the settings found in run_settings.
    """
    # NOT IMPLEMENTED YET

if __name__=='__main__':
    # need to argparse for run settings path
    parser = argparse.ArgumentParser(description='Train a model and store a'
                                                 'pickled model file.')
    # nargs='?' will look for a single argument but failover to default
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
            nargs='?', default=os.path.join("run_settings","default.json"),
            help="Path to run settings json file.")
    args = parser.parse_args()
    main(args.run_settings)
