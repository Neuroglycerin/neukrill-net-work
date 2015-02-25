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

def main(run_settings_path, verbose=False, force=False):
    # load the non-run-specific settings
    settings = utils.Settings('settings.json')
    # load the run-specific settings
    run_settings = utils.load_run_settings(run_settings_path, 
            settings,
            settings_path='settings.json', force=force)
    if run_settings['model type'] == 'sklearn':
        train_sklearn(run_settings, verbose=verbose, force=force)
    elif run_settings['model type'] == 'pylearn2':
        train_pylearn2(run_settings, verbose=verbose, force=force)
    else:
        raise NotImplementedError("Unsupported model type.")

def train_sklearn(run_settings, verbose=False, force=False):
    # unpack settings
    settings = run_settings['settings']

    # get all training file paths and class names
    image_fname_dict = settings.image_fnames

    # now being parsed from json
    augment_settings = run_settings["preprocessing"]

    # build processing function
    processing = augment.augmentation_wrapper(**augment_settings)
    
    # load data as design matrix, applying processing function
    X, y = utils.load_data(image_fname_dict, classes=settings.classes,
                           processing=processing, verbose=verbose)

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

    print("Average CV: {0} +/- {1}".format(np.mean(results),
                                    np.sqrt(np.var(results))))

    # save the model in the data directory, in a "models" subdirectory
    # with the name of the run_settings as the name of the pkl
    joblib.dump(clf, run_settings["pickle abspath"], compress=3)

    # store the raw log loss results back in the run settings json
    run_settings["crossval results"] = results
    # along with the other things we've added
    utils.save_run_settings(run_settings)

def train_pylearn2(run_settings, verbose=False, force=False):
    """
    Function to call operations for running a pylearn2 model using
    the settings found in run_settings.
    """
    import pylearn2.config
    # unpack settings
    settings = run_settings['settings']
    #read the YAML settings template 
    with open(os.path.join("yaml_templates",run_settings['yaml file'])) as y:
        yaml_string = y.read()
    # sub in the following things for default: settings_path, run_settings_path, 
    # final_shape, n_classes, save_path
    run_settings["n_classes"] = len(settings.classes)
    modeldir = os.path.join(settings.data_dir,"models")
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    run_settings["save_path"] = os.path.join(modeldir,
            run_settings['filename'] + ".pkl")
    # time for some crude string parsing
    yaml_string = yaml_string%(run_settings)
    # write the new yaml to the data directory, in a yaml_settings subdir
    yamldir = os.path.join(settings.data_dir,"yaml_settings")
    if not os.path.exists(yamldir):
        os.mkdir(yamldir)
    yaml_path = os.path.join(yamldir,run_settings["filename"]+
            run_settings['yaml file'].split(".")[0]+".yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_string)
    # then we load the yaml file using pylearn2
    train = pylearn2.config.yaml_parse.load(yaml_string)
    # and run the model!
    train.main_loop()
    import pdb
    pdb.set_trace()

if __name__=='__main__':
    # need to argparse for run settings path
    parser = argparse.ArgumentParser(description='Train a model and store a'
                                                 'pickled model file.')
    # nargs='?' will look for a single argument but failover to default
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
            nargs='?', default=os.path.join("run_settings","default.json"),
            help="Path to run settings json file.")
    # add force option 
    parser.add_argument('-f', action="store_true", help="Force overwrite of"
                        " model files/submission csvs/anything else.")
    # add verbose option
    parser.add_argument('-v', action="store_true", help="Run verbose.")

    args = parser.parse_args()
    main(args.run_settings,verbose=args.v,force=args.f)
