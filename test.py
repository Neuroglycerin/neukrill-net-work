#!/usr/bin/env python

import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing
import neukrill_net.augment as augment

import csv
import pickle
from sklearn.externals import joblib
import numpy as np
import glob
import os

def main(run_settings_path):
    # this should just run either function depending on the run settings
    settings = utils.Settings('settings.json')
    run_settings = utils.load_run_settings(run_settings_path, 
            settings_path='settings.json')

    # move into it's own function
    # HELLO BOILERPLATE
    if run_settings['model type'] == 'sklearn':
        train_sklearn(run_settings)
    elif run_settings['model type'] == 'pylearn2':
        #train_pylearn2(run_settings)
        raise NotImplementedError("Unsupported model type.")
    else:
        raise NotImplementedError("Unsupported model type.")


def train_sklearn(run_settings):
    # some more boilerplate here
    # unpack settings
    settings = run_settings['settings']

    # get all training file paths and class names
    image_fname_dict = settings.image_fnames

    # parsed from json, preproc settings are dict
    augment_settings = run_settings["preprocessing"]
    processing = augment.augmentation_wrapper(augment_settings)

    image_fname_dict = settings.image_fnames

    X, names = utils.load_data(image_fname_dict, processing=processing,
                               verbose=True)
    
    # load the model from where it's _expected_ to be saved
    clf = joblib.load(run_settings['pickle abspath'])
    p = clf.predict_proba(X)
    
    utils.write_predictions('submission.csv', p, names, settings)

if __name__ == '__main__':
    # copied code from train.py here instead of making a function
    # because this code may diverge
    # need to argparse for run settings path
    parser = argparse.ArgumentParser(description='Train a model and store a'
                                                 'pickled model file.')
    # nargs='?' will look for a single argument but failover to default
    parser.add_argument('run_settings', metavar='run_settings', type=str, 
            nargs='?', default=os.path.join("run_settings","default.json"),
            help="Path to run settings json file.")
    args = parser.parse_args()
    main(args.run_settings)
