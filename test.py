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

def main():
    settings = utils.Settings('settings.json')

    # this should be parsed from json, but hardcoded for now
    augment_settings = {'resize':(48,48)}
    processing = augment.augmentation_wrapper(augment_settings)

    image_fname_dict = settings.image_fnames

    X, names = utils.load_data(image_fname_dict, processing=processing,
                               verbose=True)
    
    clf = joblib.load('model.pkl')
    p = clf.predict_proba(X)
    
    utils.write_predictions('submission.csv', p, names, settings)
    

if __name__ == '__main__':
    main()

