#!/usr/bin/env python

import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing

import csv
import pickle
from sklearn.externals import joblib
import numpy as np
import glob
import os

def main():
    out_fname = 'submission_imsizeLR.csv'
    settings = utils.Settings('settings.json')
    
    # this should be parsed from json, but hardcoded for now
    attributes_settings = ['width','height']
    
    processing = image_processing.attributes_wrapper(attributes_settings)
    
    X, names = utils.load_data(settings.image_fnames, processing=processing,
                               verbose=False)
    
    clf = joblib.load('imsizeLR.pkl')
    p = clf.predict_proba(X)
    
    utils.write_predictions(out_fname, p, names, settings)
    
if __name__ == '__main__':
    main()

