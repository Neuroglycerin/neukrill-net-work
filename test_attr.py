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
    # this should be parsed from json, but hardcoded for now
    #attributes_settings = ['width','height']
    #pkl_file = 'imsizeLR.pkl'
    #out_fname = 'submission_imsizeLR.csv'
    
    #attributes_settings = ['width','height']
    #pkl_file = 'imsizeSVM.pkl'
    #out_fname = 'submission_imsizeSVM.csv'
    
    #attributes_settings = ['numpixels','aspectratio']
    #pkl_file = 'imsizeLR_alt.pkl'
    #out_fname = 'submission_imsizeLR_alt.csv'
    
    attributes_settings = ['width','height','mean','stderr','propwhite','propbool','propblack']
    pkl_file = 'imattr1.pkl'
    out_fname = 'submission_imattr1.csv'
    
    # Get global settings, providing file names of test data
    settings = utils.Settings('settings.json')
    
    # Make the wrapper function
    processing = image_processing.attributes_wrapper(attributes_settings)
    
    # Load the test data, with the processing applied
    X, names = utils.load_data(settings.image_fnames, processing=processing,
                               verbose=False)
    
    clf = joblib.load(pkl_file)
    p = clf.predict_proba(X)
    
    utils.write_predictions(out_fname, p, names, settings.classes)
    
if __name__ == '__main__':
    main()

