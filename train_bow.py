#!/usr/bin/env python

from __future__ import print_function

import pickle
import sklearn
import numpy as np
import glob
import os
import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing
import neukrill_net.highlevelfeatures as highlevelfeatures

import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.dummy
from sklearn.externals import joblib
import sklearn.metrics

def main():
    
    # this should be parsed from json, but hardcoded for now
    bow_options = { 'verbose': True,
                    'normalise_hist': False,
                    'n_features_max': 100,
                    'patch_size': 15,
                    'clusteralgo': 'kmeans',
                    'n_clusters': 20,
                    'random_seed': 42
                  }
    
    # Load the settings, providing 
    settings = utils.Settings('settings.json')
    
    # Load the raw data
    print('Loading the raw training data')
    rawdata, labels = utils.load_rawdata(settings.image_fnames, classes=settings.classes)
    
    # Encode the labels
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Probably not the best classifier
    clf = sklearn.linear_model.LogisticRegression()
    
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y)
    
    bow = highlevelfeatures.BagOfWords(**bow_options)
    sample = np.random.random_integers(0, len(rawdata)-1, size=(1000)) # Subsample so we can do this in sensible time
    bow.fit([rawdata[i] for i in sample])
    #bow.fit(rawdata)
    print('Bagging words for raw training data')
    X = bow.extractfeatures(rawdata)
    X = np.squeeze(X)
    
    # Try cross-validating
    print('Cross-validating')
    results = []
    for train, test in cv:
        # Make a new BOW encoding
        #bow = highlevelfeatures.BagOfWords(**bow_options)
        #bow.fit([rawdata[i] for i in train])
        #X = bow.extractfeatures(rawdata)
        
        clf.fit(X[train,:], y[train])
        p = clf.predict_proba(X[test])
        res = sklearn.metrics.log_loss(y[test], p)
        print(res)
        results.append(res)
    
    print(results)
    print('CV average = {}'.format(np.mean(results)))
    
    # Train on the whole thing and save model for later
    #bow = highlevelfeatures.BagOfWords(**bow_options)
    #bow.fit(rawdata)
    #X = bow.extractfeatures(rawdata)
    
    clf.fit(X,y)
    
    print('Loading the raw test data')
    rawtest, names = utils.load_rawdata(settings.image_fnames)
    print('Bagging words for raw test data')
    X2 = X = bow.extractfeatures(rawdata)
    X2 = np.squeeze(X2)
    
    p = clf.predict_proba(X2)
    
    utils.write_predictions('submission_bow_initial.csv', p, names, settings.classes)
    

if __name__=='__main__':
    main()
