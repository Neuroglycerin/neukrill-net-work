#!/usr/bin/env python

from __future__ import print_function

import pickle
import sklearn
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

def main():

    # this should be parsed from json, but hardcoded for now
    # augment_settings = {'resize':(48,48),'rotate':4}
    augment_settings = {'resize':(48,48)}
    
    # Get global settings
    settings = utils.Settings('settings.json')
    
    # Make the processing wrapper function
    processing = augment.augmentation_wrapper(augment_settings)

    # Load the training data, with the processing applied
    X, y = utils.load_data(settings.image_fnames, classes=settings.classes,
                           processing=processing)
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    
    y = label_encoder.fit_transform(y)
    
    # just a dummy uniform probability classifier for working purposes
    clf = sklearn.dummy.DummyClassifier(strategy='uniform')
    
    #clf = sklearn.linear_model.SGDClassifier(n_jobs=-1,
    #                                         loss='log')
    #forest = sklearn.ensemble.RandomForestClassifier(n_jobs=-1,
    #                                              n_estimators=100,
    #                                              verbose=1,
    #                                              max_depth=5)
    #scaler = sklearn.preprocessing.StandardScaler()
    #clf = sklearn.pipeline.Pipeline((("scl",scaler),("clf",forest)))
    
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y)
    
    results = []
    for train, test in cv:
        clf.fit(X[train], y[train])
        p = clf.predict_proba(X[test])
        results.append(sklearn.metrics.log_loss(y[test], p))
    
    print(results)
    print('CV average = {}'.format(np.mean(results)))
    
    joblib.dump(clf, 'model.pkl', compress=3)

if __name__=='__main__':
    main()
