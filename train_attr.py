#!/usr/bin/env python

from __future__ import print_function

import pickle
import sklearn
import numpy as np
import glob
import os
import neukrill_net.utils as utils
import neukrill_net.image_processing as image_processing

import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.dummy
from sklearn.externals import joblib
import sklearn.metrics

def main():
    
    settings = utils.Settings('settings.json')
    
    # this should be parsed from json, but hardcoded for now
    attributes_settings = ['width','height']
    
    processing = image_processing.attributes_wrapper(attributes_settings)
    
    X, y = utils.load_data(settings.image_fnames, classes=settings.classes,
                           processing=processing)
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    
    y = label_encoder.fit_transform(y)
    
    # just a dummy uniform probability classifier for working purposes
    #clf = sklearn.dummy.DummyClassifier(strategy='uniform')
    
    #clf = sklearn.linear_model.SGDClassifier(n_jobs=-1,
    #                                         loss='log')
    #clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1,
    #                                              n_estimators=100,
    #                                              verbose=1)
    
    # clf = sklearn.svm.SVC(probability=True)
    
    clf = sklearn.linear_model.LogisticRegression()
    
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y)
    
    # Try cross-validating
    results = []
    for train, test in cv:
        clf.fit(X[train], y[train])
        p = clf.predict_proba(X[test])
        results.append(sklearn.metrics.log_loss(y[test], p))
    
    print(results)
    print('CV average = {}'.format(np.mean(results)))
    
    # Train on the whole thing and save model for later
    clf.fit(X,y)
    
    joblib.dump(clf, 'imsizeLR.pkl', compress=3)

if __name__=='__main__':
    main()
