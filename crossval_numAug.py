#!/usr/bin/env python

import sys
import numpy as np
import sklearn
import neukrill_net.utils
import neukrill_net.highlevelfeatures
import neukrill_net.stacked
import time
from sklearn.externals import joblib
import sklearn.ensemble
import sklearn.pipeline
import sklearn.feature_selection
import sklearn.grid_search



def check_score(cache_path, clf, settings, train_split=0.8):
    
    X = joblib.load(cache_path)
    X_paths,y = settings.flattened_train_paths(settings.classes)
    y = np.array(y)
    
    n_augments = X.shape[0]
    
    li_train = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'train', train_split, classes=settings.classes)
    li_validate = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'validation', train_split, classes=settings.classes)
    
    X_train = X[:,li_train,:]
    X_validate = X[:,li_validate,:]
    y_train = y[li_train]
    y_validate = y[li_validate]
    XX_train = X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
    XX_validate = X_validate.reshape((X_validate.shape[0]*X_validate.shape[1],X_validate.shape[2]))
    yy_train = np.tile(y_train, n_augments)
    yy_validate = np.tile(y_validate, n_augments)
    
    pcfilter = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif, percentile=95)
    
    XX_train = pcfilter.fit_transform(XX_train, yy_train)
    XX_validate = pcfilter.transform(XX_validate)
    
    clf.fit(XX_train,yy_train)
    
    p = clf.predict_proba(XX_validate)
    p = np.reshape(p, (X_validate.shape[0], X_validate.shape[1], p.shape[1]))
    
    p_avg = p.mean(0)
    
    nll = sklearn.metrics.log_loss(y_validate, p_avg)
    
    return nll


cache_paths = ['/disk/data1/s1145806/cached_hlf_train_data_raw_ranged.pkl',
               '/disk/data1/s1145806/cached_hlf_train3_data_raw_ranged.pkl',
               '/disk/data1/s1145806/cached_hlf_train6_data_raw_ranged.pkl',
               '/disk/data1/s1145806/cached_hlf_train10_data_raw_ranged.pkl',
               '/disk/data1/s1145806/cached_hlf_train15_data_raw_ranged.pkl',
               '/disk/data1/s1145806/cached_hlf_train15alt_data_raw_ranged.pkl',
               '/disk/data1/s1145806/cached_hlf_train30_data_raw_ranged.pkl']

settings = neukrill_net.utils.Settings('settings.json')
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, max_depth=25, min_samples_leaf=3, n_jobs=16, random_state=42)

for path in cache_paths:
    print path
    nll = check_score(path, clf, settings)
    print "scored {}".format(nll)

