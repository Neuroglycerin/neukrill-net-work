#!/usr/bin/env python

import os
import sys
import numpy as np
import sklearn
import neukrill_net.utils
import neukrill_net.highlevelfeatures
import neukrill_net.stacked
import time
from sklearn.externals import joblib
import sklearn.ensemble
import sklearn.feature_selection


def predict(cache_paths, out_fname, clf, settings, generate_heldout=True):
    
    X_train = joblib.load(cache_paths[0])
    X_test = joblib.load(cache_paths[1])
    
    X_test[np.isnan(X_test)] = 0
    
    X_paths,y = settings.flattened_train_paths(settings.classes)
    y_train = np.array(y)
    
    n_augments = X_train.shape[0]
    
    XX_train = X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
    XX_test = X_test.reshape((X_test.shape[0]*X_test.shape[1],X_test.shape[2]))
    yy_train = np.tile(y_train, n_augments)
    
    pcfilter = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif, percentile=95)
    
    XX_train = pcfilter.fit_transform(XX_train, yy_train)
    XX_test  = pcfilter.transform(XX_test)
    
    clf.fit(XX_train,yy_train)
    
    p = clf.predict_proba(XX_test)
    p = np.reshape(p, (X_test.shape[0], X_test.shape[1], p.shape[1]))
    
    p_avg = p.mean(0)
    
    names = [os.path.basename(path) for path in settings.image_fnames['test']]
    
    neukrill_net.utils.write_predictions(out_fname, p_avg, names, settings.classes)
    
    if not generate_heldout:
        return
    
    li_test = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'test', train_split=0.8, classes=settings.classes)
    li_nottest = np.logical_not(li_test)
    
    X2_train = X_train[:,li_nottest,:]
    X2_test = X_train[:,li_test,:]
    
    XX_train = X2_train.reshape((X2_train.shape[0]*X2_train.shape[1],X2_train.shape[2]))
    XX_test = X2_test.reshape((X2_test.shape[0]*X2_test.shape[1],X2_test.shape[2]))
    yy_train = np.tile(y_train[li_nottest], n_augments)
    yy_test = y_train[li_nottest]
    
    XX_train = pcfilter.transform(XX_train)
    XX_test  = pcfilter.transform(XX_test)
    
    clf.fit(XX_train,yy_train)
    
    p = clf.predict_proba(XX_test)
    p = np.reshape(p, (X_test.shape[0], X_test.shape[1], p.shape[1]))
    
    p_avg = p.mean(0)
    
    joblib.dump( (p_avg, yy_test), out_fname + '_heldout.pkl', )
    


cache_paths = [('/disk/data1/s1145806/cached_hlf_train_data_raw_ranged.pkl'     , '/disk/data1/s1145806/cached_hlf_test_data_raw_ranged.pkl'     ),
               ('/disk/data1/s1145806/cached_hlf_train3_data_raw_ranged.pkl'    , '/disk/data1/s1145806/cached_hlf_test3_data_raw_ranged.pkl'    ),
               ('/disk/data1/s1145806/cached_hlf_train6_data_raw_ranged.pkl'    , '/disk/data1/s1145806/cached_hlf_test6_data_raw_ranged.pkl'    ),
               ('/disk/data1/s1145806/cached_hlf_train10_data_raw_ranged.pkl'   , '/disk/data1/s1145806/cached_hlf_test10_data_raw_ranged.pkl'   ),
               ('/disk/data1/s1145806/cached_hlf_train15_data_raw_ranged.pkl'   , '/disk/data1/s1145806/cached_hlf_test15_data_raw_ranged.pkl'   ),
               ('/disk/data1/s1145806/cached_hlf_train15alt_data_raw_ranged.pkl', '/disk/data1/s1145806/cached_hlf_test15alt_data_raw_ranged.pkl'),
               ('/disk/data1/s1145806/cached_hlf_train30_data_raw_ranged.pkl'   , '/disk/data1/s1145806/cached_hlf_test30_data_raw_ranged.pkl'   )]

n_trees = 1500
max_depth = 20

settings = neukrill_net.utils.Settings('settings.json')
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=3, n_jobs=16, random_state=42)

for pathpair in cache_paths:
    print pathpair
    out_fname = pathpair[0][:-4] + "{}trees_{}deep".format(n_trees,max_depth) + '_predictions.csv'
    predict(pathpair, out_fname, clf, settings)

