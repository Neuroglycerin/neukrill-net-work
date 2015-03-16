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


def predict(training_cache_paths, testing_cache_paths, out_fname, clf, settings, is_subfitted=True, generate_predictions=2, generate_heldout=True):
    
    t0 = time.time()
    
    print 'loading data'
    
    X_train = [joblib.load(path) for path in training_cache_paths]
    X_test = [joblib.load(path) for path in testing_cache_paths]
    
    X_train = np.concatenate(X_train, 2)
    X_test = np.concatenate(X_test, 2)
    
    X_test[np.isnan(X_test)] = 0
    
    if is_subfitted:
        # Need to remove the relevant data
        # Remove the data which is going to be held out
        li_test = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'test', train_split=0.8, classes=settings.classes)
        li_nottest = np.logical_not(li_test)
        indices_nottest = np.where(li_nottest)[0]
        
        # Split the remaining data
        inner, outer = sklearn.cross_validation.train_test_split(indices_nottest, test_size=0.25, random_state=42)
        
        outer_full = np.concatenate( (np.where(li_test)[0],outer) )
        
        X_train = X_train[:,outer_full,:]
    
    
    X_paths,y = settings.flattened_train_paths(settings.classes)
    y_train = np.array(y)
    
    n_augments = X_train.shape[0]
    
    XX_train = X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
    XX_test = X_test.reshape((X_test.shape[0]*X_test.shape[1],X_test.shape[2]))
    yy_train = np.tile(y_train, n_augments)
    
    pcfilter = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif, percentile=95)
    
    XX_train = pcfilter.fit_transform(XX_train, yy_train)
    
    if generate_predictions:
        XX_test  = pcfilter.transform(XX_test)
        
        print '{}: training'.format(time.time()-t0)
        
        clf.fit(XX_train,yy_train)
        
        print '{}: predicting'.format(time.time()-t0)
        
        p = clf.predict_proba(XX_test)
        p = np.reshape(p, (X_test.shape[0], X_test.shape[1], p.shape[1]))
        
        print '{}: writing predictions to disk'.format(time.time()-t0)
        
        p_avg = p.mean(0)
        
        names = [os.path.basename(path) for path in settings.image_fnames['test']]
        
        neukrill_net.utils.write_predictions(out_fname, p_avg, names, settings.classes)

    if not generate_heldout:
        return
    
    print '{}: generating held out predictions'.format(time.time()-t0)
    
    if is_subfitted:
        X2_train = X_train[:,outer,:]
        X2_test = X_train[:,li_test,:]
        
    else:
        li_test = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'test', train_split=0.8, classes=settings.classes)
        li_nottest = np.logical_not(li_test)
        
        X2_train = X_train[:,li_nottest,:]
        X2_test = X_train[:,li_test,:]
    
    XX_train = X2_train.reshape((X2_train.shape[0]*X2_train.shape[1],X2_train.shape[2]))
    XX_test = X2_test.reshape((X2_test.shape[0]*X2_test.shape[1],X2_test.shape[2]))
    yy_train = np.tile(y_train[li_nottest], n_augments)
    yy_test = y_train[li_test]
    
    XX_train = pcfilter.transform(XX_train)
    XX_test  = pcfilter.transform(XX_test)
    
    print '{}: training without heldout'.format(time.time()-t0)
    
    clf.fit(XX_train,yy_train)
    
    print '{}: predicting on heldout'.format(time.time()-t0)
    
    p = clf.predict_proba(XX_test)
    p = np.reshape(p, (X2_test.shape[0], X2_test.shape[1], p.shape[1]))
    
    p_avg = p.mean(0)
    
    nll = sklearn.metrics.log_loss(yy_test, p_avg)
    
    print 'NLL score is {}'.format(nll)
    
    if generate_predictions<2:
        return
    
    print '{}: writing heldout to disk'.format(time.time()-t0)
    
    joblib.dump( (p_avg, yy_test), out_fname + '_heldout.pkl', )


n_trees = 500
max_depth = 10
n_jobs = 16

settings = neukrill_net.utils.Settings('settings.json')
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=3, n_jobs=n_jobs, random_state=42)

test_cache_paths = ['/disk/data1/s1145806/cached_kpecORB_quick_train_data_raw.pkl']
train_cache_paths = ['/disk/data1/s1145806/cached_kpecORB_quick_test_data_raw.pkl']
out_fname = 'cached_kpecORB_' + "{}trees_{}deep".format(n_trees,max_depth) + '_predictions.csv'

predict(pathpair, out_fname, clf, settings, is_subfitted=False, generate_predictions=True)

