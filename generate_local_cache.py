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


# Define output path
train_pkl_path = '/disk/data1/s1145806/cached_kpec_train_data_raw.pkl'
test_pkl_path = '/disk/data1/s1145806/cached_kpec_test_data_raw.pkl'


t0 = time.time()

print "Setup..."

max_num_kp_orb = 200
max_num_kp_fast = 400
max_num_kp_mser = 200

settings = neukrill_net.utils.Settings('settings.json')
X,y = settings.flattened_train_paths(settings.classes)
X_test = settings.image_fnames['test']

X = np.array(X)
y = np.array(y)

detector_list = [lambda image: neukrill_net.image_features.get_ORB_keypoints(image, n=max_num_kp_orb, patchSize=9),
                 lambda image: neukrill_net.image_features.get_FAST_keypoints(image, n=max_num_kp_fast),
                 lambda image: neukrill_net.image_features.get_MSER_keypoints(image, n=max_num_kp_mser)]

describer_list = [neukrill_net.image_features.get_ORB_descriptions,
                  neukrill_net.image_features.get_ORB_descriptions,
                  neukrill_net.image_features.get_ORB_descriptions]

kprf_base = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, max_depth=15, min_samples_leaf=20, n_jobs=16, random_state=42)

hlf_list = []
for index,detector in enumerate(detector_list):
    hlf_list += [neukrill_net.highlevelfeatures.KeypointEnsembleClassifier(detector, describer_list[index], kprf_base,
                                                                     return_num_kp=True, n_jobs=0, verbosity=1, summary_method='vote')]

hlf = neukrill_net.highlevelfeatures.MultiHighLevelFeature(hlf_list)

# Partition the data

print "Partitioning the training data"

# Remove the data which is going to be held out
li_test = neukrill_net.utils.train_test_split_bool(settings.image_fnames, 'test', train_split=0.8, classes=settings.classes)
li_nottest = np.logical_not(li_test)
indices_nottest = np.where(li_nottest)[0]

# Split the remaining data
inner, outer = sklearn.cross_validation.train_test_split(indices_nottest, test_size=0.25, random_state=42)

print "Fitting keypoint predictions"
hlf.fit(X[inner], y[inner])

print "Transforming training data"
XF_train = hlf.transform(X)

print "Saving train cache"
joblib.dump(XF, train_pkl_path)

print "Transforming test data"
XF_test = hlf.transform(X_test)

print "Saving test cache"
joblib.dump(XF_test, test_pkl_path)

#joblib.dump(hlf, '/disk/data1/s1145806/kpec_cache/keypoint_transformer.pkl')

print "Took {} seconds".format(time.time() -t0)

